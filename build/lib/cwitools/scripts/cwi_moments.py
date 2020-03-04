from astropy.io import fits
from cwitools import coordinates, imaging, variance, kinematics

import argparse
import numpy as np
import os

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Make maps of the first and second velocity moments of a 3D object.')
parser.add_argument('cube',
                    type=str,
                    metavar='cube',
                    help='The input data cube.'
)
parser.add_argument('-var',
                    type=str,
                    metavar='path',
                    help='Variance cube, to apply inverse variance weighting.',
)
parser.add_argument('-obj',
                    type=str,
                    metavar='path',
                    help='Object Mask cube.',
)
parser.add_argument('-id',
                    type=str,
                    metavar='str',
                    help='The ID of the object to use. Use -1 for all objects. Can also provide multiple as comma-separated list.',
                    default='-1'
)
parser.add_argument('-rsmooth',
                    type=float,
                    help='Smooth spatial axes before calculating moments (FWHM).',
                    default=None
)
parser.add_argument('-wsmooth',
                    type=float,
                    help='Smooth wavelength axis before calculating moments (FWHM).',
                    default=None
)
parser.add_argument('-method',
                    type=str,
                    help="Method to use for calculating moments. 'basic':use all input data, 'positive': apply simple positive threshold, 'closing-window':iterative method for noisy data.",
                    choices=['basic', 'positive', 'closing-window'],
                    default='closing-window'
)
parser.add_argument('-filltype',
                    type=str,
                    help="Fill type for empty or bad spaxels.",
                    choices=['nan', 'value'],
                    default='nan'
)
parser.add_argument('-fillvalue',
                    type=str,
                    help="Fill value for empty or bad spaxels (if -filltype = value).",
                    default=-9999
)
parser.add_argument('-mode',
                    type=str,
                    help="Output mode for units of moment maps. ('wav'=input wavelength units, 'vel'=km/s from flux-weighted center.)",
                    choices=['vel','wav'],
                    default='wav'
)
args = parser.parse_args()

#Try to load the fits file
if os.path.isfile(args.cube):
    input_fits = fits.open(args.cube)
else: raise FileNotFoundError(args.cube)

#Extract useful stuff and create useful data structures
cube  = input_fits[0].data.copy()

#Try to load the fits file
if args.var!=None:
    if os.path.isfile(args.var):
        var_fits = fits.open(args.var)
        var_cube = var_fits[0].data
        var_cube[var_cube <= 0] = np.inf

    else: raise FileNotFoundError(args.var)


else:
    print("No variance input given. Variance will be estimated.")
    var_cube = variance.estimate_variance(input_fits)

w,y,x = cube.shape
h3D = input_fits[0].header
h2D = coordinates.get_header2d(h3D)
wav = coordinates.get_wav_axis(h3D)

if args.rsmooth!=None:
    cube = imaging.smooth_nd(cube, args.rsmooth, axes=(1,2))
    var_cube = imaging.smooth_nd(cube, args.rsmooth, axes=(1,2), var=True)

if args.wsmooth!=None:
    cube = imaging.smooth_nd(cube, args.wsmooth, axes=[0])
    var_cube = imaging.smooth_nd(cube,args.wsmooth, axes=[0], var=True)

if args.wsmooth!=None or args.rsmooth!=None:
    var_cube = variance.rescale_var(var_cube, cube, fmin=0, fmax=10)

#Load object cube info
if args.obj==None: obj_cube = np.ones_like(cube)
else:
    if os.path.isfile(args.obj): obj_cube = fits.getdata(args.obj)
    else: raise FileNotFoundError(args.obj)

    try: obj_ids = list( int(x) for x in args.id.split(',') )
    except: raise ValueError("Could not parse -objid flag. Should be comma-separated list of object IDs.")

    #Convert object cube into binary mask cube, accepting IDs based on -id flag
    if obj_ids==[-1]: obj_cube[obj_cube>0] = 1 #Accept all non-zero IDs
    elif obj_ids==[-2]: obj_cube[obj_cube>0] = 0 #Use none? (TODO: What is this for?)
    else: #Accept IDs given as comma separated list
        for obj_id in obj_ids: obj_cube[obj_cube==obj_id] = -99
        obj_cube[obj_cube>0] = 0
        obj_cube[obj_cube==-99] = 1

#Get 2D mask of object spaxels
msk_2d = np.max(obj_cube,axis=0)
msk_1d = np.max(obj_cube,axis=(1,2))
wav_obj = wav[msk_1d>0]

#Set non-object voxels to zero
cube[obj_cube==0] = 0

#Create canvas for both first and zeroth (f,z) moments
m1_map = np.zeros_like(msk_2d,dtype=float)
m2_map = np.zeros_like(m1_map)

m1_err = np.zeros_like(m1_map)
m2_err = np.zeros_like(m2_map)

try: m1_guess = wav_obj[int(len(wav_obj)/2)]
except: pass

#Only perform calculation if object has any valid spaxels
if np.count_nonzero(msk_2d)>0:

    for i in range(m1_map.shape[0]):
        for j in range(m1_map.shape[1]):

            if msk_2d[i,j]:

                spc_ij = cube[msk_1d > 0, i, j] #Get 1D spectrum at (i,j) within z-mask

                if args.method == 'closing-window':
                    m1_ij, m2_ij, m1_ij_err, m2_ij_err  = kinematics.closing_window_moments(wav_obj, spc_ij, mu1_init=m1_guess)
                elif args.method == 'basic':
                    m1_ij, m2_ij, m1_ij_err, m2_ij_err  = kinematics.basic_moments(wav_obj, spc_ij, pos_thresh=False)
                elif args.method == 'positive':
                    m1_ij, m2_ij, m1_ij_err, m2_ij_err = kinematics.basic_moments(wav_obj, spc_ij, pos_thresh=True)

                if np.isnan(m1_ij) or np.isnan(m2_ij) or m1_ij==-1:
                    msk_2d[i, j] = 0
                    continue

                else:
                    m1_map[i,j] = m1_ij #Fill in to maps if valid
                    m2_map[i,j] = m2_ij

                    m1_err[i,j] = m1_ij_err
                    m2_err[i,j] = m2_ij_err

    #Calculate integrated spectrum
    if args.mode == 'vel':

        spec_1d = np.sum(cube,axis=(1,2))
        spec_1d -= np.median(spec_1d)
        spec_1d = spec_1d[msk_1d>0]
        m1_ref = np.sum(wav_obj*spec_1d)/np.sum(spec_1d)

        #print("%30s  %10.3f %10.3f"%(args.cube.split('/')[0], m1_ref, disp_global_kms))
        #Convert moments to velocity space
        m1_map = 3e5*(m1_map - m1_ref)/m1_ref
        m2_map = 3e5*np.sqrt(m2_map)/m1_ref
        m1_err = 3e5*(m1_err)/m1_ref
        m2_err = 3e5*(m2_err)/m1_ref

else: m1_ref = 0

#Fill in empty or bad spaxels with fill value if selected
if args.filltype == 'value':

    m1_map[msk_2d == 0] = args.fillvalue
    m2_map[msk_2d == 0] = args.fillvalue
    m1_err[msk_2d == 0] = args.fillvalue
    m2_err[msk_2d == 0] = args.fillvalue
#Use NaNs if requested
elif args.filltype == 'nan':

    m1_map[msk_2d == 0] = np.nan
    m2_map[msk_2d == 0] = np.nan
    m1_err[msk_2d == 0] = np.nan
    m2_err[msk_2d == 0] = np.nan


if args.method == 'closing-window': method = "_clw"
elif args.method == 'positive': method = "_pos"
else: method = ""
m1_out_ext = ".vel%s.fits"%method if args.mode == 'vel' else ".m1%s.fits"%method
m2_out_ext = ".dsp%s.fits"%method if args.mode == 'vel' else ".m2%s.fits"%method

m1_fits = fits.HDUList([fits.PrimaryHDU(m1_map)])
m1_fits[0].header = h2D

m2_fits = fits.HDUList([fits.PrimaryHDU(m2_map)])
m2_fits[0].header = h2D

m1_err_fits = fits.HDUList([fits.PrimaryHDU(m1_err)])
m1_err_fits[0].header = h2D

m2_err_fits = fits.HDUList([fits.PrimaryHDU(m2_err)])
m2_err_fits[0].header = h2D

if args.mode == 'vel':
    m1_fits[0].header["M1REF"] = m1_ref
    m1_fits[0].header["BUNIT"] = "km/s"
    m2_fits[0].header["BUNIT"] = "km/s"
    m1_err_fits[0].header["BUNIT"] = "km/s"
    m2_err_fits[0].header["BUNIT"] = "km/s"

else:
    try: wavunit = m1_fits[0].header["CUNIT3"]
    except: wavunit = "WAV"

    m1_fits[0].header["BUNIT"] = wavunit
    m2_fits[0].header["BUNIT"] = wavunit
    m1_err_fits[0].header["BUNIT"] = wavunit
    m2_err_fits[0].header["BUNIT"] = wavunit

m1_out = args.cube.replace('.fits', m1_out_ext)
m1_fits.writeto(m1_out,overwrite=True)
print("Saved %s"%m1_out)

m2_out = m1_out.replace("vel", "dsp")
m2_fits[0].header["BUNIT"] = "km/s"
m2_fits.writeto(m2_out, overwrite=True)
print("Saved %s"%m2_out)

m1_err_out = m1_out.replace('.fits', '_err.fits')
m1_err_fits.writeto(m1_err_out,overwrite=True)
print("Saved %s"%m1_err_out)

m2_err_out = m2_out.replace('.fits', '_err.fits')
m2_err_fits.writeto(m2_err_out,overwrite=True)
print("Saved %s"%m2_err_out)
