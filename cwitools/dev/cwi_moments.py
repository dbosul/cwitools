
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from photutils import DAOStarFinder
from scipy.stats import sigmaclip
from scipy.signal import medfilt

import astropy.convolution as astConvolve
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import pyregion
import sys
import time

from cwitools import libs

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

args = parser.parse_args()

#Try to load the fits file
if os.path.isfile(args.cube):
    input_fits = fits.open(args.cube)
else: sys.exit()#raise FileNotFoundError(args.cube)

#Extract useful stuff and create useful data structures
cube  = input_fits[0].data.copy()

#Try to load the fits file
if args.var!=None:
    if os.path.isfile(args.var):
        var_fits = fits.open(args.var)
        var_cube = var_fits[0].data
        var_cube = libs.science.nonpos2inf(var_cube)
        cube /= var_cube
    else: raise FileNotFoundError(args.var)



w,y,x = cube.shape
h3D   = input_fits[0].header
h2D   = libs.cubes.get_header2d(h3D)
wcs   = WCS(h2D)
wav   = libs.cubes.get_wavaxis(h3D)

if args.rsmooth!=None: cube = libs.science.smooth3d(cube,args.rsmooth,axes=(1,2))

if args.wsmooth!=None: cube = libs.science.smooth3d(cube,args.wsmooth,axes=[0])


#Load object info
if args.obj==None: obj_cube = np.ones_like(cube)
else:
    if os.path.isfile(args.obj): obj_cube = fits.getdata(args.obj)
    else: raise FileNotFoundError(args.obj)

    try: obj_ids = list( int(x) for x in args.id.split(',') )
    except: raise ValueError("Could not parse -objid flag. Should be comma-separated list of object IDs.")

    #Convert object id cube into binary cube, accepting given IDs
    if obj_ids==[-1]: obj_cube[obj_cube>0] = 1
    elif obj_ids==[-2]: obj_cube[obj_cube>0] = 0
    else:
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

try: mu1_guess = wav_obj[int(len(wav_obj)/2)]
except: pass

#Only perform calculation if object has any valid spaxels
if np.count_nonzero(msk_2d)>0:

    #Calculate first moment
    m1_num = np.zeros_like(m1_map)
    m1_den = np.zeros_like(m1_map)

    for i in range(m1_map.shape[0]):
        for j in range(m1_map.shape[1]):

            if msk_2d[i,j]:

                spc_ij = cube[msk_1d > 0, i, j] #Get 1D spectrum at (i,j) within z-mask

                if args.method == 'closing-window':
                    m1_ij, m2_ij = libs.science.closing_window_moments(wav_obj, spc_ij, mu1_init=mu1_guess)
                elif args.method == 'basic':
                    m1_ij, m2_ij = libs.science.basic_moments(wav_obj, spc_ij, pos_thresh=False)
                elif args.method == 'positive':
                    m1_ij, m2_ij = libs.science.basic_moments(wav_obj, spc_ij, pos_thresh=True)

                if np.isnan(m1_ij) or np.isnan(m2_ij) or m1_ij==-1:
                    msk_2d[i, j] = 0
                    continue

                else:
                    m1_map[i,j] = m1_ij #Fill in to maps if valid
                    m2_map[i,j] = m2_ij



    #Calculate integrated spectrum

    spec_1d = np.sum(cube,axis=(1,2))
    spec_1d -= np.median(spec_1d)
    spec_1d = spec_1d[msk_1d>0]
    m1_ref = np.sum(wav_obj*spec_1d)/np.sum(spec_1d)

    #print("%30s  %10.3f %10.3f"%(args.cube.split('/')[0], m1_ref, disp_global_kms))
    #Convert moments to velocity space
    m1_map = 3e5*(m1_map - m1_ref)/m1_ref
    m2_map = 3e5*np.sqrt(m2_map)/m1_ref

else: m1_ref = 0

#Fill in empty or bad spaxels with fill value if selected
if args.filltype == 'value':

    m1_map[msk_2d == 0] = args.fillvalue
    m2_map[msk_2d == 0] = args.fillvalue

#Use NaNs if requested
elif args.filltype == 'nan':

    m1_map[msk_2d == 0] = np.nan
    m2_map[msk_2d == 0] = np.nan

if args.method == 'closing-window': out_ext = '.vel_clw.fits'
elif args.method == 'basic': out_ext = '.vel.fits'

m1_out = args.cube.replace('.fits', out_ext)
m1_fits = libs.cubes.make_fits(m1_map, h2D)
m1_fits[0].header["M1REF"] = m1_ref
m1_fits[0].header["BUNIT"] = "km/s"
m1_fits.writeto(m1_out,overwrite=True)
print("Saved %s"%m1_out)

m2_out = m1_out.replace("vel", "dsp")
m2_fits = libs.cubes.make_fits(m2_map,h2D)
m2_fits[0].header["M0REF"] = 0
m2_fits[0].header["BUNIT"] = "km/s"
m2_fits.writeto(m2_out, overwrite=True)
print("Saved %s"%m2_out)
