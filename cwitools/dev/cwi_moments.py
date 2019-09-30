
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

def first_moment(x, y): return np.sum(x*y)/np.sum(y)
def second_moment(x, y, mu): return np.sum(y*(x-mu)**2 )/np.sum(y)

#Closing-window method for moment calculation in noisy data
def iterative_moments(x, y, window_min=3, window_step=1):

    #Take only > 1sig values
    usex = y > np.std(y)

    #Initialize window at maximum size
    dx = x[1]-x[0]
    window = len(x)*dx

    # Loop over window size
    while window > window_min:

        # Calculate moments in current window
        mu_1 = first_moment(x[usex], y[usex])
        mu_2 = second_moment(x[usex], y[usex], mu_1)

        #Decrease window size
        window -= window_step*dx

        #Update window, centered on new first moment
        usex = (np.abs(x-mu_1) < window) & (y > np.std(y))

    #   Return both moments
    return mu_1, mu_2

def get_moments(x, y):

    mu_1 = first_moment(x, y)
    mu_2 = second_moment(x, y, mu_1)

    return mu_1, mu_2

#Timer start
tStart = time.time()

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

#Only perform calculation if object has any valid spaxels
if np.count_nonzero(msk_2d)>0:

    #Calculate first moment
    m1_num = np.zeros_like(m1_map)
    m1_den = np.zeros_like(m1_map)

    for i in range(m1_map.shape[0]):
        for j in range(m1_map.shape[1]):

            if msk_2d[i,j]:

                spc_ij = cube[msk_1d > 0, i, j] #Get 1D spectrum at (i,j) within z-mask

                m1_ij, m2_ij = iterative_moments(wav_obj, spc_ij) #Calculate moments using iterative method

                if np.isnan(m1_ij) or np.isnan(m2_ij):
                    msk_2d[i, j] = 0
                    continue

                else:
                    m1_map[i,j] = m1_ij #Fill in to maps if valid
                    m2_map[i,j] = m2_ij



    #Calculate integrated spectrum
    spec_1d = np.sum(cube[msk_1d > 0],axis=(1,2))
    m1_ref = np.sum(wav_obj*spec_1d)/np.sum(spec_1d)

    disp_global = np.sqrt(np.sum(spec_1d*np.power(wav_obj - m1_ref, 2))/np.sum(spec_1d))
    disp_global_kms = 3e5*disp_global/m1_ref

    if 1:
        thresh = spec_1d > 0
        spec_1d_thresh = spec_1d[thresh]
        wav_obj_thresh = wav_obj[thresh]
        disp_global = np.sqrt(np.sum(spec_1d_thresh*np.power(wav_obj_thresh - m1_ref, 2))/np.sum(spec_1d_thresh))
        disp_global_kms = 3e5*disp_global/m1_ref

    print("%30s  %10.3f %10.3f"%(args.cube, m1_ref, disp_global_kms))
    #Convert moments to velocity space
    m1_map = 3e5*(m1_map - m1_ref)/m1_ref
    m2_map = 3e5*m2_map/m1_ref

else: m1_ref = 0

#Zero-out values not included in nebula
m1_map[msk_2d == 0] = -5000
m2_map[msk_2d == 0] = -5000

m1_map_filt = medfilt(m1_map,kernel_size=3)
edgepx = (m1_map == -5000) & (msk_2d == 1)
m1_map_filt[edgepx] = m1_map[edgepx]
m1_map = m1_map_filt

m2_map_filt = medfilt(m2_map,kernel_size=3)
edgepx = (m2_map == -5000) & (msk_2d == 1)
m2_map_filt[edgepx] = m2_map[edgepx]
m2_map = m2_map_filt

# m1_out = args.cube.replace('.fits','.vel.fits')
# m1_fits = libs.cubes.make_fits(m1_map,h2D)
# m1_fits[0].header["M1REF"] = m1_ref
# m1_fits[0].header["BUNIT"] = "km/s"
# m1_fits.writeto(m1_out,overwrite=True)
# print("Saved %s"%m1_out)
#
# m1_err_out = args.cube.replace('.fits','.vel_err.fits')
# m1_err_fits = fits.HDUList([fits.PrimaryHDU(m1_err)])
# m1_err_fits[0].header = h2D
# m1_err_fits[0].header["BUNIT"] = "km/s"
# m1_err_fits.writeto(m1_err_out,overwrite=True)
# print("Saved %s"%m1_err_out)
#
# m2_out = args.cube.replace('.fits','.disp.fits')
# m2_fits = libs.cubes.make_fits(m2_map,h2D)
# m2_fits[0].header["M0REF"] = 0
# m2_fits[0].header["BUNIT"] = "km/s"
# m2_fits.writeto(m2_out, overwrite=True)
# print("Saved %s"%m2_out)
#
# m2_err_out = args.cube.replace('.fits','.disp_err.fits')
# m2_err_fits = fits.HDUList([fits.PrimaryHDU(m2_err)])
# m2_err_fits[0].header = h2D
# m2_err_fits[0].header["BUNIT"] = "km/s"
# m2_err_fits.writeto(m2_err_out,overwrite=True)
# print("Saved %s"%m2_err_out)
