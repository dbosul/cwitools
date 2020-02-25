from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales as getPxScales
from scipy.ndimage.filters import gaussian_filter1d as Gauss1D
from scipy.optimize import differential_evolution

import libs #CWITools import

import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Get summed spectrum within a certain radius of a source.')
mainGroup = parser.add_argument_group(title="Main",description="Basic input")
mainGroup.add_argument('cube',
                    type=str,
                    help='The input data cube.'
)
mainGroup.add_argument('par',
                    type=str,
                    help='CWITools parameter file (used for target position).'
)
mainGroup.add_argument('-r',
                    type=float,
                    help='Radius within which to sum spectrum.',
                    default=15

)
mainGroup.add_argument('-smooth',
                    type=float,
                    help='Standard deviation of 2D Gaussian spatial smoothing kernel. Default: 1.5px',
                    default=None

)
mainGroup.add_argument('-ext',
                    type=str,
                    help='File extension to use for saving spectrum. Default: .rSPC.fits',
                    default='.rSPC.fits'

)
args = parser.parse_args()


#Constants
c = 3e5 #Speed of light in km/s

#Load data
cube,hdr = fits.getdata(args.cube,header=True)

#Get WCS structures
hdr2D = libs.cubes.get2DHeader(hdr)
wcs2D = WCS(hdr2D)
wavAxis = libs.cubes.getWavAxis(hdr)

#Open CWITools params
p = libs.params.loadparams(args.par)

#Get position if not provided as a separate argument
qsoPos = wcs2D.all_world2pix(p["RA"],p["DEC"],0)

#Get meshgrid of distance from source
yy,xx = np.indices(cube[0].shape)
rr = np.sqrt( (yy-qsoPos[1])**2 + (xx-qsoPos[0])**2 )
rMask = rr<=args.r

#Zero-out any spaxels not being summed
cubeT = cube.T.copy()
cubeT[~rMask.T] = 0
spectrum = np.sum(cubeT,axis=(0,1))

#Get number of spaxels
Nspax = np.count_nonzero(rMask)

#Get area of pixel in arcsec 2
pxScls = getPxScales(wcs2D)*3600 #Pixel scales in degrees (x3600 to arcsec)
pxArea = pxScls[0]*pxScls[1] #Pixel size in arcsec2

#Diide by area in arcsec2 to go from erg/s/cm2/A to erg/s/cm2/arcsec2/A
spectrum /= (Nspax*pxArea)

if args.smooth!=None: spectrum = Gauss1D(spectrum,sigma=libs.science.fwhm2sigma(args.smooth))

outpath = args.cube.replace('.fits','.SPC.fits')
libs.cubes.saveFITS(spectrum,libs.cubes.get1DHeader(hdr),outpath)
print(outpath)
