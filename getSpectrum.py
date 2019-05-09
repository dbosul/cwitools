from astropy import units as u
from astropy.io import fits
from astropy.modeling import models,fitting
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clip

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from photutils import DAOStarFinder

import matplotlib.pyplot as plt
import argparse
import numpy as np
import pyregion
import sys
import time

import libs


#Timer start
tStart = time.time()

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Get integrated spectrum of a source and save to FITS.')
mainGroup = parser.add_argument_group(title="Main",description="Basic input")
mainGroup.add_argument('cube', 
                    type=str, 
                    metavar='cube',             
                    help='The cube to be PSF subtracted.'
)
srcGroup = parser.add_mutually_exclusive_group(required=True)
srcGroup.add_argument('-pos',
                    type=str,
                    metavar='float tuple',
                    help='Position of source (x,y).',
                    default=None
)
srcGroup.add_argument('-par',
                    type=str,
                    metavar='path',
                    help='CWITools parameter file for source.',
                    default=None
)
methodGroup = parser.add_argument_group(title="Method")
methodGroup.add_argument('-r',
                    type=float,  
                    metavar='Fit Radius',  
                    help='Radius (arcsec) within which to sum spectrum',
                    default=3
)
methodGroup.add_argument('-recenter',
                    type=str,
                    metavar='Recenter',
                    help='Auto-recenter the input positions using PSF centroid',
                    choices=["True","False"],
                    default="True"
)
fileIOGroup = parser.add_argument_group(title="File Options")
fileIOGroup.add_argument('-ext',
                    type=str,
                    metavar='File Extension',
                    help='Extension to append to subtracted cube (.ps.fits)',
                    default='.SPC.fits'
)

args = parser.parse_args()

#Try to load the fits file
try: F = fits.open(args.cube)
except: print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()
         
#Open fits image and extract info
hdr  = F[0].header
cube = F[0].data.copy()
wlIm = np.sum(cube,axis=0)

#Get dimensions 
w,y,x = cube.shape
W,Y,X = np.arange(w),np.arange(y),np.arange(x)

#Get WCS info
wcs = WCS(libs.cubes.get2DHeader(hdr))
pxScales = proj_plane_pixel_scales(wcs)
xScale,yScale = (pxScales[0]*u.deg).to(u.arcsec), (pxScales[1]*u.degree).to(u.arcsec) 
pxArea   = ( xScale*yScale ).value

#Get radius in pixels
r_px = args.r/xScale.value

#Get box around source for recentering
boxSize = 3*int(round(r_px))
yy,xx   = np.mgrid[:boxSize, :boxSize]

#Get default PSF model for re-centering
psfModel = models.Gaussian2D(amplitude=1,x_mean=boxSize/2,y_mean=boxSize/2)

#Get fitter for PSF re-centering
fitter   = fitting.LevMarLSQFitter()

#Load from position if given
if args.pos!=None:
    try: pos = tuple(float(x) for x in args.pos.split(','))
    except: print("Could not parse position argument. Should be two comma-separated floats (e.g. 45.2,33.6)");sys.exit() 
    print("Source Position: %.1f,%.1f"%(pos[0],pos[1]))
    yP,xP = pos

#Load from param file 
elif args.par!=None:   
    try: params = libs.params.loadparams(args.par)
    except: print("Could not load parameter file.");sys.exit() 
    xP,yP = wcs.all_world2pix(params["RA"],params["DEC"],0)

else: print("No source given");sys.exit()

#Get cut-out around source
psfBox = Cutout2D(wlIm,(xP,yP),(boxSize,boxSize),mode='partial',fill_value=-99).data

#Get useable spaxels
fitXY = np.array( psfBox!=-99, dtype=int)

#Run fit
psfFit = fitter(psfModel,yy,xx,psfBox,weights=fitXY)

#Get sigma/fwhm
xfwhm,yfwhm = 2.355*psfFit.x_stddev.value, 2.355*psfFit.y_stddev.value
        
#We take larger of the two for our purposes
fwhm = max(xfwhm,yfwhm)

#Only continue with well-fit, high-snr sources
if args.recenter=='True' and fitter.fit_info['nfev']<100 and fwhm<10/xScale.value:
    yP, xP = psfFit.x_mean.value+yP-boxSize/2, psfFit.y_mean.value+xP-boxSize/2
            
#Get meshgrid of distance from P
YY,XX = np.meshgrid(X-xP,Y-yP)
RR    = np.sqrt(XX**2 + YY**2)

#Get boolean mask to sum over
sumPx = RR<=r_px
          
#Zero the non-summed spaxels and sum over spatial axes to get spectrum
pCube = cube.T.copy()
pCube[sumPx.T==0] = 0
pSpec = np.sum( pCube, axis=(0,1) )

#Convert the units to proper flux units
pSpec *= 10 #Switch to 10e-18 from 10e-16

#Save FITS
pFITS = fits.HDUList([fits.PrimaryHDU(pSpec)])
pFITS[0].header = libs.cubes.get1DHeader(hdr)
pFITS.writeto(args.cube.replace('.fits',args.ext),overwrite=True)
print("Saved %s"%args.cube.replace('.fits',args.ext))

  
