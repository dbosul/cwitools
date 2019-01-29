# SYNTAX:
# python psfSub.py <cube> <x> <y>
# 
# Optional parametes
# rmin (float/arcseconds) - radius to use for fitting PSF (default 1'')
# rmax (float(arcseconds) - radius to use for subtracting PSF (default 3'')
# zmask(int tuple / pixels) - layers to mask for creating WL image (default None)
# window (float / angstrom) - wavelength window to use for creating WL image (default=150A)
# out (string) - string to append to input filename (default .ps.fits)
 


from astropy import units as u
from astropy.io import fits as fitsIO
from astropy.modeling import models,fitting
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clip

from astropy import units
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass as CoM

import matplotlib.pyplot as plt
import numpy as np
import pyregion
import sys
import time

import libs

#Timer start
tStart = time.time()


#Take any additional input params, if provided
settings = {"rmin":0.5,"rmax":3,"window":150,"zmask":(0,0),"out":".ps.fits"}
if len(sys.argv)>2:
    for item in sys.argv[2:]:
        if "=" in item:     
            key,val = item.split('=')
            if settings.has_key(key):
                if key in ["rmin","rmax","window"]: val=float(val)
                settings[key]=val
            else:
                print "Input argument not recognized: %s" % key
                sys.exit()
                
#Take minimum input 
filePath = sys.argv[1]
xP = float(sys.argv[2])
yP = float(sys.argv[3])

#Open fits image and extract info
fits = fitsIO.open(filePath)
hdr  = fits[0].header
wcs = WCS(hdr)
pxScales = proj_plane_pixel_scales(wcs)
in_cube = fits[0].data.copy()
wl_cube = in_cube.copy()

#Mask any emission in WL cube if requested
z0,z1 = (int(x) for x in settings["zmask"].split(','))
if z1>0: wl_cube[z0:z1] = 0

#Create cube for psfModel
model = np.zeros_like(in_cube)
w,y,x = in_cube.shape
Y,X   = np.arange(y),np.arange(x)

#Convert plate scale to arcseconds
xScale,yScale = (pxScales[:2]*u.deg).to(u.arcsecond)
zScale = (pxScales[2]*u.meter).to(u.angstrom)

#Convert fitting & subtracting radii to pixel values
rMin_px = int(round(settings["rmin"]/xScale.value))
rMax_px = int(round(settings["rmax"]/xScale.value))
delZ_px = int(round(0.5*settings["window"]/zScale.value))

#Get fitter for PSF fit
fitter = fitting.LevMarLSQFitter()

#Get meshgrid of distance from P
YY,XX = np.meshgrid(Y-yP,X-xP)
RR    = np.sqrt(XX**2 + YY**2)

#Get boolean masks for
fitPx = RR<=rMin_px
subPx = RR<=rMax_px

#Run through wavelength layers
for wi in range(w):
    
    #Get this wavelenght layer and subtract any median residual
    layer = in_cube[wi] 
    layer-= np.median(layer)
    
    #Get upper and lower-bounds for creating WL image
    a = max(0,wi-delZ_px)
    b = min(w,a+delZ_px)
    
    #Create PSF image
    psfImg = np.mean(wl_cube[a:b],axis=0)
    psfImg -= np.median(psfImg)
    
    #Extract portion of image used for fitting scaling factor
    psfImgFit = psfImg[fitPx]
    layerFit  = layer[fitPx]

    #Create initial guess using Astropy Scale 
    scaleGuess = models.Scale(factor=np.max(layerFit)/np.max(psfImgFit))

    #Fit
    scaleFit = fitter(scaleGuess,psfImgFit,layerFit)
                
    #Extract fit value    
    A = scaleFit.factor.value
    
    #Subtract fit from data 
    fits[0].data[wi][subPx] -= A*psfImg[subPx]

    #Add to PSF model
    model[wi][subPx] += A*psfImg[subPx]
    
outFileName = filePath.replace('.fits',settings["out"])
fits.writeto(outFileName,overwrite=True)

psfFits = fitsIO.HDUList([fitsIO.PrimaryHDU(model)])
psfFits[0].header = hdr
psfFits.writeto(outFileName.replace('.fits','.PSF.fits'),overwrite=True)

#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))        
