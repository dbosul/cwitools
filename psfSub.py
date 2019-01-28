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

#Take minimum input 
filePath = sys.argv[1]
xP = float(sys.argv[2])
yP = float(sys.argv[3])
if len(sys.argv)>4: 
    try: z0,z1 = ( int(x) for x in sys.argv[4].split(','))
    except: z0,z1 = 0,0
else: z0,z1 = 0,0

fits = fitsIO.open(filePath)
data = fits[0].data.copy()
cdata = data.copy()
if z1>0: cdata[z0:z1] = 0
model = np.zeros_like(data)
w,y,x = data.shape
hdr  = fits[0].header
wcs = WCS(hdr)
pxScales = proj_plane_pixel_scales(wcs)

pxScales[:2]*=3600

fitter = fitting.LevMarLSQFitter()

rFit = 2
rSub = 8

rFitPx = int(round(rFit/pxScales[1]))
rSubPx = int(round(rSub/pxScales[1]))

xP = int(round(xP))
yP = int(round(yP))

xFit = (xP-rFitPx, xP+rFitPx)
yFit = (yP-rFitPx, yP+rFitPx)

xSub = (xP-rSubPx, xP+rSubPx)
ySub = (yP-rSubPx, yP+rSubPx)

for wi in range(w):
    
    layer = data[wi]
    
    layer-= np.median(layer)
    
    a = max(0,wi-170)
    b = min(w,a+340)
    
    psfImg = np.mean(cdata[a:b],axis=0)
    
    psfImg -= np.median(psfImg)
    
    
    psfImgFit = psfImg[yFit[0]:yFit[1],xFit[0]:xFit[1]]
    layerFit  = layer[yFit[0]:yFit[1],xFit[0]:xFit[1]]
    
    #plt.figure()
    #plt.pcolor(psfImgFit)
    #plt.show()
    
    scaleGuess = models.Scale(factor=np.max(layerFit)/np.max(psfImgFit))

    scaleFit = fitter(scaleGuess,psfImgFit,layerFit)
                    
    A = scaleFit.factor.value
    
    fits[0].data[wi][ySub[0]:ySub[1],xSub[0]:xSub[1]] -= A*psfImg[ySub[0]:ySub[1],xSub[0]:xSub[1]]

    #model[wi][ySub[0]:ySub[1],xSub[0]:xSub[1]] += (A**2)*psfImg[ySub[0]:ySub[1],xSub[0]:xSub[1]]
    
fits.writeto(filePath.replace('.fits','.ps.fits'),overwrite=True)
#varPath = filePath.replace('icube','vcube')
#try:
#    varFITS = fitsIO.open(varPath)
#    varFITS[0].data += model
#    varFITS.writeto(varPath.replace('.fits','.ps.fits'),overwrite=True)
#except: print("Could not update variance file: %s"%varPath)
#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))        
