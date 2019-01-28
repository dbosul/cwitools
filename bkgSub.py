from astropy.io import fits as fitsIO
from astropy.modeling import models,fitting
import numpy as np
import sys
import time

import libs

#Timer start
tStart = time.time()

#Define some constants
c   = 3e5       # Speed of light in km/s
lyA = 1215.6    # Wavelength of LyA in Angstrom
v   = 1000      # Velocity window for line emission in km/s

#Take minimum input 
paramPath = sys.argv[1]
cubeType  = sys.argv[2]

#Take any additional input params, if provided
settings = {"level":"coadd","line":"lyA","k":1}
if len(sys.argv)>3:
    for item in sys.argv[3:]:      
        key,val = item.split('=')
        if settings.has_key(key):
            if key=="k": val=int(val)
            settings[key]=val
        else:
            print "Input argument not recognized: %s" % key
            sys.exit()
            
#Load parameters
params = libs.params.loadparams(paramPath)

#Check if parameters are complete
libs.params.verify(params)

#Get filenames     
if settings["level"]=="coadd":   files = [ sys.argv[2] ]#'%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubeType) ]
elif settings["level"]=="input": files = libs.io.findfiles(params,cubetype)
else:
    print("Setting 'level' must be either 'coadd' or 'input'. Exiting.")
    sys.exit()

#Calculate wavelength range
if settings["line"]=="lyA":
    wC = (1+params["ZLA"])*lyA
else:
    print("Setting - line:%s - not recognized."%settings["line"])
    sys.exit()

#Pick fitter to use for scaling PSF
fitter  = fitting.LinearLSQFitter()
bgModel = models.Polynomial1D(degree=settings["k"])

#Run through files to be cropped
for fileName in files:
    
    #Open FITS and extract info
    f = fitsIO.open(fileName)
    h = f[0].header
    d = f[0].data
    
    w,y,x = f[0].data.shape
    WW = np.array([ h["CRVAL3"] + h["CD3_3"]*(i - h["CRPIX3"]) for i in range(w)])
   
    #Get continuum wavelength mask
    contWavs = np.ones(w,dtype=bool)
    w1,w2 =  wC*(1-v/c), wC*(1+v/c) 
    a,b = libs.cubes.getband(w1,w2,h)
    a,b = max(0,a), min(w-1,b)
    contWavs[a:b] = 0
    skyMask = libs.cubes.get_skyMask(f)
    contWavs[skyMask==1] = 0
    
    WWFit = WW[contWavs]
    
    #Run through spaxels and subtract low-order polynomial
    for yi in range(d.shape[1]):
        for xi in range(d.shape[2]):
            
            spec = d[:,yi,xi]
            
            specFit = spec[contWavs]
            
            specModel = fitter(bgModel, WWFit, specFit)
            
            f[0].data[:,yi,xi] -= specModel(WW)
            
    f[0].data[skyMask==1] = 0
    
    #Write out PSF-subtracted fits
    outFile = fileName.replace('.fits','.bs.fits')
    f.writeto(outFile,overwrite=True)
    print("Saved %s" % outFile)

#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))                    
