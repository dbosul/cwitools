#!/usr/bin/env python
from astropy.io import fits as fitsIO
import numpy as np
import sys
import time
import libs

#Timer start
tStart = time.time()

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Add file extension of omitted
if not ".fits" in cubetype: cubetype += ".fits"

#Check if any parameter values are missing (set to set-up mode if so)
params = libs.params.loadparams(parampath)

#Get filenames     
files = libs.io.findfiles(params,cubetype)

#Open  FITS objects
fits = [fitsIO.open(f) for f in files] 

#Check if parameters are complete
libs.params.verify(params)

#Crop FITS and make sure units are flux-like before coadding
print("Cropping cubes."),
fits = libs.cubes.cropFITS(fits,params)

for i,f in enumerate(fits):
    cropName = files[i].replace('.fits','.c.fits')
    f.writeto(cropName,overwrite=True)
    print("Saved %s."%cropName)
    
#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))
