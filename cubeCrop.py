#!/usr/bin/env python
from astropy.io import fits as fitsIO
import numpy as np
import sys
import time
import libs

# Timer start
tStart = time.time()

# Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

# Add file extension of omitted
if not ".fits" in cubetype: cubetype += ".fits"

# Check if any parameter values are missing (set to set-up mode if so)
params = libs.params.loadparams(parampath)

# Check if parameters are complete
libs.params.verify(params)

# Get filenames     
files = libs.io.findfiles(params,cubetype)

# If there are non-NAS cubes - add sky cubes to list to be cropped (same crop params)
for i in range(len(params["IMG_ID"])):
    img,sky = params["IMG_ID"][i],params["SKY_ID"][i]
    if img!=sky:      
        files.append(files[i].replace(img,sky))
        params["IMG_ID"].append(sky)
        params["XCROP"].append(params["XCROP"][i])
        params["YCROP"].append(params["YCROP"][i])
        params["WCROP"].append(params["WCROP"][i])
        
# Open  FITS objects
fits = [fitsIO.open(f) for f in files] 

# If all input cubes are icubes - try to update variance cubes as well
propVar  = np.all([ "icube" in fileName for fileName in files])
if propVar:
    varFiles = [ f.replace("icube","vcube") for f in files ]
    try: varFits = [ fitsIO.open(v) for v in varFiles ]
    except:
        print("Could not load variance input cubes from data directory. Error will not be propagated throughout coadd.")
        propVar=False
                


# Crop FITS and make sure units are flux-like before coadding
print("Cropping cubes...\n"),
fits = libs.cubes.cropFITS(fits,params)
for i,f in enumerate(fits):
    cropName = files[i].replace('.fits','.c.fits')
    f.writeto(cropName,overwrite=True)
    print("Saved %s"%cropName)

# Crop the variance cubes too
if propVar:
    print("\nCropping corresponding variance cubes...\n"),
    var = libs.cubes.cropFITS(varFits,params)
    for i,v in enumerate(varFits):
        cropName = varFiles[i].replace('.fits','.c.fits')
        v.writeto(cropName,overwrite=True)
        print("Saved %s"%cropName)
        
# Timer end
tFinish = time.time()
print("\nElapsed time: %.2f seconds" % (tFinish-tStart))
