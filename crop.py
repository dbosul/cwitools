# Crop in WAVELENGTH
# Arguments:
#   1. Input Cube
#   2. Redshift of Target

from astropy.io import fits as fitsIO
from astropy import units
import numpy as np
import sys

import libs


#Define units 
km  = units.kilometer
s   = units.second
A   = units.angstrom

#Define some constants
c   = 3e5*km/s     # Speed of light in cm/s
lyA = 1215.6*A     # Wavelength of LyA (Angstrom)
dv  = 2000*km/s    # Velocity window

#Take minimum input 
paramPath = sys.argv[1]
cubeType  = sys.argv[2]

#Take any additional input params, if provided
settings = {"level":"coadd","line":"lyA"}
if len(sys.argv)>3:
    for item in sys.argv[3:]:      
        key,val = item.split('=')
        if settings.has_key(key): settings[key]=val
        else:
            print "Input argument not recognized: %s" % key
            sys.exit()
            
#Load parameters
params = libs.params.loadparams(paramPath)

#Get filenames     
if settings["level"]=="coadd":   files = [ '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubeType) ]
elif settings["level"]=="input": files = libs.io.findfiles(params,cubetype)
else:
    print("Setting 'level' must be either 'coadd' or 'input'. Exiting.")
    sys.exit()

#Run through files to be cropped
for fileName in files:
    
    #Open FITS and extract info
    fits = fitsIO.open(fileName)
    h = fits[0].header
    w,y,x = fits[0].data.shape
    WW = np.array([ h["CRVAL3"] + h["CD3_3"]*(i - h["CRPIX3"]) for i in range(w)])

    #Calculate wavelength range
    if settings["line"]=="lyA":
        centerWav = (1+params["ZLA"])*lyA
    else:
        print("Setting - line:%s - not recognized."%settings["line"])
        sys.exit()
        
    deltaWav = centerWav*(dv/c)
    
    w1,w2 = centerWav-deltaWav, centerWav+deltaWav

    a,b = libs.cubes.getband(w1.value,w2.value,h)

    a = max(0,a)
    b = min(w-1,b)

    #Save FITS of cropped data
    cropName = fileName.replace(".fits",".%s.fits" % settings["line"])
    cropData = fits[0].data[a:b]
    cropHDU  = fitsIO.PrimaryHDU(cropData)
    cropFITS = fitsIO.HDUList([cropHDU])
    cropFITS[0].header = fits[0].header.copy()
    cropFITS[0].header["CRPIX3"] -= a
    cropFITS.writeto(cropName,overwrite=True)
    print("Saved %s."%cropName)

