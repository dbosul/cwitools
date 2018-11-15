from astropy.io import fits as fitsIO
from astropy import units
import numpy as np
import sys
import libs
import time

# Timer start
tStart = time.time()

# Define some constants
c   = 3e5      # Speed of light in km/s
lyA = 1215.6   # Wavelength of LyA (Angstrom)
lV  = 2500     # Velocity window for line emission
cV  = 2000     # Additional velocity window for continuum emission

# Take minimum input 
paramPath = sys.argv[1]
cubeType  = sys.argv[2]

# Take any additional input params, if provided
settings = {"level":"coadd","line":"lyA"}
if len(sys.argv)>3:
    for item in sys.argv[3:]:      
        key,val = item.split('=')
        if settings.has_key(key): settings[key]=val
        else:
            print "Input argument not recognized: %s" % key
            sys.exit()
            
# Load parameters
params = libs.params.loadparams(paramPath)

# Check if parameters are complete
libs.params.verify(params)

# Get filenames     
if settings["level"]=="coadd":   files = [ '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubeType) ]
elif settings["level"]=="input":
    
    files = libs.io.findfiles(params,cubeType)   
    for i in range(len(params["IMG_ID"])):
        if params["SKY_ID"][i]!=params["IMG_ID"][i]:
            files.append(files[i].replace(params["IMG_ID"][i],params["SKY_ID"][i]))
            
else:
    print("Setting 'level' must be either 'coadd' or 'input'. Exiting.")
    sys.exit()

# Run through files to be cropped
for fileName in files:
    
    # Open FITS and extract info
    fits = fitsIO.open(fileName)
    h = fits[0].header
    w,y,x = fits[0].data.shape
    WW = np.array([ h["CRVAL3"] + h["CD3_3"]*(i - h["CRPIX3"]) for i in range(w)])

    # Calculate wavelength range
    if settings["line"]=="lyA":
        centerWav = (1+params["ZLA"])*lyA
    else:
        print("Setting - line:%s - not recognized."%settings["line"])
        sys.exit()
    
    # Get upper and lower indices of wavelength
    deltaWav = centerWav*(lV+cV)/c   
    w1,w2 = centerWav-deltaWav, centerWav+deltaWav
    a,b = libs.cubes.getband(w1,w2,h)
    a = max(0,a)
    b = min(w-1,b)

    #Save FITS of cropped data
    cropName = fileName.replace(".fits",".%s.fits" % settings["line"])
    fits[0].data = fits[0].data[a:b]
    fits[0].header["CRPIX3"] -= a
    fits.writeto(cropName,overwrite=True)
    print("Saved %s."%cropName)

    #If this is an "icube" file - also try to crop the corresponding variance cube
    if "icube" in fileName:
        varName = fileName.replace("icube","vcube")
        try: varFITS = fitsIO.open(varName)
        except: 
            print("Could not open %s to crop it."%varName)
            continue
        varFITS[0].data = varFITS[0].data[a:b]
        varFITS[0].header["CRPIX3"] -= a
        varCropName = varName.replace(".fits",".%s.fits" % settings["line"])
        varFITS.writeto(varCropName,overwrite=True)
        print("Saved %s."%varCropName)
    
#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))        
