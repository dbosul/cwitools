from astropy.io import fits as fitsIO
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

import libs

plt.style.use('ggplot')

# Timer start
tStart = time.time()

# Define some constants
c   = 3e5       # Speed of light in km/s
lyA = 1215.6    # Wavelength of LyA in Angstrom

# Take minimum input 
paramPath = sys.argv[1]
cubeType  = sys.argv[2]

# Take any additional input params, if provided
settings = {"level":"coadd","line":"lyA","dv":1000}
if len(sys.argv)>3:
    for item in sys.argv[3:]:      
        key,val = item.split('=')
        if settings.has_key(key):
            if key=="dv": val=int(val)
            settings[key]=val
        else:
            print "Input argument not recognized: %s" % key
            sys.exit()
            
# Load parameters
params = libs.params.loadparams(paramPath)

# Check if parameters are complete
libs.params.verify(params)

# Get filenames     
fileName = '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubeType)

# Open FITS
fits = fitsIO.open(fileName)
data = fits[0].data
hdr  = fits[0].header
W    = libs.cubes.getWavAxis(hdr)

# Use WCS to get pixel scales
wcs  = WCS(hdr)
xscale,yscale,wscale = proj_plane_pixel_scales(wcs)
xscale *= 3600
yscale *= 3600
wscale *= 1e10

# Get central wavelength to use
if settings["line"]=="lyA": wL = (1+params["ZLA"])*lyA
else:
    print("Input line (%s) not recognized. May not be added to code yet.")
    sys.exit() 

# Get wavelength range for NB
dw = wL*(settings["dv"]/c)
w0 = wL - dw
w1 = wL + dw

# Get indices and sum over wavelengths
useW = np.ones_like(W,dtype=bool)
useW[W<w0] = 0
useW[W>w1] = 0
NB = np.sum(data[useW],axis=0)

# Convert NB from erg/cm2/s/A to ergs/s/cm^2/arcsec2
dW = np.sum(useW)*wscale
pxlArea = xscale*yscale

NB /= pxlArea
NB /= 1e16
NB = gaussian_filter(NB,1.0)

std = np.std(NB)

T = 2
det = NB>T*std

for wi in range(data.shape[0]): data[wi][det==False] = 0

spec = np.sum(data,axis=(1,2))
spec = gaussian_filter(spec,1)
plt.figure()
plt.subplot(211)
plt.pcolor(NB,vmin=0,vmax=5e-17)
plt.xlim([0,data.shape[2]])
plt.ylim([0,data.shape[1]])
plt.colorbar()
plt.subplot(212)
plt.plot(W,spec,'kx-')
plt.xlim([W[0],W[-1]])
plt.tight_layout()
plt.show()
#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))                    
