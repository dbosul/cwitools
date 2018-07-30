# Crop in WAVELENGTH
# Arguments:
#   1. Input Cube
#   2. Redshift of Target

from astropy.io import fits as fitsIO

import numpy as np
import sys

#Define constants (cgs)
km  = 1e5       # cm->km conversion
c   = 3e10      # Speed of light in cm/s
lyA = 1215.6    # Wavelength of LyA (Angstrom)
dv  = 2000*km   # Velocity window

#Take input 
fitspath = sys.argv[1]
z  = float(sys.argv[2])

#Open FITS and extract info
fits = fitsIO.open(fitspath)
h = fits[0].header
W,Y,X = fits[0].data.shape

#Calculate wavelength range
wc = (1+z)*lyA
dwav = wc*(dv/c)
w1,w2 = wc-dwav, wc+dwav

#Calculate indices in cube (bounded by ends of array)
#Return indices for a given wavelength band (w1,w2) in angstroms
def getband(_w1,_w2,_hd):
	w0,dw,p0 = _hd["CRVAL3"],_hd["CD3_3"],_hd["CRPIX3"]
	w0 -= p0*dw
	return ( int((_w1-w0)/dw), int((_w2-w0)/dw) )
a,b = getband(w1,w2,h)

#Crop the cube
fits[0].data = fits[0].data[a:b]
fits[0].header["CRPIX3"] -= a

#Save new FITS file
savestring = fitspath.replace(".fits",".LyA.fits")
hdulist = fitsIO.HDUList([fitsIO.PrimaryHDU(fits[0].data)])
hdulist[0].header = fits[0].header
hdulist.writeto(savestring,overwrite=True)

#Save NB
fits[0].data = np.sum(fits[0].data,axis=0)
savestring = fitspath.replace(".fits",".LyA.NB.fits")
hdulist = fitsIO.HDUList([fitsIO.PrimaryHDU(fits[0].data)])
hdulist[0].header = fits[0].header
hdulist.writeto(savestring,overwrite=True)

