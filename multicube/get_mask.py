from astropy.io import fits as fitsIO
from astropy.wcs import WCS
import numpy as np
import pyregion
import sys

import libs


fitspath = sys.argv[1]
regpath = sys.argv[2]
if len(sys.argv)>3: S = float(sys.argv[3])
else: S = 0.75

fits = fitsIO.open(fitspath)
regfile = pyregion.open(regpath)

mask = libs.cubes.get_mask(fits,regfile,scaling=S)

hdulist = fitsIO.HDUList([fitsIO.PrimaryHDU(mask)])
hdulist[0].header = fits[0].header
hdulist.writeto(fitspath.replace('.fits','.mask.fits'),overwrite=True)

print "Wrote mask image to %s" % fitspath.replace('.fits','.mask.fits')
