from astropy.io import fits as fitsIO
from astropy.wcs import WCS
import numpy as np
import pyregion
import sys

import libs


maskpath = sys.argv[1]
fitspath = sys.argv[2]

print maskpath
print fitspath

fits = fitsIO.open(fitspath)
mask = fitsIO.open(maskpath)[0].data

masked_cube = libs.cubes.apply_mask(fits[0].data,mask)

hdulist = fitsIO.HDUList([fitsIO.PrimaryHDU(masked_cube)])
hdulist[0].header = fits[0].header
hdulist.writeto(fitspath.replace('.fits','.m.fits'),overwrite=True)

print "Saved masked image: %s" % fitspath.replace('.fits','.m.fits')
