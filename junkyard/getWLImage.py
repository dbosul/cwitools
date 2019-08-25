from astropy.io import fits as fitsIO
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.stats import sigmaclip

import numpy as np
import sys
import time

from CWITools import libs

#Timer start
tStart = time.time()

#Take file path as input
filePath = sys.argv[1]

#Open fits, get WCS and pixel scales
fits = fitsIO.open(filePath)
h2d  = libs.cubes.get2DHeader(fits[0].header)
wcs = WCS(h2d)
pSc = proj_plane_pixel_scales(wcs)
pSc*= 3600
pxArea = pSc[0]*pSc[1]
dLam = len(fits[0].data)*fits[0].header["CD3_3"]

#Modify and sum data to be in units of 10^-18 (erg/s/cm2/arcsec2)
fits[0].data *= 1e2
fits[0].data = np.sum(fits[0].data,axis=0)
fits[0].data /= pxArea
fits[0].data /= dLam
clipped = sigmaclip(fits[0].data).clipped
med = np.median(clipped)
fits[0].data -= med
fits[0].header = h2d
fits.writeto(filePath.replace('.fits','.WL.fits'),overwrite=True)

print filePath,np.std(clipped)
#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))
