from astropy.io import fits

import numpy as np
import os
import sys
from scipy.ndimage.filters import gaussian_filter

fitspath = os.path.abspath(sys.argv[1])
sig = float(sys.argv[2])

f = fits.open(fitspath)

f[0].data = gaussian_filter(f[0].data,[0,sig,sig])

oname = fitspath.replace('.fits','_s.fits')

f.writeto(oname,overwrite=True)

print "Saved %s" % oname
