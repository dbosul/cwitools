#######################################
#
# MASK.PY
# 
# USE SYNTAX - python mask.py <fits_file_to_mask> <reg_file_to_use>
#
# Currently, the region file must be in image coordinates of the target FITS file.
# (i.e. not FK5/WCS)
#

from astropy.io import fits as fitsIO

import numpy as np
import os
import pyregion
import sys

#INPUT#################################
fitspath = os.path.abspath(sys.argv[1])
regpath = os.path.abspath(sys.argv[2])
#######################################

#1.CREATE 3D MASK #####################
ifits = fitsIO.open(fitspath)

regfile = pyregion.open(regpath)

mask = np.zeros_like(ifits[0].data)

W,Y,X = ifits[0].data.shape

for reg in regfile:

    x,y,r = reg.coord_list
    r+=1

    x = int(round(x))
    y = int(round(y))
    r = int(round(r))
    
    x0 = max(0,x-r)
    x1 = max(0,min(X,x+r))
    
    y0 = max(0,y-r)
    y1 = max(0,min(Y,y+r))
    
    for yi in range(y0,y1):
        for xi in range(x0,x1):           
            
            if ( (xi-x)**2 + (yi-y)**2 ) <= r**2:
            
                mask[:,yi,xi] = 1              
                
                ifits[0].data[:,yi,xi] = 0
    
mfits = fitsIO.HDUList([fitsIO.PrimaryHDU(mask)])
mfits[0].header = ifits[0].header
mfits.writeto(regpath.replace('.reg','_mask.fits'),clobber=True)

ifits.writeto(fitspath.replace('.fits','_m.fits'),clobber=True)



