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

import pyregion
import os
import sys
import tools #Custom module

#INPUT#################################
fitspath = os.path.abspath(sys.argv[1])
regpath = os.path.abspath(sys.argv[2])

#USER-SET VARS#########################
maskvalue = 0

#OPEN FILES ###########################
print "Input FITS:%s\nInput Mask:%s" % (fitspath,regpath)
ifits = fitsIO.open(fitspath)
regfile = pyregion.open(regpath)

#GET 2D MASK###########################
mask = tools.cubes.get_mask(ifits,regfile)

#APPLY MASK TO DATA#############################################
for wi in range(ifits[0].data.shape[0]): ifits[0].data[wi][mask==1] = maskvalue

#OUTPUT MASK AND MASKED FITS####################################
mfitspath = regpath.replace('.reg','_mask.fits')
ifitspath = fitspath.replace('.fits','_m.fits')
print "\nOutput FITS: %s\nOutput Mask FITS: %s\n" % (ifitspath,mfitspath)
mfits = fitsIO.HDUList([fitsIO.PrimaryHDU(mask)])
mfits[0].header = ifits[0].header
mfits.writeto(regpath.replace('.reg','_mask.fits'),clobber=True)
ifits.writeto(ifitspath,clobber=True)



