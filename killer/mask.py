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
from astropy.wcs import WCS

import numpy as np
import os
import pyregion
import sys

import matplotlib.pyplot as plt

#INPUT#################################
fitspath = os.path.abspath(sys.argv[1])
regpath = os.path.abspath(sys.argv[2])

#USER-SET VARS#########################
maskvalue = 0

#OPEN FILES ###########################
print "Input FITS:%s\nInput Mask:%s" % (fitspath,regpath)
ifits = fitsIO.open(fitspath)
regfile = pyregion.open(regpath)

#EXTRACT/CREATE USEFUL VARS############
data3D = ifits[0].data
head3D = ifits[0].header
coordsys = regfile[0].coord_format
W,Y,X = data3D.shape
mask = np.zeros((Y,X))
x,y = np.arange(X),np.arange(Y) #Create X/Y image coordinate domains
xx, yy = np.meshgrid(y, x) #Create meshgrid of X, Y
    
#BUILD MASK############################
if coordsys=='image':

    for reg in regfile:
        x0,y0,R = reg.coord_list #Get position and radius
        rr = np.sqrt( (xx-x0)**2 + (yy-y0)**2 )
        mask[rr<=R] = 1            
                
elif coordsys=='fk5':  

    head2D = head3D.copy() #Create a 2D header by modifying 3D header
    for key in ["NAXIS3","CRPIX3","CD3_3","CRVAL3","CTYPE3","CNAME3","CUNIT3"]: head2D.remove(key)
    head2D["NAXIS"]=2
    head2D["WCSDIM"]=2
    wcs = WCS(head2D)    
    ra, dec = wcs.wcs_pix2world(xx, yy, 0) #Get meshes of RA/DEC
    
    for reg in regfile:    
        ra0,dec0,R = reg.coord_list        
        rr = np.sqrt( (np.cos(dec*np.pi/180)*(ra-ra0))**2 + (dec-dec0)**2 )     
        mask[rr < R] = 1

#APPLY MASK TO DATA#############################################
for wi in range(W): ifits[0].data[wi][mask==1] = maskvalue

#OUTPUT MASK AND MASKED FITS####################################
mfitspath = regpath.replace('.reg','_mask.fits')
ifitspath = fitspath.replace('.fits','_m.fits')
print "\nOutput FITS: %s\nOutput Mask FITS: %s" % (ifitspath,mfitspath)
mfits = fitsIO.HDUList([fitsIO.PrimaryHDU(mask)])
mfits[0].header = ifits[0].header
mfits.writeto(regpath.replace('.reg','_mask.fits'),clobber=True)
ifits.writeto(ifitspath,clobber=True)



