from astropy.io import fits
from copy import deepcopy
from scipy.ndimage.interpolation import rotate as scipy_rotate
from scipy.ndimage.interpolation import zoom as scipy_zoom
import numpy as np
from numpy import cos,sin
import sys


# FITS3D inherits from astropy.io.fits.HDUList and adds an astropy.wcs.WCS object as a class attribute.
# The purpose of this class is to provide 3D transformation functions (crop, scale, rotate, translate)
# which also update the 3D WCS information accordingly.

# AXES 1,2,3 are referred to as X, Y, W in this class.
# Numpy loads the data in the shape [W,Y,X]
#
# Assumptions:
# 1. CRVAL1 = RA (dd.ddd) and CRVAL2 = DEC (dd.ddd)
# 2. WCS axis info is given as CD1_1, CD2_2 etc. 

class fits3D(fits.HDUList):

    ##
    ## Simply load FITS class and make short-hand class attributes for data + header. 
    ##        
    def __init__(self,filename):
    
        fits.HDUList.__init__(self,fits.open(filename))
        
        self.data = self[0].data
        self.header = self[0].header


    ##
    ## This method is written only considering PAs of 0,90,180,270. 
    ##
    def scale1to1(self,splineorder=0):
       
        cd11,cd12,cd21,cd22 = [self.header[s] for s in ["CD1_1","CD1_2","CD2_1","CD2_2"]]

        # Get aspect ratio and zoom value for each axis
        if cd11!=0 and cd22!=0:        
            r = abs(max(cd11,cd22)/min(cd11,cd22)) 
            z = [1,r,1] if cd11<cd22 else [1,1,r]
            
        else:            
            r = max(cd12,cd21)/min(cd12,cd21) 
            z = [1,r,1] if cd21<cd12 else [1,1,r]
          
        # Get scaled data    
        self.data = scipy_zoom(self.data,z,order=splineorder)
        
        #
        # Modify central reference pixels
        # -- Keeping in mind, for WCS - (1,1) points to center of first pixel in image
        # 
        self.header["CRPIX1"] = z[2]*self.header["CRPIX1"] - z[2]/2.0 + 0.5
        self.header["CRPIX2"] = z[1]*self.header["CRPIX2"] - z[1]/2.0 + 0.5
 
        #Modify plate scales
        self.header["CD1_1"] /= z[2]
        self.header["CD2_1"] /= z[2]
        self.header["CD1_2"] /= z[1]
        self.header["CD2_2"] /= z[1]
            
    ##
    ## Rotate 3D FITS data and header +(N*90) degrees in Position Angle
    ##                  
    def rotate90(self,N=1):
        
        print "test1"
        #Repeat 90-degree WCS transformation 'N' times to maintain accurate WCS
        for i in range(N):
        
            #Extract current header info
            cd11,cd12,cd21,cd22 = [self.header[s] for s in ["CD1_1","CD1_2","CD2_1","CD2_2"]]
            x0,y0 = self.header["CRPIX1"],self.header["CRPIX2"]
            Y, X  = self.data.shape[1:]
                                      
            #Update header : CD Matrix        
            self.header["CD1_1"] = -cd12
            self.header["CD1_2"] =  cd11
            self.header["CD2_1"] = -cd22
            self.header["CD2_2"] =  cd21  
                    
            #Update header : CR PIX
            self.header["CRPIX1"] = Y-y0+1
            self.header["CRPIX2"] = x0  
            
            #Rotate data +90
            self.data = np.rot90(self.data,k=1,axes=(2,1))

            
    #Crop cube with lower and upper limit tuples for each axis
    def crop(self,xx=(0,-1),yy=(0,-1),ww=(0,-1)):
    
        #Crop cube
        self.data = self.data[ww[0]:ww[1],yy[0]:yy[1],xx[0]:xx[1]]

        #Change RA reference pixel
        self.header["CRVAL1"] += (xx[0]*self.header["CD1_1"] +yy[0]*self.header["CD1_2"])/np.cos(self.header["CRVAL2"]*np.pi/180)
        
        #Change DEC reference pixel
        self.header["CRVAL2"] += xx[0]*self.header["CD2_1"] +yy[0]*self.header["CD2_2"]
        
        #Change WAV reference pixel
        self.header["CRPIX3"] += ww[0]*self.header["CD3_3"]
        
           
               
    #Save as astropy.io.fits.HDUList object
    def save(self,path):   
        hdu = fits.PrimaryHDU(self.data)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = self[0].header        
        hdulist.writeto(path,clobber=True)





