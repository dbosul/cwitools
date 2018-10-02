#!/usr/bin/env python
#
# Fits3D: An extension of the astropy.io.fits class with methods for rotating, cropping, scaling 3D data while updating 3D WCS
#
# Inherits from astropy.io.fits.HDUList and adds an astropy.wcs.WCS object as a class attribute.
# The purpose of this class is to provide 3D transformation functions (crop, scale, rotate, translate)
# which also update the 3D WCS information accordingly.

# AXES 1,2,3 are referred to as X, Y, W in this class.
# Numpy loads the data in the shape [W,Y,X]
#
# Assumptions:
# 1. CRVAL1 = RA (dd.ddd) and CRVAL2 = DEC (dd.ddd)
# 2. WCS axis info is given as CD1_1, CD2_2 etc. 

from astropy.io import fits
from copy import deepcopy
from scipy.ndimage.interpolation import rotate as scipy_rotate
from scipy.ndimage.interpolation import zoom as scipy_zoom
import numpy as np
from numpy import cos,sin
import sys
import matplotlib.pyplot as plt


def open(filename): return fits3D(filename)

class fits3D(fits.HDUList):

    ##
    ## Simply load FITS class and make short-hand class attributes for data + header. 
    ##        
    def __init__(self,filename):
    
        fits.HDUList.__init__(self,fits.open(filename))
        
        self[0].data = self[0].data
        self[0].header = self[0].header


    ##
    ## This method is written only considering PAs of 0,90,180,270. 
    ##
    def scale1to1(self,splineorder=0):
       
        #Get CD matrix (absolute) values for calculating aspect ratio and zoom factors
        cd11,cd12,cd21,cd22 = [abs(self[0].header[s]) for s in ["CD1_1","CD1_2","CD2_1","CD2_2"]]

        # Get aspect ratio and zoom value for each axis

        if cd11!=0 and cd22!=0:        
            r = max(cd11,cd22)/min(cd11,cd22) 
            z = [1,r,1] if cd11<cd22 else [1,1,r]
            
        else:            
            r = max(cd12,cd21)/min(cd12,cd21) 
            z = [1,r,1] if cd21<cd12 else [1,1,r]

        
        # Get scaled data    
        self[0].data = scipy_zoom(self[0].data,z,order=1)/r
        #z = [ int(zi) for zi in z] #Cast to int
        #self[0].data = self.scale_cube(self[0].data,r,axis=1,var=True)

        #
        # Modify central reference pixels
        # -- Keeping in mind, for WCS - (1,1) points to center of first pixel in image
        # 
        self[0].header["CRPIX1"] = z[2]*self[0].header["CRPIX1"] - z[2]/2.0 + 0.5
        self[0].header["CRPIX2"] = z[1]*self[0].header["CRPIX2"] - z[1]/2.0 + 0.5
 
        #Modify plate scales
        self[0].header["CD1_1"] /= z[2]
        self[0].header["CD2_1"] /= z[2]
        self[0].header["CD1_2"] /= z[1]
        self[0].header["CD2_2"] /= z[1]
            
    ##
    ## Rotate 3D FITS data and header +(N*90) degrees in Position Angle
    ##                  
    def rotate90(self,N=1):
        
        if N==0: return
        elif not type(N) is int:
            print("Only integer multiples of 90-degree rotations allowed. Skipping rotation.")
            return
            
        #Repeat 90-degree WCS transformation 'N' times to maintain accurate WCS
        for i in range(N):
        
            #Extract current header info
            cd11,cd12,cd21,cd22 = [self[0].header[s] for s in ["CD1_1","CD1_2","CD2_1","CD2_2"]]
            x0,y0 = self[0].header["CRPIX1"],self[0].header["CRPIX2"]
            Y, X  = self[0].data.shape[1:]
                                      
            #Update header : CD Matrix        
            self[0].header["CD1_1"] = -cd12
            self[0].header["CD1_2"] =  cd11
            self[0].header["CD2_1"] = -cd22
            self[0].header["CD2_2"] =  cd21  
                    
            #Update header : CR PIX
            self[0].header["CRPIX1"] = Y-y0+1
            self[0].header["CRPIX2"] = x0  
            
            #Rotate data +90
            self[0].data = np.rot90(self[0].data,k=1,axes=(2,1))


    #Crop cube with lower and upper limit tuples for each axis
    def crop(self,xx=(0,-1),yy=(0,-1),ww=(0,-1)):
      
        #Crop cube
        self[0].data = self[0].data[ww[0]:ww[1],yy[0]:yy[1],xx[0]:xx[1]]

        #Change RA reference pixel
        #self[0].header["CRVAL1"] += (xx[0]*self[0].header["CD1_1"] +yy[0]*self[0].header["CD1_2"])/np.cos(self[0].header["CRVAL2"]*np.pi/180)
        self[0].header["CRPIX1"] -= xx[0]
        
        #Change DEC reference pixel
        #self[0].header["CRVAL2"] += xx[0]*self[0].header["CD2_1"] +yy[0]*self[0].header["CD2_2"]
        self[0].header["CRPIX2"] -= yy[0]
             
        #Change WAV reference pixel
        #self[0].header["CRVAL3"] += ww[0]*self[0].header["CD3_3"]
        self[0].header["CRPIX3"] -= ww[0]
        
    #Save as astropy.io.fits.HDUList object
    def save(self,path):   
        hdu = fits.PrimaryHDU(self[0].data)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = self[0].header        
        hdulist.writeto(path,overwrite=True)

    #Method for scaling cubes to 1:1 given aspect ratio (r) and short axis (axis)
    def scale_cube(self,a,r,axis=1,var=False,m=2):

        r = int(r)
        
        new_shape = np.copy(a.shape)
        new_shape[axis] *= r
        new_cube = np.zeros(new_shape)
        
        for i in range(new_shape[axis]):
            if axis==1: new_cube[:,i,:] = a[:,i/r,:]/r
            elif axis==2: new_cube[:,:,i] = a[:,:,i/r]/r

        print np.sum(new_cube)/np.sum(a)
        return new_cube



