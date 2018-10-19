#!/usr/bin/env python
#
# bkgSub - Fit low order polynomial to spectrum in every spaxel after median-filtering continuum sources.
# 
# syntax: python bkgSub.py <parameterFile> <cubeType>
#
from astropy.io import fits as fitsIO
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt

import numpy as np
import pyregion
import scipy as sc
import sys

import libs

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

if len(sys.argv)>3: S = float(sys.argv[3])
else: S = 1

medW = 20

#Load parameters
params = libs.params.loadparams(parampath)

#Get filenames
files = libs.io.findfiles(params,cubetype)

#Open FITS files 
fits = [fitsIO.open(f) for f in files] 

#Open Region File
regFile = pyregion.open(params["REG_FILE"])

#Run through fits files
for i,f in enumerate(fits):
 
    #Get Masks
    regMask = libs.cubes.get_regMask(f,regFile,scaling=S)
    skyMask = libs.cubes.get_skyMask(f)
    
    print "\nSubtracting continuum from %s" % files[i]
    
    #Filter NaNs and INFs from cube
    fits[i][0].data = np.nan_to_num(f[0].data) 
    cube = fits[i][0].data
    
    #Run cube-wide polyfit to subtract scattered light
    wcrop = tuple(int(w) for w in params["WCROP"][i].split(':'))
    W = np.array([ f[0].header["CRVAL3"] + f[0].header["CD3_3"]*(k - f[0].header["CRPIX3"]) for k in range(f[0].data.shape[0])])

    #Calculate LyA+/2000km/s indices
    lyA = 1216*(params["ZLA"]+1)
    dw = (2000*1e5/3e10)*lyA
    a,b  = libs.params.getband(lyA-dw,lyA+dw,f[0].header)

    #Make copy of sky mask and add LyA emission + bad wavelengths
    wavMaski = skyMask.copy() 
    wavMaski[:wcrop[0]+1] = 1
    wavMaski[wcrop[1]:]   = 0
    wavMaski[a:b] = 0

    #Apply spatial mask if given
    cubeM = libs.cubes.apply_mask(cube.copy(),regMask,mode="xmedian",inst=params["INST"][i])
    
    #Run polynomial fit, passing wavelength mask to function
    polyfit = libs.continuum.polyModel(cubeM,mask1D=wavMaski,inst=params["INST"][i])
    
    #Subtract Polynomial continuum model from cube
    f[0].data -= polyfit

    #Replace sky lines with median values
    print("\tMedian filtering sky-lines.")
    for a,sm in enumerate(skyMask):
        
        if sm==1:
            b = a+1
            while( skyMask[b]==1 and b<len(skyMask)-1 ): b+=1 #Run to end of masked region
            f[0].data[a:b] = np.median(cube[a-medW:b+medW],axis=0)#Median filter region
            a = b+1 #Move on to next part of spectrum
            
    #Save file
    savename = files[i].replace('.fits','.bs.fits')
    f.writeto(savename,overwrite=True)
    print "Saved %s" % savename
    
    f[0].data = polyfit
    polyname = files[i].replace('.fits','.poly.fits')
    f.writeto(polyname,overwrite=True)
    print "Saved %s" % polyname

