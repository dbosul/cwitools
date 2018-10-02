#!/usr/bin/env python
#
# bkgSub - Fit low order polynomial to spectrum in every spaxel after median-filtering continuum sources.
# 
# syntax: python bkgSub.py <parameterFile> <cubeType>
#
from scipy.ndimage.filters import gaussian_filter

import numpy as np
import pyregion
import scipy as sc
import sys

import libs

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Load pipeline parameters
params = libs.params.loadparams(parampath)
  
#Get filenames
files = libs.io.findfiles(params,cubetype)

#Open FITS files 
fits = [libs.fits3D.open(f) for f in files] 

#Open regionfile
regpath = params["REG_FILE"]
if regpath=='None': print "WARNING: No region file specified in %s. Sources will not be masked."
else: regfile = pyregion.open(regpath)

skylines = [[4353,4364],[4040,4050]]

#Subtract continuum sources
for i,f in enumerate(fits):
    
    print "\nSubtracting continuum from %s" % files[i]
    
    #Filter NaNs and INFs from cube
    print "\tFiltering NaNs/INFs"
    fits[i][0].data = np.nan_to_num(f[0].data) 

    #Get for region file mask for this fits
    if regpath!="None" and 1: 
    
        print "Mask"
        #Get 2D Mask based on region file
        regmask = libs.cubes.get_mask(f,regfile)
        
        #Apply median mask to sources    
        cube_masked = libs.cubes.apply_mask(f[0].data.copy(),regmask,mode='xmedian',inst=params["INST"][i])

    #Just use unmasked cube if no region file provided
    else: cube_masked = f[0].data
    
    #cube_masked = gaussian_filter(cube_masked,3.0)
    
    #Run cube-wide polyfit to subtract scattered light
    wcrop = tuple(int(w) for w in params["WCROP"][i].split(':'))
    W = np.array([ f[0].header["CRVAL3"] + f[0].header["CD3_3"]*(k - f[0].header["CRPIX3"]) for k in range(f[0].data.shape[0])])

    lyA = 1216*(params["ZLA"]+1)
    dw = (2000*1e5/3e10)*lyA
    a,b  = libs.params.getband(lyA-dw,lyA+dw,f[0].header)

    #print lyA-dw,lyA+dw
    usewav = np.ones(f[0].data.shape[0],dtype='bool')
    usewav[:wcrop[0]+1] = 0
    usewav[wcrop[1]:] = 0
    usewav[a:b] = 0

    #for skyline in skylines:
    #    wC,wD = skyline
    #    c,d  = libs.params.getband(wC,wD,f[0].header)
    #    if 0<c<W[-1] and 0<d<W[-1]: usewav[c:d] = 0

    
    polyfit = libs.continuum.polyModel(cube_masked,usewav,inst=params["INST"][i])
    
    #Subtract Polynomial continuum model from cube
    f[0].data -= polyfit#np.median(cube_masked[usewav])

    #Save file
    savename = files[i].replace('.fits','.bs.fits')
    f.save(savename)
    print "Saved %s" % savename
    
    f[0].data = polyfit
    polyname = files[i].replace('.fits','.poly.fits')
    f.save(polyname)

