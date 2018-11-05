#!/usr/bin/env python
#
# skySub - find best scaling and subtract sky images from data
# 
# syntax: python skySub.py <parameterFile> <cubeType> (e.g. icuber, icubes, icubep)
#
import matplotlib.pyplot as plt
import numpy as np
import pyregion
import sys

from astropy.io import fits as fitsIO
from astropy.modeling import models,fitting
from astropy.modeling.models import custom_model
from scipy.optimize import least_squares

import libs

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

updateVariance = True

#Add file extension of omitted
if not ".fits" in cubetype: cubetype += ".fits"

#Check if any parameter values are missing (set to set-up mode if so)
params = libs.params.loadparams(parampath)

#Get filenames     
ifiles  = libs.io.findfiles(params,cubetype,getSky=True)
ivfiles = [ f.replace("icube","vcube") for f in ifiles ]
sfiles  = [ f.replace(params["IMG_ID"][i],params["SKY_ID"][i]) for i,f in enumerate(ifiles) ]
svfiles = [ f.replace("icube","vcube") for f in sfiles ] 

#Check input data is available before continuing
if "" in ifiles:
    print "Some input files not found. Check paramfile and try again.\n\n"
    sys.exit()
if "" in sfiles:
    print "Some sky files not found. Check paramfile and try again.\n\n"
    sys.exit()
if "" in ivfiles or "" in svfiles:
    print "Some variance files not found - variance data will not be updated."
    updateVariance=False 
   
#Open custom FITS-3D objects
ifits = [fitsIO.open(f) for f in ifiles] 
sfits = [fitsIO.open(f) for f in sfiles] 
ivfits = [fitsIO.open(f) for f in ivfiles] 
svfits = [fitsIO.open(f) for f in svfiles] 

#Check if parameters are complete
if libs.params.paramsMissing(params):

    #Enter set-up mode
    setupMode = True
    
    #Parse FITS headers for PA, instrument, etc.
    params = libs.params.parseHeaders(params,fits)

    #Write params to file
    libs.params.writeparams(params,parampath)

#Get source mask from region file if available
if params["REG_FILE"]!='None':
    regFile = pyregion.open(params["REG_FILE"])
    maskSrc = True
else:
    print("No region file provided in param file. Sources will not be masked.")


            
fitter = fitting.LevMarLSQFitter()

#Crop to overlapping/good wavelength ranges
for i,f in enumerate(ifits):

    #Crop FITS
    x0,x1 = tuple(int(x) for x in params["XCROP"][i].split(':'))
    y0,y1 = tuple(int(y) for y in params["YCROP"][i].split(':'))
    w0,w1 = tuple(int(w) for w in params["WCROP"][i].split(':'))

        
    objData = ifits[i][0].data.copy()
    skyData = sfits[i][0].data.copy()
    
    if maskSrc: mask2d  = libs.cubes.get_regMask(f,regFile)
    else:mask2d = np.zeros_like(objData)
    
    if params['INST'][i]=="PCWI":

        if f[0].header["BUNIT"]=='FLAM':
            objData *= 1e18
            skyData *= 1e18
        
        for yi in range(objData.shape[1]):
            
            #Extract 2D slice
            sliceInt = objData[:,yi,:]
            sliceSky = skyData[:,yi,:]
            
            #Flatten 2D slice to 1D
            sliceIntFlat = sliceInt.flatten()
            sliceSkyFlat = sliceSky.flatten()
            
            #Mask cropped W,X indices for fitting
            sliceMsk = np.zeros_like(sliceInt)
            sliceMsk[:w0] = 1
            sliceMsk[w1:] = 1
            sliceMsk[:,:x0] = 1
            sliceMsk[:,x1:] = 1
            sliceMsk[:,mask2d[yi]] = 1
            sliceMskFlat = sliceMsk.flatten()
            
            
            #objFunc  = lambda x: sliceInt.flatten() - x[0]*sliceSky.flatten()
            
            #Astropy custom fittable model
            @custom_model
            def empSkyModel(x, A=1):
                global sliceSkyFlat
                return A*sliceSkyFlat
                
            #Initial guess is based on exposure times for non-flux calibrated data only
            if "cubes" in cubetype: p0=1
            else: p0 = ifits[i][0].header["EXPTIME"]/sfits[i][0].header["EXPTIME"]

            X = np.arange(len(sliceIntFlat))
            
            mod0 = empSkyModel(A=p0)
            mod1 = fitter(mod0,X,sliceIntFlat)
            A1 = mod1.A.value

            #Apply model to raw intensity and variance data
            ifits[i][0].data[:,yi,:] -= A1*sfits[i][0].data[:,yi,:]
            if updateVariance: ivfits[i][0].data[:,yi,:] += (A1**2)*svfits[i][0].data[:,yi,:]

    print ""
    
    ifits[i].writeto(ifiles[i].replace('.fits','.ss.fits'),overwrite=True)
    if updateVariance: ivfits[i].writeto(ivfiles[i].replace('.fits','.ss.fits'),overwrite=True)
    
    print("Wrote %s" % ifiles[i].replace('.fits','.ss.fits'))
    if updateVariance: print("Wrote %s"%ivfiles[i].replace('.fits','.ss.fits'))

