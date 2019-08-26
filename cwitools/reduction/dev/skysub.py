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

libs.params.verify(params)

isNAS = np.array([ params["IMG_ID"][i]==params["SKY_ID"][i] for i in range(len(params["IMG_ID"]))])
if (isNAS==True).all():
    print("All input data is NAS (IMG ID same as SKY ID). Skipping sky subtraction.")
    sys.exit()

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
   
#Open FITS objects
ifits  = [fitsIO.open(f) for f in ifiles  ] 
sfits  = [fitsIO.open(f) for f in sfiles  ] 
ivfits = [fitsIO.open(f) for f in ivfiles ] 
svfits = [fitsIO.open(f) for f in svfiles ] 

#Get source mask from region file if available
if params["REG_FILE"]!='None':
    regFile = pyregion.open(params["REG_FILE"])
    maskSrc = True
else:
    print("No region file provided in param file. Sources will not be masked.")

#Get fitter for sky subtraction optimization            
fitter = fitting.LevMarLSQFitter()

#Crop to overlapping/good wavelength ranges
for i,f in enumerate(ifits):


    objData = ifits[i][0].data.copy()
    skyData = sfits[i][0].data.copy()
    
    if maskSrc: mask2d  = libs.cubes.get_regMask(f,regFile,scaling=2,binary=True)
    else:mask2d = np.zeros_like(objData)

    for wi in range(len(objData)): objData[wi][mask2d>0] = 0
    
    if params['INST'][i]=="PCWI":

        if f[0].header["BUNIT"]=='FLAM':
            objData *= 1e18
            skyData *= 1e18
        
        for yi in range(objData.shape[1]):
            
            goodPix = np.sum(mask2d[yi]==0)
        
            #Get mean int and sky spectra (using non-masked pixels)
            sliceInt = np.sum(objData[:,yi,:].copy(),axis=1)/goodPix
            sliceSky = np.sum(skyData[:,yi,:].copy(),axis=1)/goodPix

            #Astropy custom fittable model
            @custom_model
            def empSkyModel(x, A=1):
                global sliceSky
                return A*sliceSky
                
            #Initial guess is based on exposure times for non-flux calibrated data only
            if "cubes" in cubetype: p0=1
            else: p0 = ifits[i][0].header["EXPTIME"]/sfits[i][0].header["EXPTIME"]

            X = np.arange(len(sliceInt))
            
            mod0 = empSkyModel(A=p0)
            mod1 = fitter(mod0,X,sliceInt)
            A1 = mod1.A.value

            
            #Apply model to raw intensity and variance data
            for xi in range(objData.shape[2]): ifits[i][0].data[:,yi,xi] -= A1*sliceSky/1e18
            
            if updateVariance: ivfits[i][0].data[:,yi,:] += (A1**2)*svfits[i][0].data[:,yi,:]

        skyMask = libs.cubes.get_skyMask(ifits[i],params["INST"][i])
        
        #ifits[i][0].data[skyMask==1] = 0# np.median(ifits[i][0].data[skyMask==0],axis=0)
        if updateVariance: ivfits[i][0].data[skyMask==1] = 0
            
    print ""
    
    ifits[i].writeto(ifiles[i].replace('.fits','.ss.fits'),overwrite=True)
    if updateVariance: ivfits[i].writeto(ivfiles[i].replace('.fits','.ss.fits'),overwrite=True)
    
    print("Wrote %s" % ifiles[i].replace('.fits','.ss.fits'))
    if updateVariance: print("Wrote %s"%ivfiles[i].replace('.fits','.ss.fits'))

