#!/usr/bin/env python
#
# bkgSub - Fit low order polynomial to spectrum in every spaxel after median-filtering continuum sources.
# 
# syntax: python bkgSub.py <parameterFile> <cubeType>
#

import numpy as np
import pyregion
import scipy as sc
import sys

from multicube import libs

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Load pipeline parameters
params = libs.params.loadparams(parampath)
  
#Open FITS files 
stackedpath = '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubetype)
S = libs.fits3D.open(stackedpath)

#Open regionfile
regpath = params["REG_FILE"]
if regpath=='None': print "WARNING: No region file specified in %s. Sources will not be masked."
else: regfile = pyregion.open(regpath)

skylines = [[4353,4364],[4040,4050]]
   
print "\nRunning polynomial subtraction on %s" % stackedpath

#Get for region file mask for this fits
if regpath!="None": 

    #Get 2D Mask based on region file
    regmask = libs.cubes.get_mask(S,regfile)
    
    #Apply median mask to sources    
    cube_masked = libs.cubes.apply_mask(S[0].data.copy(),regmask,mode='xmedian',inst=params["INST"][0])

#Just use unmasked cube if no region file provided
else: cube_masked = f[0].data

#Run cube-wide polyfit to subtract scattered light

W = np.array([ S[0].header["CRVAL3"] + S[0].header["CD3_3"]*(k - S[0].header["CRPIX3"]) for k in range(S[0].data.shape[0])])

lyA = 1216*(params["ZLA"]+1)
dw = (500*1e5/3e10)*lyA
a,b  = libs.params.getband(lyA-dw,lyA+dw,S[0].header)
usewav = np.ones(S[0].data.shape[0],dtype='bool')
usewav[a:b] = 0

for skyline in skylines:
    wC,wD = skyline
    c,d  = libs.params.getband(wC,wD,S[0].header)
    if 0<c<W[-1] and 0<d<W[-1]: usewav[c:d] = 0

polyfit = libs.continuum.polyModel(cube_masked,usewav,inst=params["INST"][0])

#Subtract Polynomial continuum model from cube
S[0].data -= polyfit


#Save file
savename = stackedpath.replace('.fits','_polysub.fits')
S.save(savename)
print "Saved %s" % savename


