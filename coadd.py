#!/usr/bin/env python
#
# coadd - Crop, Scale, Rotate (90deg only), Align and Coadd individual CWI observations.
# 
# syntax: python coadd.py <parameterFile> <cubeType>
#

from astropy.io import fits as fitsIO
import numpy as np
import sys
import time
import libs

#Timer start
tStart = time.time()

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Add file extension of omitted
if not ".fits" in cubetype: cubetype += ".fits"

#Check if any parameter values are missing (set to set-up mode if so)
params = libs.params.loadparams(parampath)

#Check if parameters are complete
libs.params.verify(params)

#Get filenames     
files = libs.io.findfiles(params,cubetype)

#Stack cubes and trim
stackedFITS,varFITS = libs.cubes.coadd(files,params)  

#Add redshift info to header
stackedFITS[0].header["Z"] = params["Z"]
stackedFITS[0].header["ZLA"] = params["ZLA"]

#Save stacked cube
stackedpath = '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubetype)
stackedFITS[0].writeto(stackedpath,overwrite=True)
print "\nSaved %s" % stackedpath

#Save variance cube if one was returned
if varFITS!=None:
    varpath = '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubetype.replace("icube","vcube"))
    varFITS[0].writeto(varpath,overwrite=True)
    print "Saved %s" % varpath    
    
#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))
