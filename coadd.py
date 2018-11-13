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

settings = {"trim_mode":'nantrim','vardata':False}
if len(sys.argv)>3:
    for item in sys.argv[3:]:
        
        key,val = item.split('=')
        if settings.has_key(key): settings[key]=val
        else:
            print "Input argument not recognized: %s" % key
            sys.exit()

print "Settings:"
for s in settings.keys():
    print "\t%10s: %s" % (s,settings[s])

#Set flag for whether params need to be updated
setupMode = False

#Add file extension of omitted
if not ".fits" in cubetype: cubetype += ".fits"

#Check if any parameter values are missing (set to set-up mode if so)
params = libs.params.loadparams(parampath)

#Check if parameters are complete
libs.params.verify(params)

#Get filenames     
files = libs.io.findfiles(params,cubetype)

#Open custom FITS-3D objects
fits = [libs.fits3D.open(f) for f in files] 
       
#Make all data products in 10^16 Flam
for i,f in enumerate(fits):
    if params['INST'][i]=='PCWI' and f[0].header["BUNIT"]=='FLAM':
        if settings["vardata"]: f[0].data *= (1e16)**2
        else: f[0].data *= 1e16
        f[0].header["BUNIT"] = 'FLAM16'
       
#Stack cubes and trim
stackedFITS = libs.cubes.coadd(fits,params,settings)  

#Add redshift info to header
stackedFITS[0].header["Z"] = params["Z"]
stackedFITS[0].header["ZLA"] = params["ZLA"]

#Update target parameter file (if in setup mode)
if setupMode: libs.params.writeparams(params,parampath)

#Save stacked cube
stackedpath = '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubetype)
stackedFITS[0].writeto(stackedpath,overwrite=True)
print "Saved %s" % stackedpath

#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))
