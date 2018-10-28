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

tStart = time.time()

settings = {"trim_mode":'nantrim','vardata':False}

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

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

#Get filenames     
files = libs.io.findfiles(params,cubetype)

if "" in files or files==[]:
    print "Some files not found. Please correct paramfile (check data dir and image IDs) and try again.\n\n"
    sys.exit()
    
#Open custom FITS-3D objects
fits = [libs.fits3D.open(f) for f in files] 

#### Temporary: Filter NaNs and INFs to at least avoid errors
for f in fits: f[0].data = np.nan_to_num(f[0].data)

#Check if parameters are complete
if libs.params.paramsMissing(params):

    #Enter set-up mode
    setupMode = True
    
    #Parse FITS headers for PA, instrument, etc.
    params = libs.params.parseHeaders(params,fits)

    #Write params to file
    libs.params.writeparams(params,parampath)

else:
    
    #Update WCS of each image to accurately point to SRC (e.g. QSO)
    for i,f in enumerate(fits):
        
        f[0].header["CRPIX1"] = params["SRC_X"][i]
        f[0].header["CRPIX2"] = params["SRC_Y"][i]
        f[0].header["CRVAL1"] = params["RA"]
        f[0].header["CRVAL2"] = params["DEC"]
        

#Over-write fits files with fixed WCS
for i,f in enumerate(fits): f.save(files[i])

#Make all data products in 10^16 Flam
for i,f in enumerate(fits):
    if params['INST'][i]=='PCWI' and f[0].header["BUNIT"]=='FLAM':
        if settings["vardata"]: f[0].data *= (1e16)**2
        else: f[0].data *= 1e16
        f[0].header["BUNIT"] = 'FLAM16'

#Crop FITS
print("Cropping cubes."),
for i,f in enumerate(fits):
    print("."),     
    xcrop = tuple(int(x) for x in params["XCROP"][i].split(':'))
    ycrop = tuple(int(y) for y in params["YCROP"][i].split(':'))
    wcrop = tuple(int(w) for w in params["WCROP"][i].split(':'))
    f.crop(xx=xcrop,yy=ycrop,ww=wcrop)  
print("")

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

tFinish = time.time()

print("Elapsed time: %.2f seconds" % (tFinish-tStart))

