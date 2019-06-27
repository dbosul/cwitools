#!/usr/bin/env python

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

#Get filenames
files = libs.io.findfiles(params,cubetype)

#Open custom FITS-3D objects
fits = [fitsIO.open(f) for f in files] 

#1 - verify the parameter files
libs.params.verify(params)

#2 - re-initialize param values from FITS headers
params = libs.params.parseHeaders(params,fits)


#3 - overwrite param file
params = libs.params.writeparams(params,parampath)

raw_input("")
