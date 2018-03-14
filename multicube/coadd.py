#!/usr/bin/env python
#
# coadd - Crop, Scale, Rotate (90deg only), Align and Coadd individual CWI observations.
# 
# syntax: python coadd.py <parameterFile> <cubeType>
#

from astropy.io import fits as fitsIO

import numpy as np
import sys

import libs

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

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

####
#### INSERT: MASKING STAGE HERE
####
#### Temporary: Filter NaNs and INFs to at least avoid errors
for f in fits: f[0].data = np.nan_to_num(f[0].data)

#Check if parameters are complete
if libs.params.paramsMissing(params):

    #Enter set-up mode
    setupMode = True
    
    #Parse FITS headers for PA, instrument, etc.
    params = libs.params.parseHeaders(params,fits)

    #Get location of object in cube and correct WCS  
    fits = libs.cubes.fixWCS(fits,params)
    
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

#Crop to overlapping/good wavelength ranges
for i,f in enumerate(fits):
    
    #Crop FITS
    xcrop = tuple(int(x) for x in params["XCROP"][i].split(':'))
    ycrop = tuple(int(y) for y in params["YCROP"][i].split(':'))
    wcrop = tuple(int(w) for w in params["WCROP"][i].split(':'))

    f.crop(xx=xcrop,yy=ycrop,ww=wcrop)
    
    #Scale FITS to 1:1
    f.scale1to1()

    #Rotate FITS to PA=0 by rotating +(360-PA)
    Nrot = int( (params["PA"][i])/90) % 4
    f.rotate90( N=Nrot )      

#Align cubes to be stacked
fits = libs.cubes.wcsAlign(fits,params) 

for i,f in enumerate(fits): f.save("wcs%i.fits"%i)
#Stack cubes and trim
stacked,header = libs.cubes.coadd(fits,params)   

#Make FITS object for stacked cube
stackedFITS = fitsIO.HDUList([fitsIO.PrimaryHDU(stacked)])
stackedFITS[0].header = header

#Update target parameter file (if in setup mode)
if setupMode: libs.params.writeparams(params,parampath)

#Save stacked cube
stackedpath = '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubetype)
stackedFITS[0].writeto(stackedpath,overwrite=True)
print "Saved %s" % stackedpath
