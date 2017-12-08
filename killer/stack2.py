from astropy.io import fits as fitsIO
import tools

import numpy as np
import matplotlib.pyplot as plt
import sys

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Check for variance cube input
vardata = True if 'vcube' in cubetype else False

#Set flag for whether params need to be updated
setupMode = False

#Add file extension of omitted
if not ".fits" in cubetype: cubetype += ".fits"

#Check if any parameter values are missing (set to set-up mode if so)
params = tools.params.loadparams(parampath)

#Get filenames     
files = tools.io.findfiles(params,cubetype)

#Open FITS files
fits = [fitsIO.open(f) for f in files] 

#Filter NaNs and INFs to at least avoid errors (need a more robust way of handling Value Errors)
for f in fits: f[0].data = np.nan_to_num(f[0].data)

#Check if parameters are complete
if tools.params.paramsMissing(params):
    setupMode = True
    
    #Parse FITS headers for PA, instrument, etc.
    params = tools.params.parseHeaders(params,fits)
    
    #Get location of object in cube and correct WCS  
    fits = tools.cubes.fixWCS(fits,params)

    #Write params to file
    tools.params.writeparams(params,parampath)

#Crop to overlapping/good wavelength ranges
fits = tools.cubes.crop(fits,params)

#Scale images to 1:1 aspect ratio
fits = tools.cubes.scale(fits,params,vardata)     

#Rotate images to same position Angle   
fits = tools.cubes.rotate(fits,params)          

#Align cubes to be stacked
fits = tools.cubes.align(fits,params) 

#Stack cubes and trim
stacked,header = tools.cubes.coadd(fits,params,vardata)   

#Make FITS object for stacked cube
stackedFITS = fitsIO.HDUList([fitsIO.PrimaryHDU(stacked)])
stackedFITS[0].header = header

#Fix WCS of cube using interactive mode
#stackedFITS[0].header =  tools.cubes.fixWCS(stackedFITS,params)

#Update target parameter file (if in setup mode)
if setupMode: tools.params.writeparams(params,parampath)

#Save stacked cube
stackedpath = '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubetype)
stackedFITS[0].writeto(stackedpath,clobber=True)
        
