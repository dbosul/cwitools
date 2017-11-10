import astropy
import killer_quickTools
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

#Check file extension is included in given cube type
if not ".fits" in cubetype: cubetype += ".fits"

#Check if any parameter values are missing (set to set-up mode if so)
params = killer_quickTools.loadparams(parampath)

#Get filenames    
print("Locating and loading FITS files:")   
files = killer_quickTools.findfiles(params,cubetype)

#Handle missing files
for i,f in enumerate(files):
    if f!="": print f
    else: print("File not found: ID:%s Type:%s" % (params["IMG_ID"][i],cubetype))

if any(np.array(files)==""):
    print("Some (or all) input files are missing. Please make sure files exist or comment out the relevant lines in %s with '#'" % parampath)
    sys.exit()
 
fits = [astropy.io.fits.open(f) for f in files] 

#Filter NaNs and INFs to at least avoid errors (need a more robust way of handling Value Errors)
for f in fits: f[0].data = np.nan_to_num(f[0].data)

    
#Check if parameters are complete
if killer_quickTools.paramsMissing(params):
    setupMode = True
    params = killer_quickTools.parseHeaders(params,fits)
    killer_quickTools.writeparams(params,parampath)

#Scale images to 1:1 aspect ratio
fits = killer_quickTools.scale(fits,params,vardata)     

#Rotate images to same position Angle   
fits = killer_quickTools.rotate(fits,params)          

#Align cubes to be stacked
fits = killer_quickTools.align(fits,params) 

#Stack cubes and trim
stacked,header = killer_quickTools.coadd(fits,params,vardata)   

#Update target parameter file
if setupMode: killer_quickTools.writeparams(params,parampath)

#SAVE STACKED DATA
stackedpath = '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubetype)
killer_quickTools.saveFits(stacked,stackedpath,header)
        
