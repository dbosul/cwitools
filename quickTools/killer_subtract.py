from astropy.io import fits as fitsIO

import killer_quickTools
import numpy as np
import sys

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Check file extension is included in given cube type
if not ".fits" in cubetype: cubetype += ".fits"

#Check if any parameter values are missing (set to set-up mode if so)
params = killer_quickTools.loadparams(parampath)

#Get filenames
print("Loading FITS files:")
files = killer_quickTools.findfiles(params,cubetype)

#Handle missing files
for i,f in enumerate(files):
    if f!="": print f
    else: print("File not found: ID:%s Type:%s" % (params["IMG_ID"][i],cubetype))

if any(np.array(files)==""):
    print("Some (or all) input files are missing. Please make sure files exist or comment out the relevant lines in %s with '#'" % parampath)
    sys.exit()
 
fits = [astropy.io.fits.open(f) for f in files] 
fits = [fitsIO.open(f) for f in files] #Open FITS files

#Filter NaNs and INFs to at least avoid errors (need a more robust way of handling Value Errors)
for f in fits: f[0].data = np.nan_to_num(f[0].data)

#Check if parameters are complete
if killer_quickTools.paramsMissing(params):
    setupMode = True
    params = killer_quickTools.parseHeaders(params,fits)
    killer_quickTools.writeparams(params,parampath)


#Check if parameters are complete
if killer_quickTools.paramsMissing(params): setupMode = True
else: setupMode = False

for i,f in enumerate(fits):
    
    print "Subtracting QSO from %s" % files[i].split("/")[-1]
    
    #Get QSO position from param file
    qso_pos = (params["QSO_X"][i],params["QSO_Y"][i])

    #Check if location not measured yet
    if qso_pos==(-99,-99):    
        qfinder = killer_quickTools.qsoFinder(f,z=params["Z"],title=params["IMG_ID"][i]) #Get QSO Finder tool
        qso_pos = params["QSO_X"][i], params["QSO_Y"][i] = qfinder.run() #Run tool to get x,y pos of qso
        killer_quickTools.writeparams(params,parampath) #Update params file
    
    #Run subtraction  
    qsub_fits = killer_quickTools.qsoSubtract(f,qso_pos,params["INST"][i],redshift=params["Z"])
    
    #Save subtracted cube
    qsub_path = files[i].replace('.fits','_qsub.fits')
    killer_quickTools.saveFits(qsub_fits,qsub_path,f[0].header)
    

