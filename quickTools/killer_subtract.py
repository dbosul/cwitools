from astropy.io import fits as fitsIO

import killer_quickTools
import sys

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Check file extension is included in given cube type
if not ".fits" in cubetype: cubetype2 = cubetype+".fits"

#Check if any parameter values are missing (set to set-up mode if so)
params = killer_quickTools.loadparams(parampath)

#Load data
files = killer_quickTools.findfiles(params,cubetype)

print("Loading FITS files:")
for f in files: print f

fits = [fitsIO.open(f) for f in files] #Open FITS files

#Check if parameters are complete
if killer_quickTools.paramsMissing(params): setupMode = True
else: setupMode = False

for i,f in enumerate(fits):
    
    print "Subtracting QSO from %s" % files[i].split("/")[-1]
    
    #Get QSO position from param file
    qso_pos = (params["QSO_X"][i],params["QSO_Y"][i])

    #Check if location not measured yet
    if qso_pos==(-99,-99):    
        qfinder = killer_quickTools.qsoFinder(f) #Get QSO Finder tool
        qso_pos = params["QSO_X"][i], params["QSO_Y"][i] = qfinder.run() #Run tool to get x,y pos of qso
        killer_quickTools.writeparams(params,parampath) #Update params file
    
    #Run subtraction  
    qsub_fits = killer_quickTools.qsoSubtract(f,qso_pos,params["INST"][i],redshift=params["Z"])
    
    #Save subtracted cube
    qsub_path = files[i].replace('.fits','_qsub.fits')
    killer_quickTools.saveFits(qsub_fits,qsub_path,f[0].header)
    

