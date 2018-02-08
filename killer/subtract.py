from astropy.io import fits as fitsIO

import tools
import numpy as np
import sys

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Check file extension is included in given cube type
if not ".fits" in cubetype: cubetype += ".fits"

#Check if any parameter values are missing (set to set-up mode if so)
params = tools.params.loadparams(parampath)

#Get filenames
print("Loading FITS files:")
files = tools.io.findfiles(params,cubetype)

#Handle missing files
for i,f in enumerate(files):
    if f!="": pass# print f
    else: print("File not found: ID:%s Type:%s" % (params["IMG_ID"][i],cubetype))

if any(np.array(files)==""):
    print("Some (or all) input files are missing. Please make sure files exist or comment out the relevant lines in %s with '#'" % parampath)
    sys.exit()
 

fits = [fitsIO.open(f) for f in files] #Open FITS files

#Filter NaNs and INFs to at least avoid errors (need a more robust way of handling Value Errors)
for f in fits: f[0].data = np.nan_to_num(f[0].data)

#Check if parameters are complete
if tools.params.paramsMissing(params):
    setupMode = True
    params = tools.params.parseHeaders(params,fits)
    #tools.params.writeparams(params,parampath)

#Check if parameters are complete
if tools.params.paramsMissing(params): setupMode = True
else: setupMode = False

for i,f in enumerate(fits):
    
    print "Subtracting continuum from %s" % files[i].split("/")[-1]
    
    #Get QSO position from param file
    qso_pos = (params["SRC_X"][i],params["SRC_Y"][i])
    xcrop,ycrop = params["XCROP"][i].split(':')
    xcrop = int(xcrop)
    ycrop = int(ycrop)

    #Check if location not measured yet
    if qso_pos==(-99,-99):    
        qfinder = tools.qso.qsoFinder(f,z=params["Z"],title=params["IMG_ID"][i]) #Get QSO Finder tool
        qso_pos = params["SRC_X"][i], params["SRC_Y"][i] = qfinder.run() #Run tool to get x,y pos of qso
        #tools.params.writeparams(params,parampath) #Update params file
    
    #Run subtraction  
    csub_data,cmodel = tools.qso.qsoSubtract(f,qso_pos,params["INST"][i],redshift=params["Z"],returnqso=True)
    
    print "test"
    
    csub_fits = fitsIO.HDUList([fitsIO.PrimaryHDU(csub_data)])
    csub_fits[0].header = f[0].header
    csub_path = files[i].replace('.fits','_csub.fits')
    csub_fits.writeto(csub_path,clobber=True)

    cfits = fitsIO.HDUList([fitsIO.PrimaryHDU(cmodel)])
    cfits[0].header = f[0].header
    cpath = files[i].replace('.fits','_cont.fits')
    cfits.writeto(cpath,clobber=True)


