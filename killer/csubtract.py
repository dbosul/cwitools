from astropy.io import fits as fitsIO

import fits3D
import tools
import numpy as np
import pyregion
import sys

import matplotlib.pyplot as plt

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Check file extension is included in given cube type
if not ".fits" in cubetype: cubetype += ".fits"

#Load pipeline parameters
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
    
#Open FITS files 
fits = [fits3D.open(f) for f in files] 

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
    
    print "\nSubtracting continuum sources from %s" % files[i].split("/")[-1]
    
    x0,x1 = params["XCROP"][i].split(':')
    x0 = int(x0)
    x1 = int(x1)
    
    y0,y1 = params["YCROP"][i].split(':')
    y0 = int(y0)
    y1 = int(y1)    

    #f.crop(xx=(x0,x1),yy=(y0,y1)) #Avoid bad regions of image for subtraction

    model = np.zeros_like(f[0].data)
    
    keywords3D = ["NAXIS3","CRPIX3","CD3_3","CRVAL3","CTYPE3","CNAME3","CUNIT3"]
    header2D = f[0].header.copy()    
    for key,val in header2D.items():
        if key in keywords3D:
            header2D.remove(key)
    header2D["NAXIS"]=2
    header2D["WCSDIM"]=2
    
        
    while True:

        qfinder = tools.qso.qsoFinder(f,z=params["Z"],title="Locate source to subtract.") #Get QSO Finder tool
        x,y = qfinder.run() #Run tool to get x,y pos of qso

        if x==-99 or y==-99: break
                
        x = int(x)
        y = int(y)

        csub_data,cmodel = tools.qso.qsoSubtract(f,(x,y),params["INST"][i],redshift=params["Z"],returnqso=True)
        
        f[0].data = csub_data
        model += cmodel
        

    csub_path = files[i].replace('.fits','_csub.fits')
    f.save(csub_path)
    
    #cont_path = files[i].replace('.fits','_cont.fits')
    #.save(cont_path)
  
