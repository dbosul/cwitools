import libs
import numpy as np
import pyregion
import scipy as sc
import sys

import matplotlib.pyplot as plt

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Load pipeline parameters
params = libs.params.loadparams(parampath)
  
#Get filenames
files = libs.io.findfiles(params,cubetype)

#Open FITS files 
fits = [libs.fits3D.open(f) for f in files] 

#Open regionfile
regpath = params["REG_FILE"]
if regpath=='None': print "WARNING: No region file specified in %s. Sources will not be masked."
else: regfile = pyregion.open(regpath)

#Subtract continuum sources
for i,f in enumerate(fits):
    
    print "\nSubtracting continuum from %s" % files[i]
    
    #Filter NaNs and INFs from cube
    print "\tFiltering NaNs/INFs"
    fits[i][0].data = np.nan_to_num(f[0].data) 

    #Get for region file mask for this fits
    if regpath!="None": 
    
        #Get 2D Mask based on region file
        regmask = libs.cubes.get_mask(f,regfile)
        
        #Apply median mask to sources    
        cube_masked = libs.cubes.apply_mask(f[0].data.copy(),regmask,mode='xmedian',inst=params["INST"][i])

    #Just use unmasked cube if no region file provided
    else: cube_masked = f[0].data
    
    #Run cube-wide polyfit to subtract scattered light    
    wcrop = tuple(int(w) for w in params["WCROP"][i].split(':'))
    polyfit = libs.continuum.polyModel(cube_masked,w0=wcrop[0],w1=wcrop[1])
    
    #Subtract Polynomial continuum model from cube
    f[0].data -= polyfit
    
    #Save file
    savename = files[i].replace('.fits','_bs.fits')
    f.save(savename)
    print "Saved %s" % savename

