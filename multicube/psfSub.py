from astropy.io import fits as fitsIO
from scipy.ndimage.measurements import center_of_mass

import libs
import numpy as np
import pyregion
import sys

import matplotlib.pyplot as plt

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]
regpath = sys.argv[3]

#Load pipeline parameters
params = libs.params.loadparams(parampath)
  
#Get filenames
files = libs.io.findfiles(params,cubetype)

#Open FITS files 
print("Loading FITS files:")
fits = [libs.fits3D.open(f) for f in files] 

#Open regionfile
regfile = pyregion.open(regpath)

#Subtract continuum sources
for i,f in enumerate(fits):
    
    #Filter NaNs and INFs to at least avoid errors 
    fits[i][0].data = np.nan_to_num(f[0].data)
    
    print "\nSubtracting continuum sources from %s" % files[i].split("/")[-1]

    #Get for region file mask for this fits
    regmask = libs.cubes.get_mask(f,regfile)   
    
    model = np.zeros_like(f[0].data)
    
    #Run through values in mask
    for j,m in enumerate(np.unique(regmask)):

        if m==0: continue #Ignore 0
        
        print "Source %i/%i " % (j,len(np.unique(regmask))-1),
               
        mask2 = regmask.copy() #Make copy of mask

        mask2[regmask!=m] = 0 #Filter out other mask values
                
        y,x = center_of_mass(mask2) #Get center of mass of this target
        
        x,y = int(round(x)),int(round(y)) #Round to nearest int

        csub_data,cmodel = libs.continuum.psfSubtract(f,(x,y),redshift=params["Z"],mode='specFit') #Run subtract code
        
        f[0].data = csub_data #Subtract from data
        
        model += cmodel #Add to model

        print ""
    csub_path = files[i].replace('.fits','_csub.fits')
    f.save(csub_path)
    
    cont_path = files[i].replace('.fits','_cont.fits')
    f[0].data -= model
    f.save(cont_path)
  
