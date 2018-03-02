from astropy.io import fits as fitsIO
from scipy.ndimage.measurements import center_of_mass

import fits3D
import tools
import numpy as np
import pyregion
import sys

import matplotlib.pyplot as plt

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]
regpath = sys.argv[3]

#Load pipeline parameters
params = tools.params.loadparams(parampath)
  
#Get filenames
files = tools.io.findfiles(params,cubetype)

#Open FITS files 
print("Loading FITS files:")
fits = [fits3D.open(f) for f in files] 

#Open regionfile
regfile = pyregion.open(regpath)


#Subtract continuum sources
for i,f in enumerate(fits):
    
    fits[i][0].data = np.nan_to_num(f[0].data) #Filter NaNs and INFs to at least avoid errors (need a more robust way of handling Value Errors)
    
    print "\nSubtracting continuum sources from %s" % files[i].split("/")[-1]
    
    for yi in range(f[0].data.shape[1]):
        f[0].data -= np.median(f[0].data[:,yi,:]) #TEMPORARY - MEDIAN SUBTRACT CUBE
    
    #Get for region file mask for this fits
    regmask = tools.cubes.get_mask(f,regfile)   

    model = np.zeros_like(f[0].data)

    #bgsub_data,bmodel = tools.continuum.bgSubtract(f,regmask,redshift=params["Z"])    
    #f[0].data = bmodel
    #model += bmodel   
    #f.save("medtest.fits")
    
    #Run through values in mask
    for j,m in enumerate(np.unique(regmask)):

        if m==0: continue #Ignore 0
        
        print "Source %i/%i " % (j,len(np.unique(regmask))-1),
               
        mask2 = regmask.copy() #Make copy of mask

        mask2[regmask!=m] = 0 #Filter out other mask values
                
        y,x = center_of_mass(mask2) #Get center of mass of this target
        
        x,y = int(round(x)),int(round(y)) #Round to nearest int

        csub_data,cmodel = tools.continuum.cSubtract(f,(x,y),redshift=params["Z"],mode='specFit') #Run subtract code
        
        f[0].data = csub_data #Subtract from data
        
        model += cmodel #Add to model

        print ""
    csub_path = files[i].replace('.fits','_csub.fits')
    f.save(csub_path)
    
    cont_path = files[i].replace('.fits','_cont.fits')
    f[0].data = model
    f.save(cont_path)
  
