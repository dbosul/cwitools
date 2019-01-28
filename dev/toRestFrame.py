from astropy.io import fits as fitsIO
from astropy.modeling import models,fitting
import numpy as np
import sys
import time

import libs

#Timer start
tStart = time.time()

#Take minimum input 
paramPath = sys.argv[1]
cubePath  = sys.argv[2]
            
#Load parameters
params = libs.params.loadparams(paramPath)

#Check if parameters are complete
libs.params.verify(params)

redshift = params["Z"]

f = fitsIO.open(cubePath)

f[0].header["CRVAL3"] /= (1+redshift)
f[0].header["CD3_3"] /= (1+redshift)

f.writeto(cubePath.replace('.fits','.R.fits'),overwrite=True)

