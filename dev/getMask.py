from astropy.io import fits
import libs
import numpy as np
import os
import pyregion
import sys

S = 1.0

paramPath = sys.argv[1]
fitsPath = sys.argv[2]

fitsFile = fits.open(fitsPath)
params = libs.params.loadparams(paramPath)

regPath = params["REG_FILE"]

print("Opening region file.")
if not os.path.isfile(regPath):
    print("File not found: %s" % regPath)
    sys.exit()
else:
    regFile = pyregion.open(regPath)
    mskXY = libs.cubes.get_regMask(fitsFile,regFile,scaling=S)
    mskW  = libs.cubes.get_skyMask(fitsFile,params["INST"][0])
    
    msk3D = np.zeros_like(fitsFile[0].data)
    msk3D[mskW==1] = 1
    for L in range(msk3D.shape[0]): msk3D[L][mskXY>0] = 1
    
    fitsFile[0].data = msk3D
    fitsFile.writeto(fitsPath.replace('.fits','.MASK.fits'),overwrite=True)
