from astropy.io import fits as fitsIO
from astropy.wcs import WCS
import numpy as np
import pyregion
import sys

import libs


parampath = sys.argv[1]
cubetype = sys.argv[2]

if len(sys.argv)>3: S = float(sys.argv[3])
else: S = 0.75

params = libs.params.loadparams(parampath)

print params

fitsPath = params["PRODUCT_DIR"] + params["NAME"] + "_" + cubetype

fits = fitsIO.open(fitsPath)
regfile = pyregion.open(params["REG_FILE"])

regMask = libs.cubes.get_regMask(fits,regfile,scaling=S)
skyMask = libs.cubes.get_skyMask(fits)


#Save 2D (Spatial/Source) Mask. Modify to a 2D header
regMaskPath = params["PRODUCT_DIR"] + params["NAME"] + ".MSK2D.fits"
head2D = fits[0].header.copy()
for key in ["NAXIS3","CRPIX3","CD3_3","CRVAL3","CTYPE3","CNAME3","CUNIT3"]: head2D.remove(key)
head2D["NAXIS"]=2
head2D["WCSDIM"]=2
hdulist = fitsIO.HDUList([fitsIO.PrimaryHDU(regMask)])
hdulist[0].header = fits[0].header
hdulist.writeto(regMaskPath,overwrite=True)
print "Wrote 2D (Source) Mask image to %s" % regMaskPath

#Save 1D (Wavelength/Sky) Mask. Modify to a 1D Header.
skyMaskPath = params["PRODUCT_DIR"] + params["NAME"] + ".MSK1D.fits"
hdulist = fitsIO.HDUList([fitsIO.PrimaryHDU(skyMask)])
hdulist[0].header["NAXIS"] = 1
hdulist[0].header["CD1_1"] = fits[0].header["CD3_3"]
for key in ["CRVAL","CRPIX","CTYPE","CUNIT","CNAME"]: hdulist[0].header["%s1"%key] = fits[0].header["%s3"%key]
hdulist.writeto(skyMaskPath,overwrite=True)
print "Wrote 1D (Sky) Mask image to %s" % skyMaskPath
