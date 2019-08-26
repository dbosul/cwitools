from astropy.io import fits
import sys

mskFITS = fits.open(sys.argv[1])
inpFITS = fits.open(sys.argv[2])

inpFITS[0].data *= (mskFITS[0].data==0)
inpFITS.writeto(sys.argv[2].replace('.fits','.M.fits'),overwrite=True)
