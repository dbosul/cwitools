from astropy.io import fits
import argparse
import sys

parser = argparse.ArgumentParser(description="Apply a binary mask to data of the same dimensions.")
parser.add_argument('mask',
                    type=str,
                    help='Binary mask to be applied.'
)
parser.add_argument('data',
                    type=str,
                    help='Data to be masked.'
)
parser.add_argument('-fill',
                    type=float,
                    help='Value used to mask data (Default: 0)',
                    default=0
)
parser.add_argument('-ext',
                    type=str,
                    help="File extension to be used for masked data. Default: .M.fits",
                    default=".M.fits"
)
args = parser.parse_args()
try: mskFITS = fits.open(args.mask)
except:
    print("Could not load mask. Check path and try again.\nPath:%s"%args.mask)
    sys.exit()

try: inpFITS = fits.open(args.data)
except:
    print("Could not load data. Check path and try again.\nPath:%s"%args.data)
    sys.exit()

inpFITS[0].data *= (mskFITS[0].data==0)

outFileName = args.data.replace('.fits',args.ext)

inpFITS.writeto(outFileName,overwrite=True)

print("Saved %s"%outFileName)

