"""Apply Mask: Apply a binary mask FITS image to data."""

from astropy.io import fits
import argparse
import sys

def run(maskPath,dataPath,fillValue=0):
    """
    Applies a binary mask to data.

    Args:
        maskPath (str): Path to the mask FITS file.
        dataPath (str): Path to the data FITS file.
        fileExt (str): Extension to use for masked file (Default:.M.fits)
        fillValue (float): Value to replace data with when masking (Default:0.0)

    """
    try: mask = fits.getdata(maskPath)
    except:
        print("Could not load mask. Check path and try again.\nPath:%s"%maskPath)
        sys.exit()

    try: data,header = fits.getdata(dataPath,header=True)
    except:
        print("Could not load data. Check path and try again.\nPath:%s"%dataPath)
        sys.exit()

    data_masked = data.copy()
    data_masked[ mask==1 ] = fillValue

    maskedFits = fits.HDUList([fits.PrimaryHDU([data_masked])])
    maskedFits[0].header = header.copy()

    return maskedFits

if __name__="__main__":

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

    maskedFits = run(args.mask,args.data,fillValue=args.fill,fileExt=args.ext)
    outFileName = args.data.replace('.fits',fileExt)
    maskedFits.writeto(outFileName,overwrite=True)

    print("Saved %s"%outFileName)
