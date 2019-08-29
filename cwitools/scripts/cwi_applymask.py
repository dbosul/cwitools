"""Apply Mask: Apply a binary mask FITS image to data."""

from astropy.io import fits
import argparse
import sys

def main():
    
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

    try: mask = fits.getdata(args.mask)
    except:
        print("Could not load mask. Check path and try again.\nPath:%s"%args.mask)
        sys.exit()

    try: data,header = fits.getdata(args.data,header=True)
    except:
        print("Could not load data. Check path and try again.\nPath:%s"%args.data)
        sys.exit()

    data_masked = data.copy()
    data_masked[ mask==1 ] = args.fill

    outFileName = args.data.replace('.fits',args.ext)
    maskedFits = fits.HDUList([fits.PrimaryHDU([data_masked])])
    maskedFits[0].header = header.copy()
    maskedFits.writeto(outFileName,overwrite=True)

    print("Saved %s"%outFileName)


if __name__="__main__": main()
