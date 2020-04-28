"""Create a binary mask using a DS9 region file."""
from astropy.io import fits
from cwitools import coordinates, extraction, utils

import argparse
import numpy as np
import os
import sys
import warnings

def main():

    parser = argparse.ArgumentParser(description="Apply a binary mask to data of the same dimensions.")
    parser.add_argument('reg',
                        type=str,
                        help='DS9 region file to convert into a mask.'
    )
    parser.add_argument('data',
                        type=str,
                        help='Data cube or image to create mask for.'
    )
    parser.add_argument('-out',
                        type=str,
                        help='Output filename. By default will be input data + .mask.fits',
                        default=None
    )
    parser.add_argument('-log',
                        type=str,
                        help="Log file to save this command in",
                        default=None
    )
    args = parser.parse_args()

    if os.path.isfile(args.data):
        fits_in = fits.open(args.data)
    else:
        raise FileNotFoundError(args.data)

    #Get mask
    mask = extraction.reg2mask(fits_in, args.reg)

    if args.out == None:
        outfilename = args.data.replace('.fits', '.mask.fits')
    else:
        outfilename = args.out

    mask.writeto(outfilename,overwrite=True)
    print("Saved %s" % outfilename)


if __name__=="__main__": main()
