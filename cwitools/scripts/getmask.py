"""Create a binary mask using a DS9 region file."""
from astropy.io import fits
from cwitools import coordinates, extraction, utils
from datetime import datetime

import argparse
import cwitools
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
                        metavar="<log_file>",
                        type=str,
                        help="Log file to save output in.",
                        default=None
    )
    parser.add_argument('-silent',
                        help="Set flag to suppress standard terminal output.",
                        action='store_true'
    )
    args = parser.parse_args()

    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_GETMASK:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

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
    utils.output("\tSaved %s\n" % outfilename)


if __name__=="__main__": main()
