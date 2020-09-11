
from astropy.io import fits
from cwitools import extraction, utils
from datetime import datetime

import argparse
import cwitools
import os
import sys
import warnings

def main(mask=None, data=None, fill=0, ext=".M.fits", log=None, silent=False,
label=0):
    """Apply Mask: Apply a binary mask FITS image to data."""

    #If required arguments are not present, use argparse
    if mask is None and data is None:
        parser = argparse.ArgumentParser(description="Apply a mask to data.")
        parser.add_argument('mask',
                            type=str,
                            help='Binary mask to be applied.'
        )
        parser.add_argument('data',
                            type=str,
                            help='Data to be masked.'
        )
        parser.add_argument('-label',
                            type=float,
                            help="The mask value to isolate. E.g. '0' masks all\
pixels where mask is non-zero. '3' masks all pixels where mask is NOT 3.",
                            default=label
        )
        parser.add_argument('-fill',
                            type=float,
                            help='Value used to mask data (Default: 0)',
                            default=fill
        )
        parser.add_argument('-ext',
                            type=str,
                            help="Output file extension. Default is .M.fits ",
                            default=ext
        )
        parser.add_argument('-log',
                            type=str,
                            metavar="<log_file>",
                            help="Log file to save output in.",
        )
        parser.add_argument('-silent',
                            help="Set flag to suppress standard terminal output.",
                            action='store_true'
        )
        args = parser.parse_args()

        #Extract argparse argument
        if os.path.isfile(args.mask):
            mask = fits.getdata(args.mask)
        else:
            raise FileNotFoundError(args.mask)

        if os.path.isfile(args.data):
            fits_in = fits.open(args.data)
            data, header = fits_in[0].data, fits_in[0].header
        else:
            raise FileNotFoundError(args.data)

        fill = args.fill
        ext = args.ext
        log = args.log
        silent = args.silent

        cmd = utils.get_cmd(sys.argv)

    elif mask is None or data is None:
        raise ValueError("Both data and mask must be provided.")

    else:
        cmd = "" #No command line given, so put blank placeholder here.


    #Set global parameters
    cwitools.silent_mode = silent
    cwitools.log_file = log

    #Give output summarizing mode
    titlestring = """\n{0}\n{1}\n\tCWI_APPLYMASK:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    masked_data = extraction.apply_mask(data, mask)

    if ext == None:
        outfilename = args.data.replace('.fits', '.M.fits')
    else:
        outfilename = args.data.replace('.fits', ext)

    maskedFits = utils.matchHDUType(fits_in, masked_data, header)
    maskedFits.writeto(outfilename,overwrite=True)

    utils.output("\tSaved %s\n"%outfilename)


if __name__ == "__main__": main(TBD, arg_parser=parser_init())
