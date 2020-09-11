"""CWITools Apply Mask: Apply a binary mask FITS image to data."""

#Standard Imports
import argparse
import os

#Third-party Imports
from astropy.io import fits
from cwitools import extraction, utils

#Local Imports
import cwitools


def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(description="Apply a mask to data.")
    parser.add_argument(
        'mask',
        type=str,
        help='Binary mask to be applied.'
        )
    parser.add_argument(
        'data',
        type=str,
        help='Data to be masked.'
        )
    parser.add_argument(
        '-fill',
        type=float,
        help='Value used to mask data (Default: 0)',
        default=0
        )
    parser.add_argument(
        '-ext',
        type=str,
        help="Output file extension. Default is .M.fits ",
        default=".M.fits"
        )
    parser.add_argument(
        '-log',
        type=str,
        metavar="<log_file>",
        help="Log file to save output in.",
        )
    parser.add_argument(
        '-silent',
        help="Set flag to suppress standard terminal output.",
        action='store_true'
        )
    return parser

def main(mask, data, fill=0, ext=".M.fits", log=None, silent=False):
    """Apply Mask: Apply a binary mask FITS image to data."""

    #Set global parameters
    cwitools.silent_mode = silent
    cwitools.log_file = log

    utils.output_func_summary("APPLY_MASK", locals())

    #Extract argparse argument
    if os.path.isfile(mask):
        mask_fits = fits.open(mask)
    else:
        raise FileNotFoundError(mask)

    if os.path.isfile(data):
        data_fits = fits.open(data)
    else:
        raise FileNotFoundError(data)

    masked_data = extraction.apply_mask(
        data_fits[0].data,
        mask_fits[0].data,
        fill=fill
        )

    if ext is None:
        outfilename = data.replace('.fits', '.M.fits')
    else:
        outfilename = data.replace('.fits', ext)

    out_fits = utils.matchHDUType(data_fits, masked_data, data_fits[0].header)
    out_fits.writeto(outfilename, overwrite=True)

    utils.output("\tSaved %s\n" % outfilename)


#Call using dict and argument parser if run from command-line
if __name__ == "__main__":

    arg_parser = parser_init()
    args = arg_parser.parse_args()

    main(**vars(args))
