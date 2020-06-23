
from astropy.io import fits
from cwitools import extraction, utils, coordinates
from datetime import datetime


import argparse
import cwitools
import numpy as np
import os
import sys
import warnings

def main():
    """Apply Mask: Apply a binary mask FITS image to data."""


    parser = argparse.ArgumentParser(description="Apply a mask to data.")
    parser.add_argument(
        'data',
        type=str,
        help='Data to be masked.'
    )
    parser.add_argument(
        '-masks',
        type=str,
        help="The mask value to isolate. E.g. '0' masks all pixels where mask is non-zero. '3' masks all pixels where mask is NOT 3."
    )
    parser.add_argument(
        '-mask_sky',
        help="Set flag to auto-mask some known bright sky lines.",
        action='store_true'
    )
    parser.add_argument('-out',
        type=str,
        help="Output file name. Default is to add .zmask.fits ",
        default=None
    )
    parser.add_argument(
        '-log',
        type=str,
        metavar="<log_file>",
        help="Log file to save output in."
    )
    parser.add_argument(
        '-silent',
        help="Set flag to suppress standard terminal output.",
        action='store_true'
    )
    args = parser.parse_args()

    if os.path.isfile(args.data):
        fits_in = fits.open(args.data)
        data, header = fits_in[0].data, fits_in[0].header
    else:
        raise FileNotFoundError(args.data)

    log = args.log
    silent = args.silent
    cmd = utils.get_cmd(sys.argv)

    #Set global parameters
    cwitools.silent_mode = silent
    cwitools.log_file = log

    #Give output summarizing mode
    titlestring = """\n{0}\n{1}\n\tCWI_APPLYMASK:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(parser)
    utils.output(titlestring + infostring)

    if not(args.mask_sky) and (args.masks is None):
        raise SyntaxError("Must provide mask_sky and/or masks argument.")

    if args.mask_sky:
        sky_mask = utils.get_skymask(header)
        data[sky_mask] = 0

    if args.masks is not None:
        wav_axis = coordinates.get_wav_axis(header)
        zmask = np.zeros_like(wav_axis, dtype=bool)
        for tup in args.masks.split('-'):
            w0, w1 = tuple(int(x) for x in tup.split(":"))
            zmask[(wav_axis >= w0) & (wav_axis <= w1)] = 1
        data[zmask] = 0


    if args.out is None:
        outfilename = args.data.replace('.fits', '.zmask.fits')
    else:
        outfilename = args.out

    maskedFits = utils.matchHDUType(fits_in, data, header)
    maskedFits.writeto(outfilename,overwrite=True)

    utils.output("\tSaved %s\n"%outfilename)


if __name__=="__main__": main()
