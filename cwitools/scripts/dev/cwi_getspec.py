"""Generate a pseudo-Narrowband image"""
from astropy.io import fits
from cwitools import utils, coordinates, synthesis
from datetime import datetime

import argparse
import cwitools
import numpy as np
import sys


import matplotlib.pyplot as plt

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'cube',
        type=str,
        help='The input data cube.'
    )
    parser.add_argument(
        'radius',
        type=float,
        help='The radius over which to integrate, in proper kiloparsec.'
    )
    parser.add_argument(
        'radec',
        type=float,
        nargs=2,
        help="RA and DEC in the format <dd.ddd> <dd.ddd>"
    )
    parser.add_argument(
        'redshift',
        type=float,
        help='The redshift of the source.'
    )
    parser.add_argument(
        '-var',
        type=str,
        help='The variance cube associated with the input data.'
    )
    parser.add_argument(
        '-ext',
        type=str,
        help='The extension for the output FITS Table (Default: .rspec.fits)',
        default=".rspec.fits"
    )
    parser.add_argument(
        '-wmask',
        metavar='<w0:w1,w2:w3,...>',
        type=str,
        help='Wavelength range(s) to mask when fitting',
        default=None
    )
    parser.add_argument(
        '-log',
        metavar="<log_file>",
        type=str,
        help="Log file to save output in.",
        default=None
    )
    parser.add_argument(
        '-silent',
        help="Set flag to suppress standard terminal output.",
        action='store_true'
    )
    args = parser.parse_args()

    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_GETSPC:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    masks = []
    if args.wmask is not None:
        try:
            for pair in args.wmask.split('-'):
                masks.append(tuple(int(x) for x in pair.split(':')))
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)
            
    fits_in = fits.open(args.cube)
    ra, dec = args.radec
    z = args.redshift
    radius = args.radius
    vardata = None if args.var is None else fits.getdata(args.var)

    specR = synthesis.sum_spec_r(fits_in, ra, dec, z, radius,
        var_cube = vardata,
        wmask = masks
    )

    outfile = args.cube.replace(".fits", args.ext)
    specR.writeto(outfile, overwrite=True)
    utils.output("\tSaved %s\n" % outfile)



if __name__=="__main__": main()
