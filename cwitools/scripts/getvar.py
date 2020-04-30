"""Estimate the 3D variance for a data cube"""
from astropy.io import fits
from cwitools.reduction import estimate_variance
from cwitools import utils
from datetime import datetime

import argparse
import cwitools
import os
import sys

def main():
    #Take any additional input params, if provided
    parser = argparse.ArgumentParser(description='Get estimated variance cube.')
    parser.add_argument('cube',
                        type=str,
                        metavar='path',
                        help='Input cube whose 3D variance you would like to estimate.'
    )
    parser.add_argument('-window',
                        type=int,
                        help='Size of wavelength bin, in Angstrom, for 2D layer variance estimate.',
                        default=50
    )
    parser.add_argument('-wmask',
                        type=str,
                        metavar='Wav Mask',
                        help='Wavelength range(s) to mask when fitting',
                        default=None
    )
    parser.add_argument('-sclip',
                        type=float,
                        help='Sigma-clip to apply calculating rescaling factors',
                        default=4
    )
    parser.add_argument('-fmin',
                        type=float,
                        metavar='float',
                        help='Minimum rescaling factor (default 0.9)',
                        default=0.9
    )
    parser.add_argument('-out',
                        type=str,
                        metavar='str',
                        help='Filename for output. Default is input + .var.fits',
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
    titlestring = """\n{0}\n{1}\n\tCWI_GETVAR:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(parser)
    utils.output(titlestring + infostring)

    #Try to load the fits file
    if os.path.isfile(args.cube): fits_in = fits.open(args.cube)
    else:
        raise FileNotFoundError("Input file not found.")

    #Try to parse the wavelength mask tuple
    wmasks = []
    if args.wmask != None:
        try:
            for pair in args.wmask.split('-'):
                w0,w1 = tuple(int(x) for x in pair.split(':'))
                wmasks.append([w0,w1])
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)

    vardata = estimate_variance(fits_in,
    window=args.window,
    wmasks=wmasks,
    fmin=args.fmin,
    sclip=args.sclip
    )

    if args.out == None:
        outfilename = args.cube.replace('.fits', '.var.fits')
    else:
        outfilename = args.out

    var_fits = fits.HDUList([fits.PrimaryHDU(vardata)])
    var_fits[0].header = fits_in[0].header
    var_fits.writeto(outfilename,overwrite=True)
    utils.output("\tSaved %s\n" % outfilename)

if __name__=="__main__": main()
