from astropy.io import fits
from cwitools import coordinates, utils, extraction, reduction
from datetime import datetime
from skimage import measure

import argparse
import cwitools
import numpy as np
import sys

def main():
    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Segment cube into 3D regions above a certain SNR.')
    parser.add_argument('cube',
                        type=str,
                        help='The input data cube.'
    )
    parser.add_argument('var',
                        type=str,
                        help='Variance cube. Estimated if not provided.',
                        default=None
    )
    parser.add_argument('-snrmin',
                        type=float,
                        help='The SNR threshold to use.',
                        default=3.0
    )
    parser.add_argument('-nmin',
                        type=int,
                        help='Minimum region size, in voxels.',
                        default=10
    )
    parser.add_argument('-wmask',
                        type=str,
                        help="List of wavelength ranges to include."
    )
    parser.add_argument('-out',
                        type=str,
                        help="Output filename. Default, input cube with .obj.fits",
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
    titlestring = """\n{0}\n{1}\n\tCWI_SEGMENT:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(parser)
    utils.output(titlestring + infostring)

    fits_in = fits.open(args.cube)
    data, hdr = fits_in[0].data, fits_in[0].header

    var_cube = fits.getdata(args.var)

    #Try to parse the wavelength mask tuple
    wranges = []
    if args.wmask != None:
        try:
            for pair in args.wmask.split('-'):
                w0,w1 = tuple(int(x) for x in pair.split(':'))
                wranges.append([w0,w1])
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)


    print(wranges)
    obj_fits = extraction.segment(fits_in, var_cube,
        snrmin = args.snrmin,
        nmin = args.nmin,
        wranges = wranges
    )

    if args.out == None:
        outfilename = args.cube.replace(".fits", ".obj.fits")
    else:
        outfilename = args.out

    obj_fits.writeto(outfilename, overwrite=True)
    utils.output("\tSaved %s\n" % outfilename)


if __name__=="__main__": main()
