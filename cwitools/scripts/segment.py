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
    parser.add_argument('-var',
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
    parser.add_argument('-wrange',
                        type=str,
                        help="Wavelength range to consider, in Angstrom. E.g. 4100:4200"
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

    in_fits = fits.open(args.cube)
    data, hdr = in_fits[0].data, in_fits[0].header

    if args.var != None:
        var_cube = fits.getdata(args.var)
    else:
        var_cube = reduction.estimate_variance(in_fits)

    if args.wrange != None:
        w1, w2 = tuple(float(x) for x in args.wrange.split(":"))
        z1, z2 = coordinates.get_indices(w1, w2, hdr)
    else:
        z1, z2 = 0, data.shape[0]-1

    obj_mask = extraction.segment(data, var_cube,
        snrmin = args.snrmin,
        nmin = args.nmin,
        zrange = (z1, z2)
    )

    if args.out == None:
        outfilename = args.cube.replace(".fits", ".obj.fits")
    else:
        outfilename = args.out

    in_fits[0].data = obj_mask
    in_fits.writeto(outfilename, overwrite=True)
    utils.output("\tSaved %s\n" % outfilename)


if __name__=="__main__": main()
