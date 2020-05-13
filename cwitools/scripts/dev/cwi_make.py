"""Generate a pseudo-Narrowband image"""
from astropy.io import fits
from cwitools import utils, coordinates, synthesis
from datetime import datetime

import argparse
import cwitools
import numpy as np
import sys

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'cube',
        type=str,
        help='The input data cube.'
    )
    parser.add_argument(
        'obj',
        type=str,
        help='The input object cube.'
    )
    parser.add_argument(
        'det',
        type=str,
        help='The input detections table.'
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
    titlestring = """\n{0}\n{1}\n\tCWI_OBJSB:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(parser)
    utils.output(titlestring + infostring)

    fits_in = fits.open(args.cube)
    int_cube, hdr3d = fits_in[0].data, fits_in[0].header
    obj_cube = fits.getdata(args.obj)


    for line in open(args.det):

        if line[0] == '#': continue

        cols = line.split(",")
        name = cols[0].replace(' ','')
        obj_ids = [int(x) for x in cols[1:]]

        sb_map = synthesis.obj_sb(fits_in, obj_cube, obj_ids)
        m1, m1err, m2, m2err = synthesis.obj_moments(fits_in, obj_cube, obj_ids)

        sb_out = args.cube.replace(".fits", ".{0}.sb.fits".format(name))
        sb_map.writeto(sb_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(sb_out))
        
        m1_out = args.cube.replace(".fits", ".{0}.m1.fits".format(name))
        m1.writeto(m1_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(m1_out))

        m2_out = args.cube.replace(".fits", ".{0}.m2.fits".format(name))
        m2.writeto(m2_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(m2_out))

if __name__=="__main__": main()
