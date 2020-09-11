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
        'var',
        type=str,
        help='The input variance cube.'
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
    config.silent_mode = args.silent
    config.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_OBJSB:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    fits_in = fits.open(args.cube)
    var_in = fits.open(args.var)[0].data
    int_cube, hdr3d = fits_in[0].data, fits_in[0].header
    obj_cube = fits.getdata(args.obj)


    for line in open(args.det):

        if line[0] == '#': continue

        cols = line.split(",")
        if len(cols) < 2: continue
        
        name = cols[0].replace(' ','').replace(']', 'f').replace('[','')
        obj_ids = [int(x) for x in cols[1:]]

        sb_map, sb_err = synthesis.obj_sb(fits_in, obj_cube, obj_ids, var_cube=var_in)
        m1, m1_err, m2, m2_err = synthesis.obj_moments(fits_in, obj_cube, obj_ids, var_cube=var_in)
        spec = synthesis.obj_spec(fits_in, obj_cube, obj_ids, var_cube=var_in)

        sb_out = args.cube.replace(".fits", ".{0}.sb.fits".format(name))
        sb_map.writeto(sb_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(sb_out))

        sb_err[0].data = np.sqrt(sb_err[0].data)
        sb_err_out = sb_out.replace('.fits', '.err.fits')
        sb_err.writeto(sb_err_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(sb_err_out))

        m1_out = args.cube.replace(".fits", ".{0}.m1.fits".format(name))
        m1.writeto(m1_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(m1_out))

        m1_err_out = m1_out.replace('.fits', '.err.fits')
        m1_err.writeto(m1_err_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(m1_err_out))

        m2_out = args.cube.replace(".fits", ".{0}.m2.fits".format(name))
        m2.writeto(m2_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(m2_out))

        m2_err_out = m2_out.replace('.fits', '.err.fits')
        m2_err.writeto(m2_err_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(m2_err_out))

        spec_out = args.cube.replace(".fits", ".{0}.spec.fits".format(name))
        spec.writeto(spec_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(spec_out))


if __name__ == "__main__": main(TBD, arg_parser=parser_init())
