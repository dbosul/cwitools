"""Create 2D maps of velocity and dispersion."""
from astropy.io import fits
from cwitools import extraction, reduction, utils, synthesis
from datetime import datetime

import argparse
import cwitools
import numpy as np
import os
import sys

def main():
    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Make maps of the first and second velocity moments of a 3D object.')
    parser.add_argument('cube',
                        type=str,
                        metavar='cube',
                        help='The input data cube.'
    )
    parser.add_argument('obj',
                        type=str,
                        metavar='path',
                        help='Object Mask cube.',
    )
    parser.add_argument('doublet',
                        type=str,
                        help="Tuple representing the lines of the doublet, e.g. 1548,1550 ."
    )
    parser.add_argument('-id',
                        type=str,
                        metavar='str',
                        help='The ID of the object to use. Use -1 for all objects. Can also provide multiple as comma-separated list.',
                        default='1'
    )
    parser.add_argument('-rsmooth',
                        type=float,
                        help='Smooth spatial axes before calculating moments (FWHM).',
                        default=None
    )
    parser.add_argument('-wsmooth',
                        type=float,
                        help='Smooth wavelength axis before calculating moments (FWHM).',
                        default=None
    )
    parser.add_argument('-z',
                        type=float,
                        help='Redshift of the source, assuming doublet is given in rest-frame units.',
                        default=0.0
    )
    parser.add_argument('-vmax',
                        type=float,
                        help='Maximum velocity offset to allow in model.',
                        default=2000
    )
    parser.add_argument('-disp_min',
                        type=float,
                        help='Minimum dispersion to allow in model.',
                        default=50
    )
    parser.add_argument('-disp_max',
                        type=float,
                        help='Maximum dispersion to allow in model.',
                        default=500
    )
    parser.add_argument('-ratio_min',
                        type=float,
                        help='Minimum blue-peak/red-peak ratio.',
                        default=0.5
    )
    parser.add_argument('-label',
                        type=str,
                        help='Label for output (e.g. NV, CIV_1548)',
                        default=""
    )
    parser.add_argument('-ratio_max',
                        type=float,
                        help='Maximum blue-peak/red-peak ratio.',
                        default=2.0
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
    config.silent_mode = args.silent
    config.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_FIT_DOUBLET:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    #Try to load the fits file
    if os.path.isfile(args.cube):
        fits_in = fits.open(args.cube)
    else:
        raise FileNotFoundError(args.cube)

    if args.rsmooth!=None:
        fits_in[0].data = extraction.smooth_cube_spatial(
            fits_in[0].data,
            args.rsmooth
        )

    if args.wsmooth!=None:
        fits_in[0].data = extraction.smooth_cube_wavelength(
            fits_in[0].data,
            args.wsmooth
        )

    #Load object mask
    if os.path.isfile(args.obj):
        obj_cube = fits.getdata(args.obj)
    else:
        raise FileNotFoundError(args.obj)

    #Parse object IDs
    try:
        obj_id = list( int(x) for x in args.id.split(',') )
    except:
        raise ValueError("Could not parse -objid flag. Should be comma-separated list of object IDs.")


    try:
        peak1, peak2 = tuple(float(x) for x in args.doublet.split(','))
    except:
        raise ValueError("Could not parse doublet argument. Should be tuple of floats.")

    m1_fits, m2_fits = synthesis.obj_moments_doublet(
        fits_in,
        obj_cube,
        obj_id,
        peak1,
        peak2,
        z = args.z,
        disp_min = args.disp_min,
        disp_max = args.disp_max,
        v_max = args.vmax,
        ratio_min = args.ratio_min,
        ratio_max = args.ratio_max
    )
    if args.label == "":
        args.label = "%i" % peak1
    m1_out = args.cube.replace('.fits', '.%s.m1_doublet.fits' % args.label)
    m1_fits.writeto(m1_out,overwrite=True)
    utils.output("\tSaved %s\n" % m1_out)

    m2_out = args.cube.replace('.fits', '.%s.m2_doublet.fits' % args.label)
    m2_fits[0].header["BUNIT"] = "km/s"
    m2_fits.writeto(m2_out, overwrite=True)
    utils.output("\tSaved %s\n" % m2_out)



if __name__=="__main__":
    main()
