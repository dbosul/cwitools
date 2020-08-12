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
    parser.add_argument('-snr_int',
                        type=float,
                        help='Integrated SNR threshold. Takes priority over nmin if both provided.',
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
    parser.add_argument('-include',
                        type=str,
                        help="List of wavelength ranges to include following the format 4000:4200,4500:4600,5000:5100."
    )
    parser.add_argument('-exclude',
                        type=str,
                        help="List of wavelength ranges to exclude following the format 4000:4200,4500:4600,5000:5100."
    )
    parser.add_argument('-include_neb',
                        metavar='<redshift>',
                        type=float,
                        help='Prove redshift to auto-mask nebular emission.',
                        default=None
    )
    parser.add_argument('-neb_vwidth',
                        metavar='<km/s>',
                        type=float,
                        help='Velocity width (km/s) around nebular lines to mask, if using -mask_neb.',
                        default=2000
    )
    parser.add_argument('-exclude_sky',
                        action='store_true',
                        help='Automatically exclude bright sky lines from segmentation.'
    )
    parser.add_argument('-sky_width',
                        metavar='<Angstrom>',
                        type=float,
                        help='FWHM to use when excluding sky lines. Default is automatically determined based on instrument configuration.',
                        default=None
    )
    parser.add_argument('-fill_holes',
                        help='Set to TRUE to auto-repair 3D objects by filling holes. Uses scipy.ndimage.morphology.binary_fill_holes',
                        action='store_true'
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
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    fits_in = fits.open(args.cube)
    data, hdr = fits_in[0].data, fits_in[0].header

    var_cube = fits.getdata(args.var)

    #Try to parse the wavelength mask tuple
    custom_includes = []
    custom_excludes = []
    neb_includes = []
    sky_excludes = []

    if args.include is not None:
        try:
            for pair in args.include.split('-'):
                w0,w1 = tuple(int(x) for x in pair.split(':'))
                custom_includes.append([w0,w1])
        except:
            raise ValueError("Could not parse include argument (%s)." % args.include)

    if args.exclude is not None:
        try:
            for pair in args.exclude.split('-'):
                w0,w1 = tuple(int(x) for x in pair.split(':'))
                custom_excludes.append([w0,w1])
        except:
            raise ValueError("Could not parse exclude argument (%s)." % args.exclude)

    if args.include_neb is not None:
        neb_includes = utils.get_nebmask(fits_in[0].header,
            z = args.include_neb,
            vel_window = args.neb_vwidth,
            mode = 'tuples'
        )

    if args.exclude_sky is not None:
        sky_excludes = utils.get_skymask(fits_in[0].header,
            linewidth = args.sky_width,
            mode = 'tuples'
        )

    obj_fits = extraction.segment(fits_in, var_cube,
        snrmin = args.snrmin,
        nmin = args.nmin,
        includes = custom_includes + neb_includes,
        excludes = custom_excludes + sky_excludes,
        fill_holes = args.fill_holes,
        snr_int = args.snr_int
    )

    if args.out == None:
        outfilename = args.cube.replace(".fits", ".obj.fits")
    else:
        outfilename = args.out

    obj_fits.writeto(outfilename, overwrite=True)
    utils.output("\tSaved %s\n" % outfilename)


if __name__=="__main__": main()
