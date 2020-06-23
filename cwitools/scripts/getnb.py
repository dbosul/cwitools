"""Generate a pseudo-Narrowband image"""
from astropy.io import fits
from cwitools import extraction, reduction, utils, synthesis
from datetime import datetime

import argparse
import cwitools
import numpy as np
import sys
import time

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Make channel maps of an input cube around a specified emission line.')
    parser.add_argument(
        'cube',
        type=str,
        metavar='cube',
        help='The input data cube.'
    )
    parser.add_argument(
        'center',
        type=float,
        metavar='float',
        help='Central wavelength to use for pseudo NB. Default: None.',
        default=None
    )
    parser.add_argument(
        'width',
        type=float,
        metavar='float',
        help='Pseudo-NB width in Angstrom.',
        default=None
    )
    parser.add_argument(
        '-var',
        type=str,
        metavar='var',
        help='Variance cube for SNR calculations. Estimated if not given.',
        default=None
    )
    parser.add_argument(
        '-pos',
        type=str,
        metavar='str',
        help='Position of your source as an \'x,y\' tuple.  Default: None',
        default=None

    )
    parser.add_argument(
        '-fit_rad',
        type=float,
        metavar='float',
        help='Radius (arcsec) to use for scaling PSF image. Default: 3px',
        default=1
    )
    parser.add_argument(
        '-sub_rad',
        type=float,
        metavar='float',
        help='Radius (px) around source to subtract WL image.',
        default=20
    )
    parser.add_argument(
        '-smooth',
        type=float,
        metavar='float',
        help='Smoothing scale. Default: None',
        default=None
    )
    parser.add_argument(
        '-ext',
        type=str,
        metavar='string',
        help='Extension for output image. Default: \'.pNB.fits\' ',
        default='.pNB.fits'
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
    titlestring = """\n{0}\n{1}\n\tCWI_GETNB:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(parser)
    utils.output(titlestring + infostring)

    #Load data
    fits_in = fits.open(args.cube)
    cube, hdr = fits_in[0].data, fits_in[0].header

    #Get QSO position if given
    if args.pos is not None:
        qso_pos = tuple(float(x) for x in args.pos.split(','))
    else:
        qso_pos = None

    #Load variance if given
    if args.var is not None:
        var_cube = fits.getdata(args.var)
    else:
        var_cube = reduction.estimate_variance(fits_in)

    #Apply smoothing if requested
    if args.smooth is not None:
        fits_in[0].data = extraction.smooth_nd(
            fits_in[0].data,
            args.smooth,
            axes=(1,2)
        )
        if args.var is not None:
            var_cube = extraction.smooth_nd(
                var_cube,
                args.smooth,
                axes=(1,2),
                var=True
            )

    nb_fits, wl_fits, nbvar_fits, wlvar_fits = synthesis.pseudo_nb(
        fits_in,
        args.center,
        args.width,
        pos = qso_pos,
        fit_rad = args.fit_rad,
        sub_rad = args.sub_rad,
        var_cube = var_cube
    )

    nb_out = args.cube.replace(".fits", args.ext)
    nb_fits.writeto(nb_out, overwrite=True)
    utils.output("\n\tSaved %s\n" % nb_out)

    nbvar_out = nb_out.replace(".fits", ".var.fits")
    nbvar_fits.writeto(nbvar_out, overwrite=True)
    utils.output("\tSaved %s\n" % nbvar_out)

    wl_out = nb_out.replace(".fits", ".WL.fits")
    wl_fits.writeto(wl_out, overwrite=True)
    utils.output("\tSaved %s\n" % wl_out)

    wlvar_out = nb_out.replace(".fits", ".WL.var.fits")
    wlvar_fits.writeto(wlvar_out, overwrite=True)
    utils.output("\tSaved %s\n" % wlvar_out)

if __name__=="__main__": main()
