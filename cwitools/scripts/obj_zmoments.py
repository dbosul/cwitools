"""Create 2D maps of velocity and dispersion."""

#Standard Imports
import argparse
import os

#Third-party Imports
from astropy.io import fits
import numpy as np

#Local Imports
from cwitools import  extraction, reduction, utils, synthesis, config

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Create 2D maps of velocity and dispersion."""
    )
    parser.add_argument(
        'cube',
        type=str,
        metavar='cube',
        help='The input data cube.'
    )
    parser.add_argument(
        'obj',
        type=str,
        metavar='path',
        help='Object Mask cube.',
    )
    parser.add_argument(
        'obj_id',
        type=int,
        nargs='+',
        metavar='<id1 id2 ... idN>',
        help='The ID(s) of the object to use, space-separated. Use -1 for all objects.',
        default=[1]
    )
    parser.add_argument(
        '-var',
        type=str,
        metavar='path',
        help='Variance cube, to apply inverse variance weighting.',
    )
    parser.add_argument(
        '-r_smooth',
        type=float,
        help='Smooth spatial axes before calculating moments (FWHM).'
    )
    parser.add_argument(
        '-w_smooth',
        type=float,
        help='Smooth wavelength axis before calculating moments (FWHM).'
    )
    parser.add_argument(
        '-unit',
        type=str,
        help="Output mode for units of moment maps.",
        choices=['kms', 'wav'],
        default='wav'
    )
    parser.add_argument(
        '-label',
        type=str,
        help='Label for output file (e.g. "LyA" or "HeII"). Default is "objXX", where XX is obj_id'
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
    return parser

def obj_zmoments(cube, obj, obj_id=1, var=None, r_smooth=None, w_smooth=None, unit='wav',
                 label=None, log=None, silent=None):
    """Create 2D maps of velocity and dispersion.

    Args:
        cube (str): Path to input data cube
        obj (str): Path to FITS containing 3D object masks.
        obj_id (int or list): ID (or list of IDs) of object(s) to include when
            calculating z-moments.
        var (str): Path to input variance cube.
        r_smooth (float): Spatial smoothing scale to use before moments calculation,
            given as FWHM of a Gaussian kernel.
        w_smooth (float): Wavelength smoothing scale to use before moments calculation,
            given as FWHM of a Gaussian kernel.
        unit (str): Output units for moments maps, either 'wav' for Angstroms or
            'kms' for kilometers per second.
        label (str): Custom label for output file name, which will add .<label>_m1.fits to the
            input file name for the first moment map. e.g. provide "LyA" to get ".LyA_m1.fits",
            ".LyA_m2.fits" and so on. By default, the label will "objXX" where XX is the objID for
            a single ID, or the first ID followed by a '+' for a list of IDs.
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("MOMENTS", locals())

    #Try to load the fits file
    if os.path.isfile(cube):
        data_fits = fits.open(cube)
    else:
        raise FileNotFoundError(cube)

    #Try to load the fits file
    if var is not None:
        if os.path.isfile(var):
            var_fits = fits.open(var)
            var_cube = var_fits[0].data
            var_cube[var_cube <= 0] = np.inf
        else:
            raise FileNotFoundError(var)
    else:
        utils.output("\tNo variance input given. Variance will be estimated.\n")
        var_cube = reduction.variance.estimate_variance(data_fits)

    if r_smooth is not None:
        data_fits[0].data = extraction.smooth_nd(
            data_fits[0].data,
            r_smooth,
            axes=(1, 2)
        )
        var_cube = extraction.smooth_nd(
            var_cube,
            r_smooth,
            axes=(1, 2),
            var=True
        )

    if w_smooth is not None:
        data_fits[0].data = extraction.smooth_nd(
            data_fits[0].data,
            w_smooth,
            axes=[0]
        )
        var_cube = extraction.smooth_nd(
            var_cube,
            w_smooth,
            axes=[0],
            var=True
        )

    if w_smooth is not None or r_smooth is not None:
        var_cube, _ = reduction.variance.scale_variance(var_cube, data_fits[0].data)

    #Load object mask
    obj_cube = fits.getdata(obj)

    m1_fits, m1err_fits, m2_fits, m2err_fits = synthesis.obj_moments(
        data_fits,
        obj_cube,
        obj_id,
        var_cube=var_cube,
        unit=unit
    )

    if label is None:
        if isinstance(obj_id, int):
            label = "obj%02i" % obj_id
        elif isinstance(obj_id, list):
            label = "obj%02i+" % obj_id[0]

    m1_out = cube.replace('.fits', '.%s_m1.fits' % label)
    m1_fits.writeto(m1_out, overwrite=True)
    utils.output("\tSaved %s\n" % m1_out)

    m1err_out = m1_out.replace(".fits", "_err.fits")
    m1err_fits.writeto(m1err_out, overwrite=True)
    utils.output("\tSaved %s\n" % m1err_out)

    m2_out = m1_out.replace("m1.fits", "m2.fits")
    m2_fits[0].header["BUNIT"] = "km/s"
    m2_fits.writeto(m2_out, overwrite=True)
    utils.output("\tSaved %s\n" % m2_out)

    m2err_out = m2_out.replace(".fits", "_err.fits")
    m2err_fits.writeto(m2err_out, overwrite=True)
    utils.output("\tSaved %s\n" % m2err_out)

    config.restore_output_mode()


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    obj_zmoments(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
