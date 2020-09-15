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
        '-obj_id',
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

def get_zmoments(cube, obj, obj_id=1, var=None, r_smooth=None, w_smooth=None, unit='wav',
         log=None, silent=None):
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
        utils.output("\tNo variance input given. Variance will be estimated.")
        var_cube = reduction.estimate_variance(data_fits)

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
        var_cube, _ = reduction.scale_variance(var_cube, data_fits[0].data)

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
        var_cube, _ = reduction.scale_variance(var_cube, data_fits[0].data)

    #Load object mask
    obj_cube = fits.getdata(obj)

    m1_fits, m1err_fits, m2_fits, m2err_fits = synthesis.obj_moments(
        data_fits,
        obj_cube,
        obj_id,
        var_cube=var_cube,
        unit=unit
    )

    m1_out = cube.replace('.fits', '.m1.fits')
    m1_fits.writeto(m1_out, overwrite=True)
    utils.output("\tSaved %s" % m1_out)

    m1err_out = cube.replace('.fits', '.m1_err.fits')
    m1err_fits.writeto(m1err_out, overwrite=True)
    utils.output("\tSaved %s" % m1err_out)

    m2_out = cube.replace('.fits', '.m2.fits')
    m2_fits[0].header["BUNIT"] = "km/s"
    m2_fits.writeto(m2_out, overwrite=True)
    utils.output("\tSaved %s" % m2_out)

    m2err_out = cube.replace('.fits', '.m2_err.fits')
    m2err_fits.writeto(m2err_out, overwrite=True)
    utils.output("\tSaved %s" % m2err_out)

    config.restore_output_mode()

#Call using dict and argument parser if run from command-line
if __name__ == "__main__":

    arg_parser = parser_init()
    args = arg_parser.parse_args()

    get_zmoments(**vars(args))
