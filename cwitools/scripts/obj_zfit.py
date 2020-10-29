"""Create 2D maps of velocity and dispersion."""

#Standard Imports
import argparse
import os

#Third-party Imports
from astropy.io import fits
import numpy as np

#Local Imports
from cwitools import  reduction, utils, synthesis, config

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Create 2D maps of velocity and dispersion using a singlet or doublet line\
        model."""
    )
    parser.add_argument(
        'cube',
        type=str,
        metavar='int_cube',
        help='The input data cube.'
    )
    parser.add_argument(
        'obj',
        type=str,
        metavar='obj_cube',
        help='Object Mask cube.',
    )
    parser.add_argument(
        'peak_wav',
        type=float,
        nargs='+',
        help="Peak (or peaks, for doublets) wavelength of the emission. Can provide rest-frame with\
        redshift or observed wavelength."
    )
    parser.add_argument(
        '-obj_id',
        type=int,
        help='The ID (singular) of the object to use, space-separated. Use -1 for all objects.',
        default=1
    )
    parser.add_argument(
        '-var',
        type=str,
        metavar='<var_path>',
        help='Variance cube, to apply inverse variance weighting.',
    )
    parser.add_argument(
        '-unit',
        type=str,
        help="Output mode for units of moment maps.",
        choices=['kms', 'wav'],
        default='wav'
    )
    parser.add_argument(
        '-redshift',
        metavar="<float>",
        type=float,
        default=0,
        help="Redshift of the emission"
    )
    parser.add_argument(
        '-vel_max',
        metavar="<km/s>",
        type=float,
        help='Max velocity offset for line fitting',
        default=2000
    )
    parser.add_argument(
        '-disp_bounds',
        metavar="<km/s>",
        type=float,
        nargs=2,
        help='Min/max dispersion (km/s) for line fitting',
        default=(50, 500)
    )
    parser.add_argument(
        '-ratio_bounds',
        metavar="<float>",
        type=float,
        nargs=2,
        help='Min/max blue-peak to red-peak ratio for doublet fitting.',
        default=(0.5, 2.0)
    )
    parser.add_argument(
        '-label',
        metavar="<e.g. LyA>",
        type=str,
        help='Label for output (e.g. NV, CIV_1548)',
        default=None
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

def obj_zfit(cube, obj, peak_wav, obj_id=1, var=None, unit='wav', redshift=0, vel_max=2000,
             disp_bounds=(50, 500), ratio_bounds=(0.5, 2.0), label=None, log=None, silent=None):
    """Create 2D maps of velocity and dispersion.

    Args:
        cube (str): Path to input data cube
        obj (str): Path to FITS containing 3D object masks.
        peak_wav (float or float tuple): Peak wavelength for a singlet or tuple of peak wavelengths
            for a doublet. You can provide rest-frame value and redshift, or just provide the
            observed wavelengths and leave redshift = 0.
            For doublet fits, peak separation is fixed, dispersions of both components are tied, and
            the blue-to-red peak ratio is constrained by the ratio_bounds argument.
        obj_id (int or list): ID (or list of IDs) of object(s) to include when
            calculating z-moments.
        var (str): Path to input variance cube.
        unit (str): Output units for moments maps, either 'wav' for Angstroms or
            'kms' for kilometers per second.
        redshift (float): The redshift of the emission, if providing rest-frame wavelengths for
            the 'peak' argument.
        vel_max (float tuple): The maximum velocity offset, in km/s, for the fit center relative to
            the provided peak wavelength(s).
        disp_bounds (float tuple): The lower and upper dispersion bounds for the fit. Dispersions of
            the two components in a doublet fit are tied together.
        ratio_bounds (float tuple): For doublet fits only - the min/max ratio between the blue peak
            and the red peak.
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
    utils.output_func_summary("Z-FIT", locals())

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

    #Load object mask
    obj_cube = fits.getdata(obj)

    m1_fits, m2_fits, model_fits = synthesis.obj_moments_zfit(
        data_fits,
        obj_cube,
        obj_id,
        peak_wav,
        redshift=redshift,
        vel_max=vel_max,
        disp_bounds=disp_bounds,
        ratio_bounds=ratio_bounds,
        unit=unit,
        var=var_cube
    )

    if label is None:
        if isinstance(obj_id, int):
            label = "obj%02i" % obj_id
        elif isinstance(obj_id, list):
            label = "obj%02i+" % obj_id[0]

    m1_out = cube.replace('.fits', '.%s_m1.fits' % label)
    m1_fits.writeto(m1_out, overwrite=True)
    utils.output("\tSaved %s\n" % m1_out)

    m2_out = m1_out.replace("m1.fits", "m2.fits")
    m2_fits.writeto(m2_out, overwrite=True)
    utils.output("\tSaved %s\n" % m2_out)

    model_out = m1_out.replace("_m1.fits", "_model.fits")
    model_fits.writeto(model_out, overwrite=True)
    utils.output("\tSaved %s\n" % model_out)

    config.restore_output_mode()


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    obj_zfit(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
