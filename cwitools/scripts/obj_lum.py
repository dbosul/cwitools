"""Measure the integrated luminosity of an object."""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits
from astropy.cosmology import WMAP5, WMAP7, WMAP9, Planck13, Planck15

#Local Imports
from cwitools import utils, config, measurement, extraction

COSMO_DICT = {'WMAP5':WMAP5, 'WMAP7':WMAP7, 'WMAP9':WMAP9, 'Planck13':Planck13, 'Planck15':Planck15}

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Measure the integrated luminosity of an object."""
    )
    parser.add_argument(
        'cube',
        type=str,
        help='The input data cube FITS file.'
    )
    parser.add_argument(
        'obj',
        type=str,
        help='The input object mask cube FITS file.'
    )
    parser.add_argument(
        'obj_id',
        type=int,
        nargs='+',
        help='The input object ID or IDs (space-separated).'
    )
    parser.add_argument(
        'redshift',
        type=float,
        help='Redshift of the emission.',
    )
    parser.add_argument(
        '-cosmology',
        type=str,
        help='The cosmological parameters to use WMAP5, WMAP7, WMAP9, Planck13 or Planck15.\
        Default is WMAP9.',
        choices=['WMAP5', 'WMAP7', 'WMAP9', 'Planck13', 'Planck15'],
        default='WMAP9'
    )
    parser.add_argument(
        '-var',
        type=str,
        help='Variance cube FITS file.',
    )
    parser.add_argument(
        '-log',
        metavar="<log_file>",
        type=str,
        help="Log file to save output in."
    )
    parser.add_argument(
        '-silent',
        help="Set flag to suppress standard terminal output.",
        action='store_true'
    )
    return parser

def obj_lum(cube, obj, obj_id, redshift, cosmology='WMAP9', var=None, log=None, silent=None):
    """Measure the integrated luminosity of an object.

    Args:
        fits_in (str): The input data FITS (can be 1D, 2D or 3D)
            If input is 2D, it is assumed to have units erg/s/cm2/arcsec2.
            If input is 1D or 3D, units are assumed to be erg/s/cm2/angstrom
        obj (numpy.ndarray): The object masj cube
        obj_id (int or list): The ID (or IDs) of the object(s) to include.
        redshift (float): The redshift of the source.
        cosmology (str): One of the built-in astropy cosmologies, can be 'WMAP5', 'WMAP7', 'WMAP9',
            'Planck13' or 'Planck15'.

        var (str): Array of same dimensions as data and mask, containing variance estimates.
            Used to propagate error on luminosity.
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.


    Returns:
        float: The integrated luminosity of the source in erg/s.
        float: The error on the luminosity calculation.

    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("OBJ_LUM", locals())

    if cosmology not in COSMO_DICT.keys():
        raise ValueError("Cosmology %s not included in current version." % cosmology)

    int_fits = fits.open(cube)
    obj_cube = fits.getdata(obj)
    var_cube = None if var is None else fits.getdata(var)

    #Correct input if it is in 10^16 * FLAM, the standard KCWI units
    if "FLAM16" in int_fits[0].header["BUNIT"]:
        int_fits[0].data /= 1e16
        if var_cube is not None:
            var_cube /= 1e32

    utils.output("\n#%7s %15s %15s\n" % ("OBJ_ID", "L [erg/s]", "L_ERR [erg/s]"))

    for o_id in obj_id:

        bin_mask = extraction.obj2binary(obj_cube, o_id)

        lum, lum_err = measurement.luminosity(
            int_fits,
            redshift=redshift,
            mask=bin_mask,
            cosmo=COSMO_DICT[cosmology],
            var_data=var_cube
            )
        utils.output("%8i %15.4E %15.4E\n" % (o_id, lum, lum_err))

    if var is None:
        utils.output("(Note: No variance input given. Error estimated from variance in data cube.)\n")

    config.restore_output_mode()
    return lum, lum_err


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    obj_lum(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
