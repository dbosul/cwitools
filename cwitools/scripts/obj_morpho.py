"""Measure the size and symmetry of a 3D object."""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits
from astropy.cosmology import WMAP5, WMAP7, WMAP9, Planck13, Planck15

#Local Imports
from cwitools import utils, config, measurement, synthesis

COSMO_DICT = {'WMAP5':WMAP5, 'WMAP7':WMAP7, 'WMAP9':WMAP9, 'Planck13':Planck13, 'Planck15':Planck15}

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Measure the size and symmetry of a 3D object."""
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
        help='The input object ID or space-separated list of multiple IDs. Objects will be measured\
        individually.'
    )
    parser.add_argument(
        '-redshift',
        type=float,
        help='Redshift of the emission. Needed to calculate physical distances.',
    )
    parser.add_argument(
        '-cosmology',
        type=str,
        help='The cosmological parameters to use WMAP5, WMAP7, WMAP9, Planck13 or Planck15.\
        Used to calculate physical distances. Default is WMAP9.',
        choices=['WMAP5', 'WMAP7', 'WMAP9', 'Planck13', 'Planck15'],
        default='WMAP9'
    )
    parser.add_argument(
        '-r_unit',
        type=str,
        help="The unit for radii: pixels ('px'), arcseconds ('arcsec'), proper kpc ('pkpc'), or\
        comoving kiloparsecs ('ckpc').",
        choices=['px', 'arcsec', 'pkpc', 'ckpc'],
        default='WMAP9'
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

def obj_morpho(cube, obj, obj_id, cosmology='WMAP9', redshift=None, r_unit='px', log=None,
               silent=None):
    """Measure the size and symmetry of an object.

    Args:
        fits_in (str): The input data FITS (can be 1D, 2D or 3D)
            If input is 2D, it is assumed to have units erg/s/cm2/arcsec2.
            If input is 1D or 3D, units are assumed to be erg/s/cm2/angstrom
        obj (numpy.ndarray): The object masj cube
        obj_id (int or list): The ID (or IDs) of the object(s) to include.
        cosmology (str): One of the built-in astropy cosmologies, can be 'WMAP5', 'WMAP7', 'WMAP9',
            'Planck13' or 'Planck15'.
        redshift (float): The redshift of the source.
        r_unit (str): The unit to use for measured radii. Choices:
            'px' - pixels
            'arcsec' - arcseconds
            'pkpc' - proper kiloparsec
            'ckpc' - comoving kiloparsec
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.


    Returns:
        float: The integrated luminosity of the source in erg/s.
        float: The error on the luminosity calculation.

    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("OBJ_MORPHO", locals())

    if cosmology not in COSMO_DICT.keys():
        raise ValueError("Cosmology %s not included in current version." % cosmology)

    cosmo = COSMO_DICT[cosmology]
    int_fits = fits.open(cube)
    obj_fits = fits.open(obj)

    u_str = "[" + r_unit + "]"
    utils.output("\n#%7s %15s %15s %15s %15s\n" %
                 ("OBJ_ID", "R_eff" + u_str, "R_max" + u_str, "R_rms" + u_str, "Ecc.")
                )

    for o_id in obj_id:

        r_eff = measurement.eff_radius(
            obj_fits, o_id,
            unit=r_unit,
            redshift=redshift,
            cosmo=cosmo
            )
        r_max = measurement.max_radius(
            int_fits, obj_fits[0].data, o_id,
            unit=r_unit,
            redshift=redshift,
            cosmo=cosmo
            )
        r_rms = measurement.rms_radius(
            int_fits, obj_fits[0].data, o_id,
            unit=r_unit,
            redshift=redshift,
            cosmo=cosmo
            )

        sb_map = synthesis.obj_sb(int_fits, obj_fits[0].data, o_id)[0].data

        asym = measurement.eccentricity(sb_map)

        utils.output("%8i %15.2f %15.2f %15.2f %15.2f\n" % (o_id, r_eff, r_max, r_rms, asym))

    config.restore_output_mode()


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    obj_morpho(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
