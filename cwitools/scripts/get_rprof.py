"""Generate a surface brightness map of a 3D object."""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import utils, config, synthesis

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Generate a surface brightness map of a 3D object."""
    )
    parser.add_argument(
        'sb_map',
        type=str,
        help='The input data cube FITS file.'
    )
    parser.add_argument(
        'pos',
        type=float,
        nargs=2,
        help='The central coordinate of the radial profile, as a space-separated float tuple.'
    )
    parser.add_argument(
        '-pos_type',
        type=str,
        help="The type of coordinate given as the 'pos' argument, eithe 'image' or 'radec'. Default\
        is 'image', meaning pos is given as (x, y) image coordinate, in pixels.",
        choices=['image', 'radec'],
        default='image'
    )
    parser.add_argument(
        '-r_min',
        type=float,
        help='The innermost radius of the radial profile. Default: 0.'
    )
    parser.add_argument(
        '-r_max',
        type=float,
        help='The innermost radius of the radial profile. Default: maximum extent of input.'
    )
    parser.add_argument(
        '-n_bins',
        type=int,
        help='The number of radial bins between r_min and r_max. Default: 10',
        default=10
    )
    parser.add_argument(
        '-scale',
        type=str,
        help="'log' create bins with equal size in Log(R), 'linear' to create bins with equal size\
        in R. Default: 'linear'",
        choices=['log', 'linear'],
        default='linear'
    )
    parser.add_argument(
        '-r_unit',
        type=str,
        help="The radial unit to use: pixels ('px'), arcseconds ('arcsec'), proper kpc ('pkpc'),\
        comoving kpc ('ckpc')",
        choices=['px', 'arcsec', 'pkpc', 'ckpc'],
        default='px'
    )
    parser.add_argument(
        '-redshift',
        type=float,
        help='The redshift of the emission. Needed to calculate physical units.'
    )
    parser.add_argument(
        '-var',
        type=str,
        help='Variance cube FITS file.'
    )
    parser.add_argument(
        '-mask',
        type=str,
        help='Mask FITS file, to exclude certain spaxels from the radial profile.'
    )
    parser.add_argument(
        '-out',
        type=str,
        help='Output file name. Default is input name with .rprof.fits extension added.'
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

def get_rprof(sb_map, pos, pos_type='image', r_min=None, r_max=None, n_bins=10, scale='lin',
              r_unit='px', redshift=None, var=None, mask=None, out=None, log=None, silent=None):
    """Generate a surface brightness map of a 3D object.

    Args:
        fits_in (str): Path to FITS file containing SB map.
        pos (float tuple): The center of the profile in units determined by the 'pos_type' argument.
            Default is image coordinates, (x, y).
        pos_type (str): The type of coordinate given for the 'pos' argument.
            'radec' - (RA, DEC) tuple in decimal degrees
            'image' - (x, y) tuple in image coordinates (default)
        r_min  (float): The minimum radius, in units determined by runit.
        r_max (float): The maximum radius, in units determined by runit.
        nbins (int): The number of radial bins between r_min  and r_max to use.
        scale (str): The scale for the radial bins.
            'linear' makes bins equal size in linear space.
            'log' makes bins equal size in log space.
        runit (str): The unit of r_min  and r_max. Can be 'pkpc' or 'px'
            'pkpc' Proper kiloparsec, redshift must also be provided.
            'px' pixels (i.e. distance in image coordinates)
        redshift (float): The redshift of the emission, needed to calculate physical distances.
        var (str): Path to FITS file containing SB map variance.
        mask (str): Path to FITS file containing 2D binary mask of regions to exclude.
        out (str): The output file name for the radial profile. Default is to add ".rprof.fits"
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.
    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("OBJ_SB", locals())

    sb_fits = fits.open(sb_map)

    var_map = None if var is None else fits.getdata(var)

    rprof_fits = synthesis.radial_profile(
        sb_fits,
        pos,
        r_min=r_min,
        r_max=r_max,
        n_bins=n_bins,
        scale=scale,
        mask=mask,
        var=var_map,
        r_unit=r_unit,
        redshift=redshift,
        pos_type=pos_type
    )

    if out is None:
        out = sb_map.replace(".fits", ".rprof.fits")

    rprof_fits.writeto(out, overwrite=True)

    utils.output("\tSaved %s\n" % out)
    config.restore_output_mode()

def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    get_rprof(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
