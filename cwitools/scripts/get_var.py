"""Estimate 3D variance based on an input data cube."""

#Standard Imports
import argparse
import os

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import utils, config, reduction

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Estimate 3D variance based on an input data cube.""")
    parser.add_argument(
        'cube',
        type=str,
        metavar='path',
        help='Input cube whose 3D variance you would like to estimate.'
    )
    parser.add_argument(
        '-window',
        type=int,
        help='Size of wavelength bin, in Angstrom, for 2D layer variance estimate.',
        default=50
    )
    parser.add_argument(
        '-wmask',
        type=str,
        nargs='+',
        metavar='<w0:w1>',
        help='Wavelength range(s) in the form (A:B) to mask when fitting.'
    )
    parser.add_argument(
        '-mask_neb',
        metavar='<redshift>',
        type=float,
        help='Prove redshift to auto-mask nebular emission.'
    )
    parser.add_argument(
        '-vwidth',
        metavar='<km/s>',
        type=float,
        help='Velocity width (km/s) around nebular lines to mask, if using -mask_neb.',
        default=500
    )
    parser.add_argument(
        '-out',
        type=str,
        metavar='str',
        help='Filename for output. Default is input + .var.fits'
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

def get_var(cube, window=50, wmask=None, mask_neb_z=None, mask_neb_dv=500, out=None,
            log=None, silent=None):
    """Estimate 3D variance based on an input data cube.

    Args:
        cube (str): Path to input data cube FITS file
        window (int): Wavelength window (Angstrom) to use for z-bins, used
            to get local 2D variance estimation.
        wmask (list): List of wavelength ranges to exclude from analysis images,
            provided as a list of float-like tuples e.g. [(4100,4200), (5100,5200)]
        mask_neb_z (float): Redshift of nebular emission to auto-mask.
        mask_neb_dv (float): Velocity width, in km/s, of nebular emission masks.
        out (str): Output file name for estimate variance FITS file.
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("GET_VAR", locals())

    #Try to load the fits file
    if os.path.isfile(cube):
        data_fits = fits.open(cube)
    else:
        raise FileNotFoundError("Input file not found.")

    if wmask is None:
        wmask = []
    if mask_neb_z is not None:
        wmask += utils.get_nebmask(
            data_fits[0].header,
            redshift=mask_neb_z,
            vel_window=mask_neb_dv,
            mode='tuples'
        )

    vardata = reduction.variance.estimate_variance(
        data_fits,
        window=window,
        wmasks=wmask
    )

    if out is None:
        out = cube.replace('.fits', '.var.fits')

    var_fits = fits.HDUList([fits.PrimaryHDU(vardata)])

    var_fits[0].header = data_fits[0].header
    var_fits.writeto(out, overwrite=True)
    utils.output("\tSaved %s\n" % out)
    config.restore_output_mode()


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()

    #Parse wmask argument properly into list of float-tuples
    if isinstance(args.wmask, list):
        try:
            for i, wpair in enumerate(args.wmask):
                args.wmask[i] = tuple(float(x) for x in wpair.split(':'))
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)

    get_var(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
