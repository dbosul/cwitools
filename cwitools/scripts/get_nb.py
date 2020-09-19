"""Generate a pseudo-Narrowband image"""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits

#Local imports
from cwitools import extraction, reduction, utils, synthesis, config

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description='Generate a pseudo-Narrowband image'
        )
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
        help='Central wavelength to use for pseudo NB. Default: None.'
        )
    parser.add_argument(
        'width',
        type=float,
        metavar='float',
        help='Pseudo-NB width in Angstrom.'
        )
    parser.add_argument(
        '-var',
        type=str,
        metavar='var',
        help='Variance cube for SNR calculations. Estimated if not given.',
        )
    parser.add_argument(
        '-pos',
        type=float,
        nargs=2,
        metavar='<x y>',
        help='Position of your source as an \'x,y\' tuple.  Default: None',
        )
    parser.add_argument(
        '-r_fit',
        type=float,
        metavar='float',
        help='Radius (arcsec) to use for scaling PSF image. Default: 3px',
        default=1
        )
    parser.add_argument(
        '-r_sub',
        type=float,
        metavar='float',
        help='Radius (px) around source to subtract WL image.',
        default=20
        )
    parser.add_argument(
        '-smooth',
        type=float,
        metavar='float',
        help='Smoothing scale. Default: None'
        )
    parser.add_argument(
        '-ext',
        type=str,
        metavar='string',
        help='Extension for output image. Default: \'.pNB.fits\' ',
        default='.pNB.fits'
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


def get_nb(cube, center, width, var=None, pos=None, r_fit=1.5, r_sub=20,
           smooth=None, ext=".pNB.fits", log=None, silent=None):
    """Generate a pseudo-NB image from a data cube.

    Args:
        cube (str): Path to input data cube FITS file
        center (float): The central wavelength of the pNB, in Angstrom.
        width (float): The bandwidth of the pNB, in Angstrom.
        var (str): Path to associated variance cube FITS file
        pos (float tuple): Provide the x,y location the source to subtract.
            Leave empty to skip white-light subtraction.
        r_fit (float): Radius (px) to use for scaling the PSF.
        r_sub (float): Radius (px) to use when subtracting PSF.
        smooth (float): Size (FWHM) of Gaussian smoothing kernel to apply to NB
        ext (str): File extension to use for masked FITS (".M.fits")
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("GET_NB", locals())

    #Load data
    fits_in = fits.open(cube)

    #Load variance if given
    if var is not None:
        var_cube = fits.getdata(var)
    else:
        var_cube = reduction.variance.estimate_variance(fits_in)

    #Apply smoothing if requested
    if smooth is not None:
        fits_in[0].data = extraction.smooth_nd(
            fits_in[0].data,
            smooth,
            axes=(1, 2)
        )
        if var is not None:
            var_cube = extraction.smooth_nd(
                var_cube,
                smooth,
                axes=(1, 2),
                var=True
            )

    nb_fits, nbvar_fits, wl_fits, wlvar_fits = synthesis.pseudo_nb(
        fits_in,
        center,
        width,
        pos=pos,
        fit_rad=r_fit,
        sub_rad=r_sub,
        var_cube=var_cube
    )

    nb_out = cube.replace(".fits", ext)
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
    config.restore_output_mode()

def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    get_nb(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
