"""Generate a White-light image from a data cube"""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import reduction, utils, synthesis
import cwitools

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Generate a White-light image from a data cube"""
    )
    parser.add_argument(
        'cube',
        type=str,
        help='The input data cube.'
    )
    parser.add_argument(
        '-wmask',
        type=str,
        metavar='Wav Mask',
        help='Wavelength range(s) to mask when making WL image.'
    )
    parser.add_argument(
        '-var',
        type=str,
        help='Variance cube, for calculating WL variance. Estimated if not given.'
    )
    parser.add_argument(
        '-out',
        help="Output file name. Default is parameter file + .WL.fits"
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

def main(cube, wmask=None, var=None, out=None, log=None, silent=True, arg_parser=None):
    """Generate a White-light image from a data cube"""\

    #Set global parameters
    cwitools.silent_mode = silent
    cwitools.log_file = log

    utils.output_func_summary("GET_WL", locals())

    #Load data
    data_fits = fits.open(cube)

    if var is not None:
        var_cube = fits.getdata(var)
    else:
        var_cube = reduction.estimate_variance(data_fits)

    #utils.output("%s,"%args.cube.split('/')[-2], end='')
    wl_fits, wl_var_fits = synthesis.whitelight(
        data_fits,
        var_cube=var_cube,
        wmask=wmask
    )

    if out is None:
        out = cube.replace('.fits', '.WL.fits')

    var_out = out.replace(".fits", ".var.fits")

    wl_fits.writeto(out, overwrite=True)
    utils.output("\tSaved %s\n" % out)

    wl_var_fits.writeto(var_out, overwrite=True)
    utils.output("\tSaved %s\n" % var_out)

#Call using dict and argument parser if run from command-line
if __name__ == "__main__":

    arg_parser = parser_init()
    args = arg_parser.parse_args()

    #Parse wmask argument properly into list of float-tuples
    if isinstance(args.wmask, list):
        try:
            for i, wpair in enumerate(args.wmask):
                args.wmask[i] = tuple(float(x) for x in wpair.split(':'))
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)

    main(**vars(args))
