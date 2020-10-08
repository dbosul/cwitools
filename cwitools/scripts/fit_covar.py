"""Fit covariance calibration curve given 3D data and variance."""
#Standard Imports

#Third-party Imports
import argparse
from astropy.io import fits

#CWITools Imports
from cwitools import reduction, utils, config


def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(description='Fit covariance calibration\
    curve given 3D data and variance.')
    parser.add_argument(
        'cube',
        type=str,
        metavar='int_cube',
        help='Input data cube.'
        )
    parser.add_argument(
        'var',
        type=str,
        help='Input variance cube.',
        metavar='var_cube'
        )
    parser.add_argument(
        '-wrange',
        type=float,
        nargs=2,
        metavar='Wav Mask',
        help='Wavelength range to use for extracting curve',
        default=None
        )
    parser.add_argument(
        '-alpha_bounds',
        type=float,
        nargs=2,
        help='Range of allowable alpha values.',
        default=None
        )
    parser.add_argument(
        '-norm_bounds',
        type=float,
        nargs=2,
        help='Range of allowable normalization factors.',
        default=None
        )
    parser.add_argument(
        '-thresh_bounds',
        type=float,
        nargs=2,
        help='Range of allowable values for the threshold.',
        default=None
        )
    parser.add_argument(
        '-mask_sky',
        help="Set to auto-mask bright sky lines.",
        action='store_true'
        )
    parser.add_argument(
        '-xybins',
        type=int,
        nargs=2,
        help="A space-separated tuple of ints specifying the range of bin sizes to use (e.g. 1 10)"
        )
    parser.add_argument(
        '-obj',
        type=str,
        help='Object mask - use to remove 3D objects.',
        default=None
        )
    parser.add_argument(
        '-plot',
        help="Set flag to display plot of fit.",
        action='store_true'
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

def fit_covar(cube, var, xybins=None, wrange=None, alpha_bounds=None, norm_bounds=None,
              thresh_bounds=None, mask_sky=False, obj=None, plot=False,
              log=None, silent=None):
    """Fit covariance calibration curve given 3D data and variance.

    Args:
        cube (str): Path to input data cube
        var (str): Path to input variance cube
        xybins (list): List of integer bin sizes, k, such that the data is binned 'k x k' spatially.
            Default is 1 to 1/5 size of the smaller spatial axis. e.g. for input data with 100x60
            spatial dimensions, the bin sizes will be [1, 2, 3, ..., 11]
        wrange (int tuple): Wavelength range to focus on for calibration.
        alpha_bounds (float tuple): Fitting bounds on the alpha parameter
        norm_bounds (float tuple): Fitting bounds on the normalization factor
        thresh_bounds (float tuple): Fitting bounds on the kernel-size threshold
            that separates the logarithmic/flat model regimes.
        mask_sky (bool): Set to TRUE to auto-mask sky lines
        obj (str): Path to FITS containing 3D object mask of regions to exclude.
        plot (bool): Set to True to show diagnostic plots.
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("FIT_COVAR", locals())

    #Try to load the fits file
    data_fits = fits.open(cube)
    var_cube = fits.getdata(var)

    if obj is not None:
        obj_cube = fits.getdata(obj) > 0
    else:
        obj_cube = None

    if alpha_bounds is None:
        alpha_bounds = (0.1, 10)

    if norm_bounds is  None:
        norm_bounds = (1, 2)

    if thresh_bounds is None:
        thresh_bounds = (15, 60)

    fits_out, params, bins, ratios = reduction.variance.fit_covar_xy(
        data_fits,
        var_cube,
        mask=obj_cube,
        mask_sky=mask_sky,
        model_bounds=[alpha_bounds, norm_bounds, thresh_bounds],
        wrange=wrange,
        plot=plot,
        return_all=True,
        xybins=xybins
    )

    utils.output("\t%10s%10s\n" % ("BinArea", "Ratio"))
    for i, bin_i in enumerate(bins):
        utils.output("\t%10i%10.3f\n" % (bin_i, ratios[i]))

    utils.output("\n\tCovariance Model Parameters:\n"
                 "\t\tAlpha\t= %5.2f (header['COV_ALPH'])\n"
                 "\t\tNorm\t= %5.2f (header['COV_NORM'])\n"
                 "\t\tThresh\t= %5.2f (header['COV_THRE'])\n"
                 "\t\tBeta\t= %5.2f (header['COV_BETA'])\n"
                 % (params[0], params[1], params[2], 0.00)
                 )

    fits_out.writeto(cube, overwrite=True)
    utils.output("\n\tUpdated header and saved to %s\n" % cube)
    config.restore_output_mode()

def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()

    if args.xybins is not None:
        args.xybins = range(args.xybins[0], args.xybins[1] + 1)

    fit_covar(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
