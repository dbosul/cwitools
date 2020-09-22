"""Estimate the 3D variance for a data cube"""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits
import numpy as np

#Local Imports
from cwitools import coordinates, utils, reduction, config

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(description='Get estimated variance cube.')
    parser.add_argument(
        'data',
        type=str,
        help='Data cube.'
        )
    parser.add_argument(
        'var',
        type=str,
        help='Variance cube.'
        )
    parser.add_argument(
        '-snr_min',
        type=float,
        help='SNR Threshold for detection.',
        default=2
        )
    parser.add_argument(
        '-n_min',
        type=int,
        help='Minimum size of detection',
        default=100
        )
    parser.add_argument(
        '-wrange',
        type=float,
        nargs=2,
        metavar='<float>',
        help='Wavelength range to use while fitting'
        )
    parser.add_argument(
        '-plot',
        action='store_true',
        help="Display diagnostic plots."
        )
    parser.add_argument(
        '-out',
        type=str,
        metavar='str',
        help='Filename for output. Default is input + .scaled.fits',
        default=None
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

def scale_var(data, var, snr_min=2, n_min=100, wrange=None, plot=False, out=None, log=None,
              silent=True):
    """Scale a variance estimate to match the noise properties of the associated data.

    Args:
        data (str): Path to the 3D data FITS we are estimating variance for
        var (str): Path to the FITS containing the initial 3D variance estimate
        snr_min (float): Signal-to-noise ratio (SNR) threshold to use for iterative scaling method.
            Contiguous regions of size n_min above a SNR of snr_min will be rejected as systematics
            or emission regions, and the scaling will be based only on remaining background regions.
        n_min (int): Minimum size of a contiguous region with SNR > snr_min to count as a systematic
            and be excluded from the variance scaling.
        wrange (float tuple): The range of wavelengths to use when scaling the variance estimate,
            in units of Angstrom.
        plot (bool): Set to TRUE to show diagnostic plots.
        snr_range (float tuple): The range of SNR values to use when finding scaling factor. Default
            is -5 to +5.
        snr_bins (int): The number of SNR bins across snr_range to use for generating histograms.
            Scaling factors are determined by best-fit Gaussian models to SNR histograms, assuming
            background (i.e. shot-noise) limited observations. Default: 100


    Returns:
        numpy.ndarray: The rescaled variance estimate
        float: The final rescaling factor, f, such that var_out = f * var_in
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("SCALE_VAR", locals())

    data_fits = fits.open(data)
    var_fits = fits.open(var)

    if wrange is not None:
        wav_axis = coordinates.get_wav_axis(data_fits[0].header)
        zmask = (wav_axis >= wrange[0]) & (wav_axis <= wrange[1])
        data_fit = data_fits[0].data[zmask]
        var_fit = var_fits[0].data[zmask]
    else:
        data_fit = data_fits[0].data
        var_fit = var_fits[0].data

    scaled_var, scale_factor = reduction.variance.scale_variance(
        data_fit,
        var_fit,
        n_min=n_min,
        snr_min=snr_min,
        plot=plot
    )
    var_out = var_fits[0].data * scale_factor

    if out is None:
        out = var.replace('.fits', '.scaled.fits')

    utils.output("Std-dev of Noise SNR Distribution = %.3f\n" % np.sqrt(scale_factor))
    utils.output("Variance Scaled by %.3f to assert Standard Normal Distribution\n" % scale_factor)

    scaled_var_fits = utils.match_hdu_type(var_fits, var_out, var_fits[0].header)

    scaled_var_fits.writeto(out, overwrite=True)
    utils.output("Saved %s\n" % out)
    config.set_temp_output_mode(log, silent)

def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    scale_var(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
