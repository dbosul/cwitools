"""Perform adaptive kernel smoothing on a data cube."""
#Standard imports
import argparse

#Third-party imports
from astropy.io import fits

#Local imports
from cwitools import extraction, utils, config

def parser_init():
    """Create argument parser for this script."""
    parser = argparse.ArgumentParser(
        description='Perform Adaptive-Kernel-Smoothing on a data cube (requires variance cube).'
        )
    parser.add_argument(
        'int_path',
        type=str,
        metavar='input cube',
        help='The cube to be smoothed.'
        )
    parser.add_argument(
        'var_path',
        type=str,
        metavar='variance',
        help='The associated variance cube.'
        )
    parser.add_argument(
        '-snr_min',
        type=float,
        metavar='float',
        help='The objective minimum signal-to-noise level (Default:3)',
        default=3
        )
    parser.add_argument(
        '-snr_max',
        type=float,
        metavar='float',
        help='(Soft) maximum SNR, used to detect when oversmoothing occurs. Default: 1.1*snr_min',
        default=None
        )
    parser.add_argument(
        '-xy_mode',
        type=str,
        metavar='str',
        help='Spatial moothing mode (box/gaussian) - Default: gaussian',
        default='gaussian',
        choices=['box', 'gaussian']
        )
    parser.add_argument(
        '-z_mode',
        type=str,
        metavar='str',
        help='Wavelength moothing mode (box/gaussian) - Default: gaussian',
        default='gaussian',
        choices=['box', 'gaussian']
        )
    parser.add_argument(
        '-xy_range',
        type=float,
        nargs=2,
        metavar='<float>',
        help='Minimum and maximum spatial smoothing scale (Default:1.5 to 8.0 px)',
        default=(1.5, 8.0)
        )
    parser.add_argument(
        '-z_range',
        type=float,
        metavar='<float>',
        nargs=2,
        help='Minimum and maximum wavelength smoothing scale (Default:1.5 to 6.0 px)',
        default=(1.5, 6.0)
        )
    parser.add_argument(
        '-xy_step_min',
        type=float,
        metavar='float (px)',
        help='Minimum spatial scale step-size (Default:0.1px)',
        default=0.2
        )
    parser.add_argument(
        '-z_step_min',
        type=float,
        metavar='float (px)',
        help='Minimum wavelength scale step-size (Default:0.5px)',
        default=0.2
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

def asmooth(int_path, var_path, snr_min=3, snr_max=None, xy_mode='gaussian', z_mode='box',
            xy_range=(1.5, 8.0), z_range=(1.5, 6.0), xy_step_min=0.5, z_step_min=0.5, log=None,
            silent=False):
    """Perform adaptive kernel smoothing on data.

    3D Algorithm based on 2D algorithm by Ebeling, White & Ranjaran 2006. This 3D algorithm has not
    yet been tested in a peer reviewed journal, and is still somewhat experimental.
    Users are encouraged test the code themselves if they wish to use it for publications.

    Args:
        int_path (str): The path to the intensity cube FITS file
        var_fits (str): The path to the variance cube FITS file.
        snr_min (float): The minimum SNR for voxel detection
        snr_max (float): A soft upper limit on SNR, used to detect when over-smoothing is occurring
        xy_mode (str): The type of kernel to use for spatial (xy) smoothing
            'gaussian' - a 2D Gaussian kernel
            'box' - a 2D Box kernel
        z_mode (str): The type of kernel to use for wavelength-axis (z) smoothing
            'gaussian' - a 2D Gaussian kernel
            'box' - a 2D Box kernel
        xy_range (float tuple): Range of smoothing scales to use for spatial axes
        z_range (float tuple): Range of smoothing scales to use for z-axis
        xy_step_min (float): Minimum step size to use for increasing spatial kernel size
        z_step_min (float): Minimum step size to use for increasing wavelength kernel size

    Returns:
         numpy.ndarray: adaptively smoothed intensity cube
         numpy.ndarray: variance cube associated with smoothed data
         numpy.ndarray: signal-to-noise cube
         numpy.ndarray: mask cube, where 1 = detected
         numpy.ndarray: cube showing spatial kernel sizes used for detections
         numpy.ndarray: cube showing wavelength kenel sizes used for detections
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("ASMOOTH 3D", locals())

    int_fits = fits.open(int_path)
    var_fits = fits.open(var_path)

    res = extraction.asmooth3d(
        int_fits, var_fits,
        snr_min=snr_min,
        snr_max=snr_max,
        xy_mode=xy_mode,
        z_mode=z_mode,
        xy_range=xy_range,
        z_range=z_range,
        xy_step_min=xy_step_min,
        z_step_min=z_step_min,
    )

    icube_det_out = int_path.replace(".fits", ".AKS.fits")
    icube_det_fits = utils.match_hdu_type(int_fits, res[0], int_fits[0].header)
    icube_det_fits.writeto(icube_det_out, overwrite=True)
    utils.output("Saved %s\n" % icube_det_out)

    vcube_det_out = int_path.replace(".fits", ".AKS.var.fits")
    vcube_det_fits = utils.match_hdu_type(int_fits, res[1], var_fits[0].header)
    vcube_det_fits.writeto(vcube_det_out, overwrite=True)
    utils.output("Saved %s\n" % vcube_det_out)

    mcube_det_out = int_path.replace(".fits", ".AKS.mask.fits")
    mcube_det_fits = utils.match_hdu_type(int_fits, res[2], int_fits[0].header)
    mcube_det_fits.writeto(mcube_det_out, overwrite=True)
    utils.output("Saved %s\n" % mcube_det_out)

    snr_det_out = int_path.replace(".fits", ".AKS.snr.fits")
    snr_det_fits = utils.match_hdu_type(int_fits, res[3], int_fits[0].header)
    snr_det_fits.writeto(snr_det_out, overwrite=True)
    utils.output("Saved %s\n" % snr_det_out)

    kr_vals_out = int_path.replace(".fits", ".AKS.kr.fits")
    kr_vals_fits = utils.match_hdu_type(int_fits, res[4], int_fits[0].header)
    kr_vals_fits.writeto(kr_vals_out, overwrite=True)
    utils.output("Saved %s\n" % kr_vals_out)

    kw_vals_out = int_path.replace(".fits", ".AKS.kw.fits")
    kw_vals_fits = utils.match_hdu_type(int_fits, res[5], int_fits[0].header)
    kw_vals_fits.writeto(kw_vals_out, overwrite=True)
    utils.output("Saved %s\n" % kw_vals_out)

def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    asmooth(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
