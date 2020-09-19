"""Mask specific wavelength ranges in a cube."""

#Standard Imports
import argparse
import os

#Third-party Imports
from astropy.io import fits
import numpy as np

#Local Imports
from cwitools import  utils, coordinates, config

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Mask specific wavelength ranges in a cube."""
    )
    parser.add_argument(
        'data',
        type=str,
        help='Data to be masked.'
    )
    parser.add_argument(
        '-wmask',
        type=str,
        nargs='+',
        help="Wavelength ranges to mask, each in the form A:B and separated by spaces."
    )
    parser.add_argument(
        '-mask_sky',
        help="Set flag to auto-mask some known bright sky lines.",
        action='store_true'
    )
    parser.add_argument(
        '-out',
        type=str,
        help="Output file name. Default is to add .zmask.fits "
    )
    parser.add_argument(
        '-log',
        type=str,
        metavar="<log_file>",
        help="Log file to save output in."
    )
    parser.add_argument(
        '-silent',
        help="Set flag to suppress standard terminal output.",
        action='store_true'
    )
    return parser

def mask_z(data, wmask=None, mask_sky=False, out=None, log=None, silent=None):
    """Mask specific wavelength ranges in a cube.

    Args:
        data (str): Path to the input data (FITS file) to be masked.
        wmask (list): List of wavelength ranges to mask, given as a list of
            float tuples in units of Angstroms. e.g. [(4100,4200), (5100,5200)]
        mask_sky (bool): Set to TRUE to auto-mask sky emission lines.
        out (str): File extension to use for masked FITS (".M.fits")
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("MASK_Z", locals())

    if os.path.isfile(data):
        data_fits = fits.open(data)
    else:
        raise FileNotFoundError(data)

    if not(mask_sky) and (wmask is None):
        raise SyntaxError("Must provide mask_sky and/or masks argument.")

    if mask_sky:
        sky_mask = utils.get_skymask(data_fits[0].header)
        data_fits[0].data[sky_mask] = 0

    if wmask is not None:
        wav_axis = coordinates.get_wav_axis(data_fits[0].header)
        zmask = np.zeros_like(wav_axis, dtype=bool)
        for (wav0, wav1) in wmask:
            zmask[(wav_axis >= wav0) & (wav_axis <= wav1)] = 1
        data_fits[0].data[zmask] = 0

    if out is None:
        out = data.replace('.fits', '.zmask.fits')

    data_fits.writeto(out, overwrite=True)

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

    mask_z(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
