"""Mask specific wavelength ranges in a cube."""

#Standard Imports
import argparse
import os

#Third-party Imports
from astropy.io import fits
import numpy as np

#Local Imports
from cwitools import  utils, coordinates
import cwitools

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

def main(data, wmask=None, mask_sky=False, out=None, log=None, silent=True):
    """Mask specific wavelength ranges in a cube."""

    #Set global parameters
    cwitools.silent_mode = silent
    cwitools.log_file = log

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
        for tup in args.masks.split('-'):
            wav0, wav1 = tuple(int(x) for x in tup.split(":"))
            zmask[(wav_axis >= wav0) & (wav_axis <= wav1)] = 1

        data_fits[0].data[zmask] = 0

    if out is None:
        out = data.replace('.fits', '.zmask.fits')

    data_fits.writeto(out, overwrite=True)

    utils.output("\tSaved %s\n" % out)


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
