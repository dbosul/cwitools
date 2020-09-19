"""Rebin a data cube along the XY or Z axes."""
#Standard Imports
import argparse
import os
import sys

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import utils, config, reduction

def parser_init():
    """Create command-line argument parser for this script."""
    #Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="""Rebin a data cube along the XY or Z axes."""
    )
    parser.add_argument(
        'cube',
        type=str,
        help='Input cube to be binned.'
    )
    parser.add_argument(
        '-xybin',
        type=int,
        help='Number of pixels to bin in X,Y axes',
        default=1
    )
    parser.add_argument(
        '-zbin',
        type=int,
        help='Number of pixels to bin in Z axis.',
        default=1
    )
    parser.add_argument(
        '-ext',
        type=str,
        help='File extension to add for binned cube (Default: .binned.fits)',
        default=".binned.fits"
    )
    parser.add_argument(
        '-vardata',
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

def rebin(cube, xybin=1, zbin=1, vardata=False, ext=".binned.fits", log=None,
          silent=None):
    """Rebin a data cube along the XY or Z axes.

    Args:
        cube (str): Path to input data cube FITS file
        xybin (int): Bin-size for spatial axes
        zbin (int) Bin-size for wavelength axis
        vardata (bool): Set to TRUE if rebinning variance data
        ext (str): File extension for output file
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """
    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("REBIN", locals())

    #Load data
    if os.path.isfile(cube):
        data_fits = fits.open(cube)
    else:
        raise FileNotFoundError("Input file not found.\nFile:%s" % cube)

    #Check that user has actually set the bin options
    if zbin == 1 and xybin == 1:
        utils.output("Binning 1x1x1 won't change anything!\nExiting.")
        sys.exit()

    binned_fits = reduction.cubes.rebin(
        data_fits,
        bin_xy=xybin,
        bin_z=zbin,
        vardata=vardata
    )

    outfilename = cube.replace(".fits", ext)

    binned_fits.writeto(outfilename, overwrite=True)
    utils.output("\tSaved %s\n" % outfilename)
    config.set_temp_output_mode(log, silent)


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    rebin(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
