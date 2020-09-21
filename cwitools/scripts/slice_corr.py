"""Perform slice-to-slice correction on an input cube."""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import reduction, utils, config

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Perform slice-to-slice correction on an input cube."""
    )
    parser.add_argument(
        'clist',
        type=str,
        help='The input id list.'
    )
    parser.add_argument(
        'ctype',
        type=str,
        help='The input cube type.'
    )
    parser.add_argument(
        '-mask_reg',
        type=str,
        help='DS9 region of areas to exclude when slice-correcting.',
        default=None
    )
    parser.add_argument(
        '-sigclip',
        type=float,
        help='Sigma clip value to apply to each 1D profile before taking median.'
    )
    parser.add_argument(
        '-ext',
        type=str,
        help='The filename extension to add to modified cubes. Default: .f.fits',
        default=".sc.fits"
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


def slice_corr(clist, ctype, mask_reg=None, ext=None, log=None, silent=None):
    """Perform slice-to-slice correction on an input cube.

    Args:
        clist (str): Path to a CWITools .list file
        ctype (str): Type of CWI data cube to work with (e.g. 'icubes.fits')
        mask_reg (str): Path to a DS9 region file to use to exclude regions
            when measuring slice backgrounds.
        ext (str): File extension for output file
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("SLICE_CORR", locals())

    #Load files
    cdict = utils.parse_cubelist(clist)
    file_list = utils.find_files(
        cdict["ID_LIST"],
        cdict["DATA_DIRECTORY"],
        ctype,
        cdict["SEARCH_DEPTH"]
    )

    for file_in in file_list:
        fits_in = fits.open(file_in)

        fits_corrected = reduction.cubes.slice_corr(
            fits_in,
            mask_reg=mask_reg
        )

        out_file = file_in.replace('.fits', ext)
        fits_corrected.writeto(out_file, overwrite=True)
        utils.output("\tSaved %s\n" % out_file)

    config.restore_output_mode()


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    slice_corr(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
