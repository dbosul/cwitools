"""Perform slice-to-slice correction on an input cube."""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import reduction, utils
import cwitools

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


def main(clist, ctype, mask_reg=None, ext=None, log=None, silent=True):
    """Perform slice-to-slice correction on an input cube."""

    #Set global parameters
    cwitools.silent_mode = silent
    cwitools.log_file = log

    #Give output for log file
    utils.output_func_summary("SLICE_CORR", locals())

    #Load files
    cdict = utils.parse_cubelist(clist)
    file_list = utils.find_files(
        cdict["ID_LIST"],
        cdict["INPUT_DIRECTORY"],
        ctype,
        cdict["SEARCH_DEPTH"]
    )

    for file_in in file_list:
        fits_in = fits.open(file_in)

        fits_corrected = reduction.slice_corr(
            fits_in,
            mask_reg=mask_reg
        )

        out_file = file_in.replace('.fits', ext)
        fits_corrected.writeto(out_file, overwrite=True)
        utils.output("\tSaved %s\n" % out_file)

#Call using dict and argument parser if run from command-line
if __name__ == "__main__":

    arg_parser = parser_init()
    args = arg_parser.parse_args()

    main(**vars(args))
