from astropy.io import fits
from cwitools import utils, reduction
from datetime import datetime

import argparse
import cwitools
import numpy as np
import os
import sys


def main():

    #Handle input with argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('clist',
                        type=str,
                        help='The input id list.'
    )
    parser.add_argument('ctype',
                        type=str,
                        help='The input cube type.'
    )
    parser.add_argument('-sigclip',
                        type=float,
                        help='Sigma clip value to apply to each 1D profile before taking median.'
    )
    parser.add_argument('-ext',
                        type=str,
                        help='The filename extension to add to modified cubes. Default: .f.fits',
                        default=".sc.fits"
    )
    parser.add_argument('-log',
                        metavar="<log_file>",
                        type=str,
                        help="Log file to save output in.",
                        default=None
    )
    parser.add_argument('-silent',
                        help="Set flag to suppress standard terminal output.",
                        action='store_true'
    )
    args = parser.parse_args()

    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_SLICECORR:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    #Load files
    clist = utils.parse_cubelist(args.clist)
    file_list = utils.find_files(
        clist["ID_LIST"],
        clist["INPUT_DIRECTORY"],
        args.ctype,
        clist["SEARCH_DEPTH"]
    )

    for file_in in file_list:
        fits_in = fits.open(file_in)
        fits_corrected = reduction.slice_corr(fits_in)
        out_filename = file_in.replace('.fits', args.ext)
        fits_corrected.writeto(out_filename, overwrite=True)
        utils.output("\tSaved %s\n" % out_filename)

if __name__=="__main__": main()
