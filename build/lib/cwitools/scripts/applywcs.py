"""Apply WCS corrections to data cubes"""
from astropy.io import fits
from cwitools import utils
from datetime import datetime

import argparse
import cwitools
import numpy as np
import os
import warnings
import sys

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Apply a WCS corrections file to data.')
    parser.add_argument('wcs_table',
                        type=str,
                        help='WCS correction file (see cwi_measurewcs.py)',
    )
    parser.add_argument('ctype',
                        metavar="cube_type(s)",
                        type=str,
                        help='Type(s) of file to apply to. Use comma to separate multiple values.',
    )
    parser.add_argument('-ext',
                        metavar="<file_ext>",
                        type=str,
                        help='File extension for corrected files (Def: .wc.fits)',
                        default=".wc.fits"
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

<<<<<<< HEAD
    #Get command that was issues
    argv_string = " ".join(sys.argv)
    cmd_string = "python " + argv_string + "\n"

    #Give output summarizing mode
    timestamp = datetime.now()

    infostring = """\n{0}\n{1}\n\tCWI_APPLYWCS:\n
\t\tWCS_TABLE = {2}
\t\tCTYPE = {3}
\t\tEXT = {4}
\t\tLOG = {5}
\t\tSILENT = {6}\n\n""".format(timestamp, cmd_string, args.wcs_table, args.ctype,
    args.ext, args.log, args.silent)

    utils.output(infostring)

    try:
        wcs_tab = open(args.wcs_table)

=======
    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_APPLYWCS:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(parser)
    utils.output(titlestring + infostring)

    try:
        wcs_tab = open(args.wcs_table)
>>>>>>> v0.6_dev2
    except FileNotFoundError:
        utils.output("\tCould not find WCS correction file: %s\n" % args.wcs_table)
        exit()

<<<<<<< HEAD

=======
>>>>>>> v0.6_dev2
    ids = []
    cr_matrix = []
    in_dir = "."
    search_depth = 3

    for i, line in enumerate(wcs_tab):

        line = line.replace("\n", "")

        if "INPUT_DIRECTORY" in line:
            in_dir = line.split("=")[1].replace(" ", "")

        elif "SEARCH_DEPTH" in line:
            search_depth = int(line.split("=")[1])

        elif line[0] == ">":
            vals = line[1:].split()
            ids.append(vals[0])
            cr_cols = [float(x) for x in vals[1:]]
            cr_matrix.append(cr_cols)

        else:
            continue

    cr_matrix = np.array(cr_matrix)

    ctypes = args.ctype.split(',')

    utils.output("\n\tCorrecting WCS Axes based on %s\n" % args.wcs_table)
    utils.output("\n\t%40s %10s %10s %10s\n" % ("Filename", "Ax1Cor?", "Ax2Cor?", "Ax3Cor?"))

    for ctype in ctypes:

        input_files = utils.find_files(ids, in_dir, ctype, depth=search_depth)

        for i, filename in enumerate(input_files):

            in_fits = fits.open(filename)
            ax1, ax2, ax3 = "No", "No", "No"

            if 0 <= cr_matrix[i, 0] <= 360:
                in_fits[0].header["CRVAL1"] = cr_matrix[i, 0]
                in_fits[0].header["CRPIX1"] = cr_matrix[i, 3]
                ax1 = "Yes"

            else:
                warnings.warn("Invalid RA / CRVAL1. Must be 0-360 deg.")

            if -90 <= cr_matrix[i, 1] <= 90:
                in_fits[0].header["CRVAL2"] = cr_matrix[i, 1]
                in_fits[0].header["CRPIX2"] = cr_matrix[i, 4]
                ax2 = "Yes"

            else:
                warnings.warn("Invalid DEC / CRVAL2. Must be -90 to +90 deg.")

            if cr_matrix[i, 2] > 0:
                in_fits[0].header["CRVAL3"] = cr_matrix[i, 2]
                in_fits[0].header["CRPIX3"] = cr_matrix[i, 5]
                ax3 = "Yes"

            outfilename = filename.replace(".fits", args.ext)
            in_fits.writeto(outfilename, overwrite=True)
            outfilename_short = outfilename.split("/")[-1]
            utils.output("\t%40s %10s %10s %10s\n" % (outfilename_short, ax1, ax2, ax3))

if __name__=="__main__":
    main()
