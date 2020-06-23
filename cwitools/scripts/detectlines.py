"""Automatically associate 3D objects with known emission lines."""
from astropy.io import fits
from cwitools import utils, coordinates, extraction
from datetime import datetime

import argparse
import cwitools
import numpy as np
import sys
import time

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Make channel maps of an input cube around a specified emission line.')
    parser.add_argument(
        'obj',
        type=str,
        help='The input data cube.'
    )
    parser.add_argument(
        '-lines',
        type=str,
        help='The emission lines (rest-frame) to search for, comma-separated.',
        default=None
    )
    parser.add_argument(
        '-z',
        type=float,
        help='The redshift of the emission. (Default:0)',
        default=0
    )
    parser.add_argument(
        '-dv',
        type=float,
        help='The velocity window around each line, in km/s (Default:500).',
        default=500
    )
    parser.add_argument(
        '-out',
        type=str,
        help='Name for output table with object candidates. (Default:detections.tab)',
        default="detections.tab"
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
    args = parser.parse_args()

    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_DETECTLINES:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    obj_fits = fits.open(args.obj)

    if args.lines is not None:
        try:
            lines = [float(x) for x in args.lines.split(',')]
        except:
            raise ValueError("-lines must be comma-separated list of float-like values.")
    else:
        lines = None

    candidates = extraction.detect_lines(obj_fits, lines, z=args.z, dv=args.dv)

    tab_string = ""
    if len(candidates.keys()) > 0:
        tab_string += "\n\t\t#{0:<9}, IDs".format("LINE")

    for key, item in candidates.items():
        tab_string += "\n\t\t{0:<10}".format(key)
        for id in item:
            tab_string += ",{0}".format(id)

    if tab_string == "":
        utils.output("\n\tNo objects associated with lines found.\n")
    else:

        tab_string += '\n'
        utils.output("\n\tObject associations found:\n")
        utils.output("{0}".format(tab_string))

        tab_string_basic = tab_string[1:].replace('\t', '')
        file_out = open(args.out, 'w')
        file_out.write(tab_string_basic)
        file_out.close()
        utils.output("\n\tResults saved to {0}\n".format(args.out))

if __name__=="__main__": main()
