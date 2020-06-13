"""Crop a data cube"""
from astropy.io import fits
from cwitools import reduction, utils
from datetime import datetime

import argparse
import cwitools
import os
import sys

def main():

    #Handle input with argparse
    parser = argparse.ArgumentParser(description="""
    Crop axes of a single data cube or multiple data cubes. There are two usage\
 options: (1) Run directly on a single cube (e.g. cwi_crop -cube mycube.fits\
 -wcrop 4100,4200 -xcrop 10,60 ) and (2) run using a CWITools cube list,\
 loading all input cubes of a certaintype (e.g. cwi_crop -list mytarget.list\
  -cube icubes.fits ...).\n Multiple cube types can be specified when using\
 the latter format, just separate them with commas (no spaces).
    """)
    parser.add_argument('cube',
                        type=str,
                        help='Individual cube or cube type(s) to be cropped.',
                        default=None
    )
    parser.add_argument('-list',
                        metavar="<cube_list>",
                        type=str,
                        help='CWITools parameter file (for working on a list of input cubes).',
                        default=None
    )
    parser.add_argument('-wcrop',
                        metavar="<w0:w1>",
                        type=str,
                        help="Wavelength range, in Angstrom, to crop to (syntax 'w0:w1') or 'auto' for automatic cropping.",
                        default='auto'
    )
    parser.add_argument('-xcrop',
                        metavar="<x0:x1>",
                        type=str,
                        help="Subrange of x-axis to crop to (syntax 'x0:x1') or 'auto' for automatic cropping. Default: auto",
                        default='auto'
    )
    parser.add_argument('-ycrop',
                        metavar="<y0:y1>",
                        type=str,
                        help="Subrange of y-axis to crop to (syntax 'y0:y1') or 'auto' for automatic cropping. Default: auto.",
                        default='auto'
    )
    parser.add_argument('-trim_mode',
                        metavar="<zero/edge>",
                        type=str,
                        help="Trim mode to use when cropping x/y with 'auto'. 'zero' trims empty rows/columns. 'edge' detects and removes edge features.",
                        default='zero'
    )
    parser.add_argument('-trim_sclip',
                        metavar="<sigma_clip>",
                        type=float,
                        help="Sigma-clipping threshold to use on slices if using 'auto' with -trim_mode 'edge'",
                        default=3
    )
    parser.add_argument('-auto_pad',
                        metavar="<1[,1]>",
                        type=str,
                        help="Additional margin (px) on axes 1 and 2 to add to automatically determined crop parameters. Can be an integer or comma-separated integer tuple. Default 0.",
                        default="0"
    )
    parser.add_argument('-ext',
                        metavar='<file_ext>',
                        type=str,
                        help='The filename extension to add to cropped cubes. Default: .c.fits',
                        default=".c.fits"
    )
    parser.add_argument('-plot',
                        help="Show automatically determined plot parameters, if using 'auto' for any.",
                        action='store_true'
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
    titlestring = """\n{0}\n{1}\n\tCWI_CROP:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    #Make list out of single cube if working in that mode
    if args.list != None:

        clist = utils.parse_cubelist(args.list)
        ctypes = args.cube.split(",")
        file_list = []
        for ctype in ctypes:
            file_list += utils.find_files(
                clist["ID_LIST"],
                clist["INPUT_DIRECTORY"],
                ctype,
                clist["SEARCH_DEPTH"]
            )

    elif args.list == None and os.path.isfile(args.cube):

        file_list = [args.cube]

    #Make sure usage is understood if some odd mix
    else:
        raise SyntaxError("""
        Usage should be one of the following modes:\n\
        \n\tGive an individual cube as the 'cube' argument
        OR\
        \n\tGive a comma-separated list of cube types (e.g. icubes.fits) and the -list argument
        """)

    #Assign auto parameters or parse user input as needed

    if args.xcrop.lower() != 'auto':
        try:
            xcrop = tuple(int(x) for x in args.xcrop.split(':'))
        except:
            raise ValueError("Could not parse -xcrop, should be colon-separated integer tuple.")

    if args.ycrop.lower() != 'auto':
        try:
            ycrop = tuple(int(y) for y in args.ycrop.split(':'))
        except:
            raise ValueError("Could not parse -ycrop, should be colon-separated integer tuple.")

    if args.wcrop.lower() != 'auto':
        try:
            wcrop = tuple(int(w) for w in args.wcrop.split(':'))
        except:
            raise ValueError("Could not parse -wcrop, should be colon-separated integer tuple.")

    try:
        if "," in args.auto_pad:
            auto_pad = tuple(int(x) for x in args.auto_pad.split(','))
        else:
            auto_pad = int(args.auto_pad)
    except:
        raise ValueError("auto_pad must be integer or comma-separated tuple of integers.")

    # Open fits objects
    for filename in file_list:

        fitsfile = fits.open(filename)

        #Calculate all automatic crop params if any requesed
        if 'auto' in [args.xcrop, args.ycrop, args.wcrop]:

            xcrop_auto, ycrop_auto, wcrop_auto = reduction.get_crop_params(fitsfile,
                zero_only = (args.trim_mode == 'zero'),
                pad = auto_pad,
                nsig = args.trim_sclip,
                plot = args.plot
            )

            #Assign auto parameters where requested
            if args.xcrop.lower() == 'auto':
                xcrop = xcrop_auto

            if args.ycrop.lower() == 'auto':
                ycrop = ycrop_auto

            if args.wcrop.lower() == 'auto':
                wcrop = wcrop_auto

        # Pass to trimming function
        trimmedFits = reduction.crop(fitsfile,
            xcrop = xcrop,
            ycrop = ycrop,
            wcrop = wcrop
        )

        outfile = filename.replace('.fits', args.ext)
        trimmedFits.writeto(outfile, overwrite=True)
        utils.output("\tSaved %s\n" % outfile)

    utils.output("\n")


if __name__=="__main__": main()
