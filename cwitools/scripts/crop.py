"""Crop a data cube along spatial and/or wavelength axes."""

#Standard Imports
from datetime import datetime
import argparse
import os
import sys

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import reduction, utils
import cwitools

def parser_init():
    """Create command-line argument parser for this script."""
    #Handle input with argparse
    parser = argparse.ArgumentParser(description="""
    Crop axes of a single data cube or multiple data cubes. There are two usage\
 options: (1) Run directly on a single cube (e.g. cwi_crop -cube mycube.fits\
 -wcrop 4100,4200 -xcrop 10,60 ) and (2) run using a CWITools cube list,\
 loading all input cubes of a certaintype (e.g. cwi_crop -list mytarget.list\
  -cube icubes.fits ...).\n Multiple cube types can be specified when using\
 the latter format, just separate them with spaces.
    """)
    parser.add_argument(
        'clist',
        type=str,
        nargs='+',
        help='Input cubes or a CWITools list of input cubes.',
        default=None
        )
    parser.add_argument(
        '-ctype',
        metavar="<cube_type>",
        type=str,
        nargs='+',
        help='Extension(s) for each cube. Only applied when input is a list.',
        default=None
        )
    parser.add_argument(
        '-wcrop',
        metavar="<w0 w1>",
        type=float,
        nargs='+',
        help="Wavelength range [A] to crop to. Use -1 -1 automatic cropping.",
        default=None
        )
    parser.add_argument(
        '-xcrop',
        metavar="<x0 x1>",
        type=int,
        nargs='+',
        help="X-axis range [px] to crop to. Use -1 -1 for automatic cropping.",
        default=None
        )
    parser.add_argument(
        '-ycrop',
        metavar="<y0 y1>",
        type=int,
        nargs='+',
        help="Y-axis range [px] to crop to. Use -1 -1 for automatic cropping.",
        default=None
        )
    parser.add_argument(
        '-trim_mode',
        metavar="<zero/edge>",
        type=str,
        help="Trim mode to use when cropping x/y with 'auto'. 'zero' trims empty rows/columns. 'edge' detects and removes edge features.",
        default='zero'
        )
    parser.add_argument(
        '-trim_sclip',
        metavar="<sigma_clip>",
        type=float,
        help="Sigma-clipping threshold to use on slices if using 'auto' with -trim_mode 'edge'",
        default=3
        )
    parser.add_argument(
        '-auto_pad',
        metavar="<1 (1)>",
        type=float,
        nargs='+',
        help="Additional margin (px) on axes 1 and 2 to add to automatically determined crop parameters. Seperated by space. Default 0.",
        default=0
        )
    parser.add_argument(
        '-ext',
        metavar='<file_ext>',
        type=str,
        help='The filename extension to add to cropped cubes. Default: .c.fits',
        default=".c.fits"
        )
    parser.add_argument(
        '-plot',
        help="Show automatically determined plot parameters, if using 'auto' for any.",
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


def main(clist, ctype=None, wcrop=None, xcrop=None, ycrop=None, trim_mode=None,
         trim_sclip=None, auto_pad=None, ext=None, plot=None, log=None, silent=None,
         arg_parser=None):

    if arg_parser is not None:
        args = arg_parser.parse_args()
        clist = args.clist
        ctype = args.ctype
        wcrop = args.wcrop
        xcrop = args.xcrop
        ycrop = args.ycrop
        trim_mode = args.trim_mode
        trim_sclip = args.trim_sclip
        auto_pad = args.auto_pad
        ext = args.ext
        plot = args.plot
        log = args.log
        silent = args.silent

        #Give output summarizing mode
        cmd = utils.get_cmd(sys.argv)
        infostring = utils.get_arg_string(args)

    else:
        infostring = ""

    #Set global parameters
    cwitools.silent_mode = silent
    cwitools.log_file = log

    titlestring = """\n{0}\n{1}\n\tCWI_CROP:""".format(datetime.now(), cmd)
    utils.output(titlestring + infostring)

    #Make sure clist type is 'list' before next part
    if isinstance(clist, str):
        clist = [clist]
    elif not isinstance(clist, list):
        raise ValueError("clist must be a string or list of strings.")

    #If ctype is given as a string, also change to list of strings
    if isinstance(ctype, str):
        ctype = [ctype]

    #If only one string given with ".list" extension, load from CWITools list
    if len(clist) == 1 and ".list" in clist[0]:

        if ctype is None:
            raise SyntaxError("""-ctype must be specified if using CWITools list.\n""")

        cdict = utils.parse_cubelist(clist[0])
        file_list = []
        for c_t in ctype:
            file_list += utils.find_files(
                cdict["ID_LIST"],
                cdict["INPUT_DIRECTORY"],
                c_t,
                cdict["SEARCH_DEPTH"]
            )

    #Otherwise, input must be a list of FITS files
    else:

        #Validate that the input is a list of FITS files
        for file_name in clist:
            if ".fits" not in file_name or not os.path.isfile(file_name):
                raise SyntaxError("clist must be either a single FITS file path,\
                a list of FITS file paths, or a CWITools .list file path")

        file_list = clist

    if not isinstance(auto_pad, list):
        auto_pad = int(auto_pad)
    else:
        auto_pad = tuple(int(x) for x in auto_pad)

    #Flag if any crop param has been given as automatic (-1 -1)
    auto_flags = [wcrop == [-1, -1], ycrop == [-1, -1], xcrop == [-1, -1]]

    # Open fits objects
    for file_name in file_list:

        fits_file = fits.open(file_name)

        #Default to input values
        wcrop_i = wcrop
        ycrop_i = ycrop
        xcrop_i = xcrop

        #If any auto-crop params requested, obtain these and then assign
        if any(auto_flags):

            wcrop_auto, ycrop_auto, xcrop_auto = reduction.get_crop_params(
                fits_file,
                zero_only=(trim_mode == 'zero'),
                pad=auto_pad,
                nsig=trim_sclip,
                plot=plot
            )

            #Assign auto parameters where requested
            if auto_flags[0]:
                wcrop_i = wcrop_auto

            if auto_flags[1]:
                ycrop_i = ycrop_auto

            if auto_flags[2]:
                xcrop_i = xcrop_auto

        # Pass to trimming function
        cropped_fits = reduction.crop(
            fits_file,
            xcrop=xcrop_i,
            ycrop=ycrop_i,
            wcrop=wcrop_i
        )

        out_file = os.path.basename(file_name).replace('.fits', args.ext)
        cropped_fits.writeto(out_file, overwrite=True)
        utils.output("\tSaved %s\n" % out_file)

    utils.output("\n")


if __name__ == "__main__":
    main("", arg_parser=parser_init())
