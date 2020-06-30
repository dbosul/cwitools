"""Crop a data cube"""
from astropy.io import fits
from cwitools import reduction, utils
from datetime import datetime

import argparse
import cwitools
import os
import sys

def parser_init():

    #Handle input with argparse
    parser = argparse.ArgumentParser(description="""
    Crop axes of a single data cube or multiple data cubes. There are two usage\
 options: (1) Run directly on a single cube (e.g. cwi_crop -cube mycube.fits\
 -wcrop 4100,4200 -xcrop 10,60 ) and (2) run using a CWITools cube list,\
 loading all input cubes of a certaintype (e.g. cwi_crop -list mytarget.list\
  -cube icubes.fits ...).\n Multiple cube types can be specified when using\
 the latter format, just separate them with commas (no spaces).
    """)
    parser.add_argument('file_in',
                        type=str,
                        nargs='+',
                        help='Input cubes or a CWITools list of input cubes.',
                        default=None
    )
    parser.add_argument('-cubetype',
                        metavar="<cube_list>",
                        type=str,
                        nargs='+',
                        help='Extension(s) for each cube. Only applied when input is a list.',
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
                        metavar="<1 (1)>",
                        type=float,
                        nargs='+',
                        help="Additional margin (px) on axes 1 and 2 to add to automatically determined crop parameters. Seperated by space. Default 0.",
                        default=0
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

    return parser


def core(args, parser):

    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_CROP:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    # Spliting difference use cases and creates a file_list with [0][:]
    #   being the sublist that shares the same cropping parameters.
    if len(args.file_in) == 1:
        if '.fits' in args.file_in:
            # Single cube
            file_list = [args.file_in]
        else:
            # list file
            ctypes = args.cubetype
            if ctypes is None:
                raise SyntaxError("""
                        -cubetype needs to be specified if using CWITools list.\n
                        """)
            if type(ctypes) == type(''):
                ctypes = [ctypes]

            clist = utils.parse_cubelist(args.file_in[0])
            file_list = []
            for idfile in clist['ID_LIST']:
                tmp_list=[]
                for ctype in ctypes:
                    tmp_list += utils.find_files(
                        [idfile],
                        clist["INPUT_DIRECTORY"],
                        ctype,
                        clist["SEARCH_DEPTH"]
                    )
                file_list.append(tmp_list)
                import pdb; pdb.set_trace()

    else:
        # multiple FITS files
        file_list = [args.file_in]

    import pdb; pdb.set_trace()
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

    if args.auto_pad != type([]):
        auto_pad = int(args.auto_pad)
    else:
        auto_pad = tuple(int(x) for x in args.auto_pad)

    # Open fits objects
    for sublist in file_list:

        # Setup the universal crop param for this sublist
        xcrop = args.xcrop
        ycrop = args.ycrop
        wcrop = args.wcrop

        for i,filename in enumerate(sublist):

            fitsfile = fits.open(filename)

            #Calculate all automatic crop params if any requesed
            if i == 0:
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

            outfile = os.path.basename(filename).replace('.fits', args.ext)
            trimmedFits.writeto(outfile, overwrite=True)
            utils.output("\tSaved %s\n" % outfile)

    utils.output("\n")


if __name__=="__main__":
    parser = parser_init()
    args = parser.parse_args()
    core(args, parser)


def main(file_in, cubetype=None, wcrop=None, xcrop=None, ycrop=None,
        trim_mode=None, trim_sclip=None, auto_pad=None,
        ext=None, plot=None, log=None, silent=None):

    parser = parser_init()

    # construct args
    if type(file_in)==type(''):
        str_list = [file_in]
    else:
        str_list = file_in
    if cubetype is not None:
        str_list.append('-cubetype')
        if type(cubetype)==type('') or len(cubetype)==1:
            str_list.append(str(cubetype))
        else:
            for i in cubetype:
                str_list.append(str(i))
    if wcrop is not None:
        str_list.append('-wcrop')
        str_list.append(str(wcrop[0]) + ':' + str(wcrop[1]))
    if xcrop is not None:
        str_list.append('-xcrop')
        str_list.append(str(xcrop[0]) + ':' + str(xcrop[1]))
    if ycrop is not None:
        str_list.append('-ycrop')
        str_list.append(str(ycrop[0]) + ':' + str(ycrop[1]))
    if trim_mode is not None:
        str_list.append('-trim_mode')
        str_list.append(str(trim_mode))
    if trim_sclip is not None:
        str_list.append('-trim_sclip')
        str_list.append(str(trim_sclip))
    if auto_pad is not None:
        str_list.append('-auto_pad')
        for i in auto_pad:
            str_list.append(str(i))
    if ext is not None:
        str_list.append('-ext')
        str_list.append(str(ext))
    if plot is not None:
        if plot:
            str_list.append('-plot')
    if log is not None:
        str_list.append('-log')
        str_list.append(str(log))
    if silent is not None:
        if silent:
            str_list.append('-silent')
    args = parser.parse_args(str_list)

    core(args, parser)
