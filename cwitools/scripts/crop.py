"""Crop a data cube along spatial and/or wavelength axes."""

#Standard Imports
import argparse
import os

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import utils, config, reduction

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
        '-outdir',
        metavar='<file_ext>',
        type=str,
        help='The directory to save cropped files to. Default is same directory as input data.'
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
        help="Show automatically determined plot parameters, if using 'auto'\
        for any.",
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

def crop(clist, ctype=None, wcrop=None, xcrop=None, ycrop=None, plot=None, ext=".c.fits",
         outdir=None, log=None, silent=None):
    """Crops a data cube (FITS) along spatial of wavelength axes.

    Args:
        clist (str): Input cubes specified in one of three ways:
            1) As a CWITools .list file
            2) As a file path to a single FITS file
            3) As a list of file paths
        ctype (str): The type(s) of cube to crop, specified as a string or list
            of strings (e.g. 'icubes.fits' or ['icubes.fits', 'vcubes.fits']).
            Only used in combination with a CWITools .list file.
        wcrop (int tuple): Wavelength range (Angstrom) to crop z-axis (axis 0)
            to. Use (-1, -1) to automatically trim to "WAVGOOD" range.
        ycrop (int tuple): Range to crop y-axis (axis 1) to. Use (-1, -1)  to
            trim empty rows. See get_crop_params for more complete method.
        xcrop (int tuple): Range to crop x-axis (axis 2) to. Use (-1, -1) to
            trim empty rows. See get_crop_params for more complete method.
        plot (bool): Set to True to show diagnostic plots.
        ext (str): File extension to use for masked FITS (".M.fits")
        outdir (str): Output directory for files. Default is the same directory as input.
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """
    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("CROP", locals())

    #Make sure clist type is 'list' before next part
    if isinstance(clist, str):
        clist = [clist]
    elif not isinstance(clist, list):
        raise ValueError("clist must be a string or list of strings.")

    #Make sure output directory exists before we start
    if outdir is not None:
        if not os.path.isdir(outdir):
            raise NotADirectoryError(outdir)
        outdir = os.path.abspath(outdir)

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
                cdict["DATA_DIRECTORY"],
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

            wcrop_auto, ycrop_auto, xcrop_auto = reduction.cubes.get_crop_params(
                fits_file,
                plot=plot
            )
            print(wcrop_auto, ycrop_auto, xcrop_auto)
            #Assign auto parameters where requested
            if auto_flags[0]:
                wcrop_i = wcrop_auto

            if auto_flags[1]:
                ycrop_i = ycrop_auto

            if auto_flags[2]:
                xcrop_i = xcrop_auto

        # Pass to trimming function
        cropped_fits = reduction.cubes.crop(
            fits_file,
            xcrop=xcrop_i,
            ycrop=ycrop_i,
            wcrop=wcrop_i
        )

        if outdir is None:
            out_file = file_name.replace('.fits', ext)
        else:
            outdir = os.path.abspath(outdir)
            out_file = outdir + '/' + os.path.basename(file_name).replace('.fits', ext)

        cropped_fits.writeto(out_file, overwrite=True)
        utils.output("\tSaved %s\n" % out_file)

    utils.output("\n")
    config.restore_output_mode()


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    crop(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
