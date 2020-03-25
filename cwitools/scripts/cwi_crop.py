"""Crop a data cube"""
from cwitools import reduction, utils
from astropy.io import fits
import argparse
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
                        help="Wavelength range, in Angstrom, to crop to (syntax 'w0,w1') (Default:0,-1).",
                        default='0:-1'
    )
    parser.add_argument('-xcrop',
                        metavar="<x0:x1>",
                        type=str,
                        help="Subrange of x-axis to crop to (syntax 'x0,x1') (Default:0,-1)",
                        default='0:-1'
    )
    parser.add_argument('-ycrop',
                        metavar="<y0:y1>",
                        type=str,
                        help="Subrange of y-axis to crop to (syntax 'y0,y1') (Default:0,-1)",
                        default='0:-1'
    )
    parser.add_argument('-ext',
                        metavar='<file_ext>',
                        type=str,
                        help='The filename extension to add to cropped cubes. Default: .c.fits',
                        default=".c.fits"
    )
    parser.add_argument('-auto',
                        help="Automatically determine ALL crop settings. Overrides other parameters.",
                        action='store_true'
    )
    parser.add_argument('-plot',
                        help="Show profiles of each axis with crop region overlaid.",
                        action='store_true'
    )
    parser.add_argument('-log',
                        metavar='<log_file>',
                        type=str,
                        help="Log file to save this command in",
                        default=None
    )
    args = parser.parse_args()



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


    try: x0, x1 = (int(x) for x in args.xcrop.split(':'))
    except:
        raise ValueError("Could not parse -xcrop, should be colon-separated integer tuple.")


    try: y0, y1 = (int(y) for y in args.ycrop.split(':'))
    except:
        raise ValueError("Could not parse -ycrop, should be colon-separated integer tuple.")

    try: w0, w1 = (int(w) for w in args.wcrop.split(':'))
    except:
        raise ValuError("Could not parse -wcrop, should be colon-separated integer tuple.")

    # Open fits objects
    for filename in file_list:

        fitsfile = fits.open(filename)

        # Pass to trimming function
        trimmedFits = reduction.crop(fitsfile,
            xcrop=(x0,x1),
            ycrop=(y0,y1),
            wcrop=(w0,w1),
            auto=args.auto,
            plot=args.plot
        )

        outfile = filename.replace('.fits', args.ext)
        trimmedFits.writeto(outfile, overwrite=True)
        print("Saved %s" % outfile)

    #Log after successful completion
    utils.log_command(sys.argv, logfile=args.log)

if __name__=="__main__": main()
