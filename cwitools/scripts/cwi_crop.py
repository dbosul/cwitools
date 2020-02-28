from cwitools import parameters, reduction
from astropy.io import fits

import argparse
import os
def main():

    #Handle input with argparse
    parser = argparse.ArgumentParser(description="""
    Crop axes of a single data cube or multiple data cubes. There are two usage
    options. (1) Run directly on a single cube (e.g. cwi_crop -cube mycube.fits
    -wcrop 4100,4200 -xcrop 10,60 ) and (2) run using a CWITools parameter file,
    loading all input cubes of a certaintype (e.g. cwi_crop -params mytarget.param
    -cubetype icubes.fits -wcrop 4100,4200 -xcrop 10,60)
    """)
    parser.add_argument('-cube',
                        type=str,
                        help='Cube to be cropped (for working on a single cube).',
                        default=None
    )
    parser.add_argument('-params',
                        type=str,
                        help='CWITools parameter file (for working on a list of input cubes).',
                        default=None
    )
    parser.add_argument('-cubetype',
                        type=str,
                        help='The cube type to load (e.g. icubes.fits) if working with a parameter file.',
                        default=None

    )
    parser.add_argument('-wcrop',
                        type=str,
                        help="Wavelength range, in Angstrom, to crop to (syntax 'w0,w1') (Default:0,-1).",
                        default='0:-1'
    )
    parser.add_argument('-xcrop',
                        type=str,
                        help="Subrange of x-axis to crop to (syntax 'x0,x1') (Default:0,-1)",
                        default='0:-1'
    )
    parser.add_argument('-ycrop',
                        type=str,
                        help="Subrange of y-axis to crop to (syntax 'y0,y1') (Default:0,-1)",
                        default='0:-1'
    )
    parser.add_argument('-ext',
                        type=str,
                        help='The filename extension to add to cropped cubes. Default: .c.fits',
                        default=".c.fits"
    )
    parser.add_argument('-auto',
                        help="Automatically determine ALL crop settings. Overrides other parameters.",
                        action='store_true'
    )
    parser.add_argument('-plot',
                        help="Automatically determine ALL crop settings. Overrides other parameters.",
                        action='store_true'
    )

    args = parser.parse_args()

    #Make list out of single cube if working in that mode
    if args.cube!=None and args.params==None and args.cubetype==None:

        if os.path.isfile(args.cube): fileList = [args.cube]
        else:
            raise FileNotFoundError("Input file not found. \nFile:%s"%args.cube)

    #Load list from parameter files if working in that mode
    elif args.cube==None and args.params!=None and args.cubetype!=None:

        # Check if any parameter values are missing (set to set-up mode if so)
        if os.path.isfile(args.params): params = parameters.load_params(args.params)
        else:
            raise FileNotFoundError("Parameter file not found.\nFile:%s"%args.params)

        # Get filenames
        fileList = parameters.find_files(
            params["ID_LIST"],
            params["INPUT_DIRECTORY"],
            args.cubetype,
            depth=params["SEARCH_DEPTH"]
        )

    #Make sure usage is understood if some odd mix
    else:
        raise SyntaxError("""
        Usage should be one of the following modes:\
        \n\nUse -cube argument to specify one input cube to crop\
        \nOR\
        \nUse -params AND -cubetype flag together to load cubes from parameter file.
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
    for fileName in fileList:

        fitsFile = fits.open(fileName)

        # Pass to trimming function
        trimmedFits = reduction.crop(fitsFile,
            xcrop=(x0,x1),
            ycrop=(y0,y1),
            wcrop=(w0,w1),
            auto=args.auto,
            plot=args.plot
        )

        outFileName = fileName.replace('.fits',args.ext)
        trimmedFits.writeto(outFileName,overwrite=True)
        print("Saved %s"%outFileName)

if __name__=="__main__": main()
