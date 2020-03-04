"""Stack input cubes into a master frame using a CWITools parameter file."""
from astropy.io import fits
from cwitools import parameters, reduction

import argparse
import os
import time

def main():

    #Timer start
    tStart = time.time()

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Coadd data cubes.')

    modeGroup = parser.add_mutually_exclusive_group(required=True)
    modeGroup.add_argument('-cubelist',
                        type=str,
                        help='A comma-separated list of cubes.'
    )
    modeGroup.add_argument('-param',
                        type=str,
                        help='A CWITools Parameter file.'
    )

    methodGroup = parser.add_argument_group(title="Methods",description="Parameters related to coadd methods.")
    methodGroup.add_argument('-cubetype',
                        type=str,
                        help='The type of cube (i.e. file extension such as \'icubed.fits\') to coadd'
    )
    methodGroup.add_argument('-pxthresh',
                        type=float,
                        help='Fraction of a coadd-frame pixel that must be covered by an input frame to be included (0-1)',
                        default=0.5
    )
    methodGroup.add_argument('-expthresh',
                        type=float,
                        help='Crop cube to include only spaxels with this fraction of the maximum overlap (0-1)',
                        default=0.75
    )
    methodGroup.add_argument('-pa',
                        type=float,
                        help='Position Angle of output frame.',
                        default=0
    )
    fileIOGroup = parser.add_argument_group(title="Input/Output",description="File input/output options.")
    fileIOGroup.add_argument('-vardata',
                        type=str,
                        help='Set to TRUE if coadding variance data.',
                        choices=["True","False"],
                        default="False"
    )
    fileIOGroup.add_argument('-out',
                        type=str,
                        help='Output file name.',
                        default=None
    )
    fileIOGroup.add_argument('-v',help="Show progress and file names.",action='store_true')

    args = parser.parse_args()

    args.vardata = (args.vardata.upper()=="TRUE")

    if args.cubelist==None and args.cubetype==None:
        raise RuntimeError("Must provide -cubetype if using -param.")

    #Make list out of single cube if working in that mode
    if args.cubelist!=None and args.param==None and args.cubetype==None:
        fileList = []
        cubes = args.cubelist.split(',')
        for cubePath in cubes:
            if os.path.isfile(cubePath): fileList.append(cubePath)
            else:
                raise FileNotFoundError(cubePath)

        if args.out==None:
            outFileName = fileList[0].replace('.fits','_coadd.fits')
        else:
            outFileName = args.out

    elif args.cubelist==None and args.param!=None and args.cubetype!=None:

        # Check if any parameter values are missing (set to set-up mode if so)
        if os.path.isfile(args.param): params = parameters.load_params(args.param)
        else:
            raise FileNotFoundError(args.param)

        # Get filenames
        fileList = parameters.find_files(
            params["ID_LIST"],
            params["INPUT_DIRECTORY"],
            args.cubetype,
            depth=params["SEARCH_DEPTH"]
        )


        #Make output filename
        if args.out==None:
            outFileName = '%s%s_%s' % (params["OUTPUT_DIRECTORY"],params["TARGET_NAME"],args.cubetype)
        else:
            outFileName = args.out

    #Make sure usage is understood if some odd mix
    else:
        raise SyntaxError("""\n\nUsage should be one of the following modes:\
        \n\nUse -cubelist flag to manually specify a list of cubes to coadd. Must be comma-separated.\
        \nOR\
        \nUse -params AND -cubetype flag together to load/coadd cubes with a parameter file.
        """)

    fitsList = [fits.open(x) for x in fileList]

    #Coadd the fits files
    stackedFITS = reduction.coadd(fitsList,
                      pxthresh=args.pxthresh,
                      expthresh=args.expthresh,
                      pa=args.pa,
                      vardata = args.vardata,
                      verbose=args.v
    )

    #Save stacked cube

    stackedFITS.writeto(outFileName,overwrite=True)

    #Timer end
    tFinish = time.time()
    print("\nSaved %s" % outFileName)
    print("Elapsed time: %.2f seconds" % (tFinish-tStart))


if __name__=="__main__": main()
