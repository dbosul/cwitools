"""Stack input cubes into a master frame using a CWITools parameter file."""

from cwitools.params import loadparams,findfiles
from cwitools.reduction import coadd

import argparse
import time

def main():

    #Timer start
    tStart = time.time()

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Coadd data cubes.')

    mainGroup = parser.add_argument_group(title="Main",description="Basic input")
    mainGroup.add_argument('-params',
                        type=str,
                        help='Path CWITools Parameter file.'
    )
    mainGroup.add_argument('-cubetype',
                        type=str,
                        help='The type of cube (i.e. file extension such as \'icubed.fits\') to coadd'
    )
    mainGroup.add_argument('-cubelist',
                        type=str,
                        help='A comma-separated list of cubes to coadd (instead of using a parameter file.)'
    )
    methodGroup = parser.add_argument_group(title="Methods",description="Parameters related to coadd methods.")
    methodGroup.add_argument('-pxThresh',
                        type=float,
                        help='Fraction of a coadd-frame pixel that must be covered by an input frame to be included (0-1)',
                        default=0.5
    )
    methodGroup.add_argument('-expThresh',
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
    fileIOGroup.add_argument('-varData',
                        type=str,
                        help='Set to TRUE if coadding variance data.',
                        choices=["True","False"],
                        default="False"
    )
    fileIOGroup.add_argument('-o',
                        type=str,
                        help='Output file name.',
                        default=None
    )
    args = parser.parse_args()

    args.plot = (args.plot.upper()=="TRUE")
    args.varData = (args.varData.upper()=="TRUE")


    #Make list out of single cube if working in that mode
    if args.cubelist!=None and args.params==None and args.cubetype==None:
        fileList = []
        cubes = cubelist.split(',')
        for cubePath in cubes:
            if os.path.isfile(cubePath): fileList.append(cubePath)
            else:
                raise FileNotFoundError(cubePath)

        if args.o==None:
            outFileName = fileList[0].replace('.fits','_coadd.fits')
        else:
            outFileName = args.o

    elif args.cubelist==None and args.params!=None and args.cubetype!=None:

        # Check if any parameter values are missing (set to set-up mode if so)
        if os.path.isfile(ags.paramPath): params = params.loadparams(args.paramPath)
        else:
            raise FileNotFoundError(args.paramPath)

        # Get filenames
        fileList = params.findfiles(params,cubeType)

        #Make output filename
        if args.o==None:
            outFileName = '%s%s_%s' % (params["OUTPUT_DIRECTORY"],params["TARGET_NAME"],cubeType)
        else:
            outFileName = args.o

    #Make sure usage is understood if some odd mix
    else:
        raise SyntaxError("""
        Usage should be one of the following modes:\
        \n\nUse -cubelist flag to manually specify a list of cubes to coadd. Must be comma-separated.\
        \nOR\
        \nUse -params AND -cubetype flag together to load/coadd cubes with a parameter file.
        """)

    #Coadd the fits files
    stackedFITS = coadd(fileList,
                      pxThresh=args.pxThresh,
                      expThresh=args.expThresh,
                      pa=args.pa,
                      varData = args.varData
    )

    #Save stacked cube

    stackedFITS.writeto(outFileName,overwrite=True)

    #Timer end
    tFinish = time.time()
    print("\nSaved %s" % outFileName)
    print("Elapsed time: %.2f seconds" % (tFinish-tStart))


if __name__=="__main__": main()
