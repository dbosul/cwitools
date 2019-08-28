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
    mainGroup.add_argument('paramPath',
                        type=str,
                        metavar='Parameter File',
                        help='Path CWITools Parameter file.'
    )
    mainGroup.add_argument('cubeType',
                        type=str,
                        metavar='Cube Type',
                        help='The type of cube (i.e. file extension such as \'icubed.fits\') to coadd'
    )

    methodGroup = parser.add_argument_group(title="Methods",description="Parameters related to coadd methods.")
    methodGroup.add_argument('-pxThresh',
                        type=float,
                        metavar='Pixel Threshold',
                        help='Fraction of a coadd-frame pixel that must be covered by an input frame to be included (0-1)',
                        default=0.5
    )
    methodGroup.add_argument('-expThresh',
                        type=float,
                        metavar='Exposure Threshold',
                        help='Crop cube to include only spaxels with this fraction of the maximum overlap (0-1)',
                        default=0.75
    )
    methodGroup.add_argument('-pa',
                        type=float,
                        metavar='float (deg)',
                        help='Position Angle of output frame.',
                        default=0
    )
    fileIOGroup = parser.add_argument_group(title="Input/Output",description="File input/output options.")
    fileIOGroup.add_argument('-varData',
                        type=str,
                        metavar='bool',
                        help='Set to TRUE if coadding variance data.',
                        choices=["True","False"],
                        default="False"
    )
    args = parser.parse_args()

    args.plot = (args.plot.upper()=="TRUE")
    args.varData = (args.varData.upper()=="TRUE")

    #Try to load the param file
    params = loadparams(args.paramPath)

    #Get filenames
    fileList = findfiles(params,args.cubeType)

    #Coadd the fits files
    stackedFITS = coadd(fileList,
                      pxThresh=args.pxThresh,
                      expThresh=args.expThresh,
                      pa=args.pa,
                      varData = args.varData
    )

    #Save stacked cube
    outFileName = '%s%s_%s' % (params["OUTPUT_DIRECTORY"],params["TARGET_NAME"],cubeType)
    stackedFITS.writeto(outFileName,overwrite=True)

    #Timer end
    tFinish = time.time()
    print("\nSaved %s" % outFileName)
    print("Elapsed time: %.2f seconds" % (tFinish-tStart))


if __name__=="__main__": main()
