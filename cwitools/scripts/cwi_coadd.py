"""Stack input cubes into a master frame using a CWITools parameter file."""
from astropy.io import fits
from cwitools import reduction, utils

import argparse
import os
import sys
import time

def main():

    #Timer start
    tstart = time.time()

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Coadd data cubes.')

    parser.add_argument('list',
                        type=str,
                        help='Comma-separated list of FITS files or CWITools .list file.'
    )

    methodGroup = parser.add_argument_group(title="Methods",description="Parameters related to coadd methods.")
    methodGroup.add_argument('-ctype',
                        metavar="<cube_type>",
                        type=str,
                        help='The type of cube (i.e. file extension such as \'icubed.fits\') to coadd'
    )
    methodGroup.add_argument('-pxthresh',
                        metavar="0-1",
                        type=float,
                        help='Fraction of a coadd-frame pixel that must be covered by an input frame to be included (0-1)',
                        default=0.5
    )
    methodGroup.add_argument('-expthresh',
                        metavar="0-1",
                        type=float,
                        help='Crop cube to include only spaxels with this fraction of the maximum overlap (0-1)',
                        default=0.75
    )
    methodGroup.add_argument('-pa',
                        metavar="<dd.dd>",
                        type=float,
                        help='Position Angle of output frame.',
                        default=0
    )
    fileIOGroup = parser.add_argument_group(title="Input/Output",description="File input/output options.")
    fileIOGroup.add_argument('-vardata',
                        help='Set flag if coadding variance data.',
                        action='store_true'
    )
    fileIOGroup.add_argument('-out',
                        metavar="<file_out>",
                        type=str,
                        help='Output file name.',
                        default=None
    )
    fileIOGroup.add_argument('-v',help="Show progress and file names.",action='store_true')
    fileIOGroup.add_argument('-log',
                        metavar="<log_file>",
                        type=str,
                        help="Log file to save this command in",
                        default=None
    )
    args = parser.parse_args()

    if args.out != None:
        outfile = args.out

    if "," in args.list:
        cubes = args.list.split(',')
        file_list = []
        for x in cubes:
            if os.path.isfile(x):
                file_list.append(x)
            else:
                raise FileNotFoundError(x)

        #Get output name from first file
        if args.out == None:
            outfile = file_list[0].replace(".fits", "_coadd.fits")

    elif os.path.isfile(args.list) and args.ctype != None:

        clist = utils.parse_cubelist(args.list)
        file_list = utils.find_files(
            clist["ID_LIST"],
            clist["INPUT_DIRECTORY"],
            args.ctype,
            depth=clist["SEARCH_DEPTH"]
        )

        #Get output name from list name
        if args.out==None:
            name = args.list.split(".")[0]
            outfile = '%s%s_%s' % (clist["OUTPUT_DIRECTORY"], name, args.ctype)

    else:
        raise SyntaxError("Syntax should be either a comma-separated list of\
        files to coadd or a CWITools .list file along with -ctype.")


    fits_list = [fits.open(x) for x in file_list]

    if len(fits_list) == 0:
        raise RuntimeError("No files matching search found. Check usage and try again.")

    #Coadd the fits files
    coadd_fits = reduction.coadd(fits_list,
                      pxthresh=args.pxthresh,
                      expthresh=args.expthresh,
                      pa=args.pa,
                      vardata = args.vardata,
                      verbose=args.v
    )

    #Save stacked cube

    coadd_fits.writeto(outfile,overwrite=True)

    #Timer end
    tfinish = time.time()
    print("\nSaved %s" % outfile)
    print("Elapsed time: %.2f seconds" % (tfinish-tstart))

    utils.log_command(sys.argv, logfile=args.log)


if __name__=="__main__": main()
