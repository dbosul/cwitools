"""Stack input cubes into a master frame using a CWITools parameter file."""
from astropy.io import fits
from cwitools import reduction, utils
from datetime import datetime

import argparse
import cwitools
import os
import sys
import time

def main():

    #Timer start
    tstart = time.time()

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Coadd data cubes.')
    parser.add_argument('cube_list',
                        type=str,
                        help='Comma-separated list of FITS files or CWITools .list file.'
    )
    parser.add_argument('-ctype',
                        metavar="<cube_type>",
                        type=str,
                        help='The main type of cube (e.g. icubes.fits) to coadd, if using .list file.'
    )
    parser.add_argument('-masks',
                        metavar="<cube_type>",
                        type=str,
                        help='Comma-separated list of 3D mask files or mask type (e.g. mcubes.fits) if using .list file'
    )
    parser.add_argument('-var',
                        metavar="<cube_type>",
                        type=str,
                        help='Comma-separated list of 3D variance files or mask type (e.g. vcubes.fits) if using .list file'
    )
    parser.add_argument('-pxthresh',
                        metavar="0-1",
                        type=float,
                        help='Fraction of a coadd-frame pixel that must be covered by an input frame to be included (0-1)',
                        default=0.5
    )
    parser.add_argument('-expthresh',
                        metavar="0-1",
                        type=float,
                        help='Crop cube to include only spaxels with this fraction of the maximum overlap (0-1)',
                        default=0.75
    )
    parser.add_argument('-drizzle',
                        metavar="<0-1>",
                        type=float,
                        help='Drizzle factor. Typical values are 0.7-0.9.',
                        default=1.0
    )
    parser.add_argument('-pa',
                        metavar="<dd.dd>",
                        type=float,
                        help='Position Angle of output frame.',
                        default=0
    )
    parser.add_argument('-out',
                        metavar="<file_out>",
                        type=str,
                        help='Output file name.',
                        default=None
    )
    parser.add_argument('-v',help="Show progress and file names.",action='store_true')
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
    titlestring = """\n{0}\n{1}\n\tCWI_COADD:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(parser)
    utils.output(titlestring + infostring)

    #Get output filename if given
    if args.out != None:
        outfile = args.out

    #Get list of cube names if given as cs-list
    if "," in args.cube_list:

        cubes = args.cube_list.split(',')
        cube_list = []

        for x in cubes:
            if os.path.isfile(x):
                cube_list.append(x)
            else:
                raise FileNotFoundError(x)

        #Get output name from first file
        if args.out == None:
            outfile = file_list[0].replace(".fits", "_coadd.fits")

    #Get CWITools list file if that is given
    elif os.path.isfile(args.cube_list) and args.ctype != None:

        cube_list = args.cube_list

        #Get output name from list name
        if args.out==None:
            name = args.cube_list.split(".")[0]
            outfile = '%s%s_%s' % (clist["OUTPUT_DIRECTORY"], name, args.ctype)

    else:
        raise SyntaxError("Syntax should be either a comma-separated list of\
        files to coadd or a CWITools .list file along with -ctype.")

    #Coadd the fits files
    coadd_result = reduction.coadd(
        cube_list,
        cube_type = args.ctype,
        masks_in = args.masks,
        var_in = args.var,
        px_thresh = args.pxthresh,
        exp_thresh = args.expthresh,
        pa = args.pa,
        verbose = args.v,
        drizzle = args.drizzle
    )

    if args.var is not None:
        coadd_fits, coadd_var_fits = coadd_result

        coadd_fits.writeto(outfile, overwrite = True)
        utils.output("\n\tSaved %s\n" % outfile)

        var_outfile = outfile.replace(".fits", ".var.fits")
        coadd_var_fits.writeto(var_outfile, overwrite = True)
        utils.output("\n\tSaved %s\n" % var_outfile)

    else:
        coadd_fits = coadd_result

        coadd_fits.writeto(outfile, overwrite = True)
        utils.output("\n\tSaved %s\n" % outfile)



    #Timer end
    tfinish = time.time()

    utils.output("\tElapsed time: %.2f seconds\n" % (tfinish-tstart))

if __name__=="__main__": main()
