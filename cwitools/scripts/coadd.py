"""Stack input cubes into a master frame using a CWITools parameter file."""

#Standard Imports
from datetime import datetime
import argparse
import os
import sys
import time

#Local Imports
import cwitools
from cwitools import reduction, utils

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(description='Coadd data cubes.')
    parser.add_argument(
        'clist',
        nargs='+',
        type=str,
        help='CWITools .list file or space-separated list of FITS files'
        )
    parser.add_argument(
        '-ctype',
        metavar="<cube_type>",
        type=str,
        help='The main type of cube (e.g. icubes.fits) to coadd, if using .list file.'
        )
    parser.add_argument(
        '-masks',
        metavar="<cube_type>",
        type=str,
        help='Comma-separated list of 3D mask files or mask type (e.g. mcubes.fits) if using .list file'
        )
    parser.add_argument(
        '-var',
        metavar="<cube_type>",
        type=str,
        help='Comma-separated list of 3D variance files or mask type (e.g. vcubes.fits) if using .list file'
        )
    parser.add_argument(
        '-px_thresh',
        metavar="0-1",
        type=float,
        help='Fraction of a coadd-frame pixel that must be covered by an input frame to be included (0-1)',
        default=0.5
        )
    parser.add_argument(
        '-exp_thresh',
        metavar="0-1",
        type=float,
        help='Crop cube to include only spaxels with this fraction of the maximum overlap (0-1)',
        default=0.75
        )
    parser.add_argument(
        '-drizzle',
        metavar="<0-1>",
        type=float,
        help='Drizzle factor. Typical values are 0.7-0.9.',
        default=1.0
        )
    parser.add_argument(
        '-pa',
        metavar="<dd.dd>",
        type=float,
        help='Position Angle of output frame.',
        default=0
        )
    parser.add_argument(
        '-out',
        metavar="<file_out>",
        type=str,
        help='Output file name.',
        default=None
        )
    parser.add_argument(
        '-verbose',
        help="Show progress and file names.",
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

def main(clist, ctype=None, masks=None, var=None, px_thresh=0.5, exp_thresh=0.75,
         drizzle=1.0, pa=0, out=None, verbose=False, log=None, silent=True, arg_parser=None):
    """Coadd a list of 3D FITS cubes together."""

    if arg_parser is not None:
        args = arg_parser.parse_args()
        clist = args.clist
        ctype = args.ctype
        masks = args.masks
        var = args.var
        px_thresh = args.px_thresh
        exp_thresh = args.exp_thresh
        drizzle = args.drizzle
        pa = args.pa
        out = args.out
        pbar = args.pbar
        log = args.log
        silent = args.silent

        #Give output summarizing mode
        cmd = utils.get_cmd(sys.argv)
        titlestring = """\n{0}\n{1}\n\tCWI_COADD:""".format(datetime.now(), cmd)
        infostring = utils.get_arg_string(args)
        utils.output(titlestring + infostring)

    #Timer start
    tstart = time.time()

    #Set global parameters
    cwitools.silent_mode = silent
    cwitools.log_file = log

    #Get output filename if given
    if out is not None:
        outfile = out

    #Get list of cube names if given as cs-list
    if isinstance(clist, list) and len(clist) > 0:

        for c in clist:
            if not os.path.isfile(c):
                raise FileNotFoundError(c)

        #Get output name from first file
        if out is None:
            outfile = clist[0].replace(".fits", "_coadd.fits")

    #Get CWITools list file if that is given
    elif os.path.isfile(clist) and ctype is not None:

        #Get output name from list name
        if out is None:
            name = clist.split(".")[0]
            outfile = '%s%s_%s' % (clist["OUTPUT_DIRECTORY"], name, ctype)

    else:
        raise SyntaxError("Syntax should be either a comma-separated list of\
        files to coadd or a CWITools .list file along with -ctype.")

    #Coadd the fits files
    coadd_result = reduction.coadd(
        clist,
        cube_type=ctype,
        masks_in=masks,
        var_in=var,
        px_thresh=px_thresh,
        exp_thresh=exp_thresh,
        pa=pa,
        verbose=pbar,
        drizzle=drizzle
    )

    if var is not None:
        coadd_fits, coadd_var_fits = coadd_result

        coadd_fits.writeto(outfile, overwrite=True)
        utils.output("\n\tSaved %s\n" % outfile)

        var_outfile = outfile.replace(".fits", ".var.fits")
        coadd_var_fits.writeto(var_outfile, overwrite=True)
        utils.output("\n\tSaved %s\n" % var_outfile)

    else:
        coadd_fits = coadd_result

        coadd_fits.writeto(outfile, overwrite=True)
        utils.output("\n\tSaved %s\n" % outfile)

    #Timer end
    tfinish = time.time()
    utils.output("\tElapsed time: %.2f seconds\n" % (tfinish-tstart))

if __name__ == "__main__":
    main("", arg_parser=parser_init())
