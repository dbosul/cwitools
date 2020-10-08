"""Stack input cubes into a master frame."""

#Standard Imports
import argparse
import os
import time

#Local Imports
from cwitools import utils, config, reduction

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
        help='Comma-separated list of 3D mask files or mask type (e.g. mcubes.fits)\
        if using .list file'
        )
    parser.add_argument(
        '-var',
        metavar="<cube_type>",
        type=str,
        help='Comma-separated list of 3D variance files or mask type (e.g. vcubes.fits)\
        if using .list file'
        )
    parser.add_argument(
        '-px_thresh',
        metavar="0-1",
        type=float,
        help='Fraction of a coadd-frame pixel that must be covered by an input\
        frame to be included (0-1)',
        default=0.5
        )
    parser.add_argument(
        '-exp_thresh',
        metavar="0-1",
        type=float,
        help='Crop cube to include only spaxels with this fraction of the maximum\
        overlap (0-1)',
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

def coadd(clist, ctype=None, masks=None, var=None, px_thresh=0.5, exp_thresh=0.75,
          drizzle=1.0, pa=0, out=None, verbose=False, log=None, silent=None):
    """Coadd a list of 3D FITS cubes together.

    Args:
        clist (str or list): Input files to be added, specified in one of 3 ways:
            a) A path to a CWITools .list file (must also provide -cube_type)
            b) A Python list of file paths
            c) A Python list of HDU/HDUList objects
        ctype (str): The file type for the main coadd (e.g 'icubes.fits'),
            if using a .list file.
        masks (str): 3D PCWI/KCWI pipeline masks to  load and apply to data.
            Can be given in three ways:
              a) As a list of HDU-like objects (HDU or HDUList)
              b) As a list of file paths
              c) As a cube type (e.g. "mcubes.fits") - this option only works
                 if cube_list is given as a CWITools .list file.
        var (str): Specification of 3D PCWI/KCWI variance cubes to
            load and use for propagating error. Same rules apply as masks_in.
        px_thresh (float): Minimum fractional pixel overlap.
            This is the overlap between an input pixel and a pixel in the
            output frame. If a given pixel from an input frame covers less
            than this fraction of an output pixel, its contribution will be
            rejected.
        exp_thresh (float): Minimum exposure time, as fraction of maximum.
            If an area in the coadd has a stacked exposure time less than
            this fraction of the maximum overlapping exposure time, it will be
            trimmed from the coadd. Default: 0.1.
        drizzle (float): The drizzle factor to use, as a fraction of pixels size.
            E.g. 0.2 will shrink input pixels by 20%.
        pa (float): The desired position-angle of the output data.
        out (str): The output filename for the coadd.
        verbose (bool): Set to TRUE to display progress bar and extra info.
        log (str): The path to a log file to save output to (default: None)
        silent (bool): Set to FALSE to turn on standard terminal output.

    Returns:
        None
    """
    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("COADD", locals())

    #Timer start
    tstart = time.time()

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
            outfile = '%s_%s_coadd.fits' % (clist.split(".")[0], ctype.split(".")[0])

    else:
        raise SyntaxError("Syntax should be either a comma-separated list of\
        files to coadd or a CWITools .list file along with -ctype.")

    #Coadd the fits files
    coadd_result = reduction.cubes.coadd(
        clist,
        cube_type=ctype,
        masks_in=masks,
        var_in=var,
        px_thresh=px_thresh,
        exp_thresh=exp_thresh,
        pos_ang=pa,
        verbose=verbose,
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
    config.restore_output_mode()

#Entry-point method for setup-tools
def main():

    arg_parser = parser_init()
    args = arg_parser.parse_args()

    #If a ".list" file is given, extract the filename instead of passing a list
    if isinstance(args.clist, list) and len(args.clist) == 1 and ".list" in args.clist[0]:
        args.clist = args.clist[0]

    coadd(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
