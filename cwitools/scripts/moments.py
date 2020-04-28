"""Create 2D maps of velocity and dispersion."""
from astropy.io import fits
from cwitools import extraction, reduction, utils, synthesis
from datetime import datetime

import argparse
import cwitools
import numpy as np
import os
import sys

def main():
    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Make maps of the first and second velocity moments of a 3D object.')
    parser.add_argument('cube',
                        type=str,
                        metavar='cube',
                        help='The input data cube.'
    )
    parser.add_argument('obj',
                        type=str,
                        metavar='path',
                        help='Object Mask cube.',
    )
    parser.add_argument('-id',
                        type=str,
                        metavar='str',
                        help='The ID of the object to use. Use -1 for all objects. Can also provide multiple as comma-separated list.',
                        default='1'
    )
    parser.add_argument('-var',
                        type=str,
                        metavar='path',
                        help='Variance cube, to apply inverse variance weighting.',
    )
    parser.add_argument('-rsmooth',
                        type=float,
                        help='Smooth spatial axes before calculating moments (FWHM).',
                        default=None
    )
    parser.add_argument('-wsmooth',
                        type=float,
                        help='Smooth wavelength axis before calculating moments (FWHM).',
                        default=None
    )
    parser.add_argument('-unit',
                        type=str,
                        help="Output mode for units of moment maps.",
                        choices=['kms','wav'],
                        default='wav'
    )
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

    #Get command that was issues
    argv_string = " ".join(sys.argv)
    cmd_string = "python " + argv_string + "\n"

    #Give output summarizing mode
    timestamp = datetime.now()
    infostring = """\n{0}\n{1}\n\tCWI_MOMENTS:\n
\t\tCUBE = {2}
\t\tOBJ = {3}
\t\tID = {4}
\t\tVAR = {5}
\t\tRSMOOTH = {6}
\t\tWSMOOTH = {7}
\t\tUNIT = {8}
\t\tLOG = {9}
\t\tSILENT = {10}\n\n""".format(timestamp, cmd_string, args.cube, args.obj,
args.id, args.var, args.rsmooth, args.wsmooth, args.unit, args.log, args.silent)
    utils.output(infostring)

    #Try to load the fits file
    if os.path.isfile(args.cube):
        fits_in = fits.open(args.cube)
    else: raise FileNotFoundError(args.cube)

    #Try to load the fits file
    if args.var!=None:
        if os.path.isfile(args.var):
            var_fits = fits.open(args.var)
            var_cube = var_fits[0].data
            var_cube[var_cube <= 0] = np.inf
        else: raise FileNotFoundError(args.var)
    else:
        utils.output("\tNo variance input given. Variance will be estimated.")
        var_cube = reduction.estimate_variance(fits_in)

    if args.rsmooth!=None:
        cube = extraction.smooth_nd(cube, args.rsmooth, axes=(1,2))
        var_cube = extraction.smooth_nd(cube, args.rsmooth, axes=(1,2), var=True)
        reduction.rescale_var(var_cube, cube)

    if args.wsmooth!=None:
        cube = extraction.smooth_nd(cube, args.wsmooth, axes=[0])
        var_cube = extraction.smooth_nd(cube,args.wsmooth, axes=[0], var=True)
        reduction.rescale_var(var_cube, cube)

    #Load object mask
    if os.path.isfile(args.obj): obj_cube = fits.getdata(args.obj)
    else: raise FileNotFoundError(args.obj)

    #Parse object IDs
    try: obj_id = list( int(x) for x in args.id.split(',') )
    except: raise ValueError("Could not parse -objid flag. Should be comma-separated list of object IDs.")


    m1_fits, m1err_fits, m2_fits, m2err_fits = synthesis.obj_moments(
        fits_in,
        obj_cube,
        obj_id,
        var_cube=var_cube,
        unit=args.unit
    )

    m1_out_ext = ".m1.fits"
    m2_out_ext = ".m2.fits"


    m1_out = args.cube.replace('.fits', '.m1.fits')
    m1_fits.writeto(m1_out,overwrite=True)
    utils.output("\tSaved %s" % m1_out)

    m1err_out = args.cube.replace('.fits', '.m1_err.fits')
    m1err_fits.writeto(m1err_out,overwrite=True)
    utils.output("\tSaved %s" % m1err_out)

    m2_out = args.cube.replace('.fits', '.m2.fits')
    m2_fits[0].header["BUNIT"] = "km/s"
    m2_fits.writeto(m2_out, overwrite=True)
    utils.output("\tSaved %s" % m2_out)

    m2err_out = args.cube.replace('.fits', '.m2_err.fits')
    m2err_fits.writeto(m2err_out,overwrite=True)
    utils.output("\tSaved %s" % m2err_out)



if __name__=="__main__":
    main()
