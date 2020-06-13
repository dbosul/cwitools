"""Generate a pseudo-Narrowband image"""
from astropy.io import fits
from cwitools import coordinates, reduction, utils, synthesis
from datetime import datetime

import argparse
import cwitools
import numpy as np
import sys
import time

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Make channel maps of an input cube around a specified emission line.')
    parser.add_argument('cube',
                        type=str,
                        help='The input data cube.'
    )
    parser.add_argument('-wmask',
                        type=str,
                        metavar='Wav Mask',
                        help='Wavelength range(s) to mask when making WL image.',
                        default=None
    )
    parser.add_argument('-var',
                        type=str,
                        help='Variance cube, for calculating WL variance. Estimated if not given.',
                        default=None
    )
    parser.add_argument('-out',
                        help="Output file name. Default is parameter file + .WL.fits",
                        default=None
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

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_GETWL:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)


    #Load data
    infits = fits.open(args.cube)
    cube, hdr = infits[0].data, infits[0].header
    hdr2D = coordinates.get_header2d(hdr)

    #Try to parse the wavelength mask tuple
    if args.wmask != None:
        try:
            masks = []
            for pair in args.wmask.split('-'):
                w0,w1 = tuple(int(x) for x in pair.split(':'))
                masks.append((w0,w1))
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)
    else:
        masks = []

    if args.var!=None:
        varcube = fits.getdata(args.var)
    else:
        varcube = reduction.estimate_variance(infits)

    #utils.output("%s,"%args.cube.split('/')[-2], end='')
    WL, WL_var = synthesis.whitelight(infits, var_cube=varcube, wmask=masks)

    if args.out == None:
        outfilename = args.cube.replace('.fits', '.WL.fits')
    else:
        outfilename = args.out


    wl_fits = fits.HDUList([fits.PrimaryHDU(WL)])
    wl_fits[0].header = hdr2D
    wl_fits.writeto(outfilename, overwrite=True)
    utils.output("\tSaved %s\n" % outfilename)

    var_outfilename = outfilename.replace(".fits", ".var.fits")
    wl_var_fits = fits.HDUList([fits.PrimaryHDU(WL_var)])
    wl_var_fits[0].header = hdr2D
    wl_var_fits.writeto(var_outfilename, overwrite=True)
    utils.output("\tSaved %s\n" % var_outfilename)

if __name__=="__main__": main()
