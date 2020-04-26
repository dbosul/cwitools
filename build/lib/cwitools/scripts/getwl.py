"""Generate a pseudo-Narrowband image"""
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from cwitools import coordinates, reduction, utils, synthesis
from scipy.stats import sigmaclip

import argparse
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyregion
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
                        type=str,
                        help="Log file to save this command in",
                        default=None
    )
    args = parser.parse_args()

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

    #print("%s,"%args.cube.split('/')[-2], end='')
    WL, WL_var = synthesis.get_wl(infits, var=varcube, wmasks=masks)

    if args.out == None:
        outfilename = args.cube.replace('.fits', '.WL.fits')
    else:
        outfilename = args.out


    wl_fits = fits.HDUList([fits.PrimaryHDU(WL)])
    wl_fits[0].header = hdr2D
    wl_fits.writeto(outfilename, overwrite=True)
    print("Saved %s" % outfilename)

    var_outfilename = outfilename.replace(".fits", ".var.fits")
    wl_var_fits = fits.HDUList([fits.PrimaryHDU(WL_var)])
    wl_var_fits[0].header = hdr2D
    wl_var_fits.writeto(var_outfilename, overwrite=True)
    print("Saved %s" % var_outfilename)

if __name__=="__main__": main()
