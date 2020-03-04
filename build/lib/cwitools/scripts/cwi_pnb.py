from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from cwitools import parameters, coordinates, imaging, variance
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
    mainGroup = parser.add_argument_group(title="Main",description="Basic input")
    mainGroup.add_argument(
        'cube',
        type=str,
        metavar='cube',
        help='The input data cube.'
    )
    nbGroup = parser.add_argument_group(title="pseudo-NB Parameters")
    nbGroup.add_argument(
        '-wav',
        type=float,
        metavar='float',
        help='Central wavelength to use for pseudo NB. Default: None.',
        default=None
    )
    widthGroup = parser.add_mutually_exclusive_group(required=True)
    widthGroup.add_argument(
        '-dv',
        type=float,
        metavar='float',
        help='Pseudo-NB width in km/s.',
        default=None
    )
    widthGroup.add_argument(
        '-dw',
        type=float,
        metavar='float',
        help='Pseudo-NB width in Angstrom.',
        default=None
    )
    nbGroup.add_argument(
        '-var',
        type=str,
        metavar='var',
        help='Variance cube for SNR calculations. Estimated if not given.',
        default=None
    )
    nbGroup.add_argument(
        '-par',
        type=str,
        metavar='path',
        help='CWITools parameter file (used for target position).'
    )
    nbGroup.add_argument(
        '-pos',
        type=str,
        metavar='str',
        help='Position of your source as an \'x,y\' tuple.  Default: None',
        default=None

    )
    nbGroup.add_argument(
        '-cwidth',
        type=float,
        metavar='float',
        help='Bandwidth (in km/s) to use for continuum image.',
        default=2000
    )
    nbGroup.add_argument(
        '-fit_rad',
        type=float,
        metavar='float',
        help='Radius (px) around source to use for fitting WL image. Default: 3',
        default=3
    )
    nbGroup.add_argument(
        '-sub_rad',
        type=float,
        metavar='float',
        help='Radius (px) around source to subtract WL image.',
        default=20
    )
    nbGroup.add_argument(
        '-mask_psf',
        help='Set to mask PSF core used for fitting WL image.',
        action='store_true'
    )
    nbGroup.add_argument(
        '-smooth',
        type=float,
        metavar='float',
        help='Standard deviation of 2D Gaussian spatial smoothing kernel. Default: 1.5px',
        default=None
    )
    nbGroup.add_argument('-medsub',
        help="Median correct before WL subtraction.",
        action="store_true"
    )

    nbGroup.add_argument('-ext',
                        type=str,
                        metavar='string',
                        help='Extension added to input filename for output image. Default: \'.pNB.fits\' ',
                        default='.pNB.fits'
    )
    maskGroup = parser.add_mutually_exclusive_group(required=False)
    maskGroup.add_argument(
        '-fg_mask',
        help="Mask file to use for masking foreground."
    )
    maskGroup.add_argument(
        '-fg_reg',
        help="Region file of foreground sources or sources to mask."
    )
    args = parser.parse_args()

    #Load data
    infits = fits.open(args.cube)
    cube, hdr = infits[0].data, infits[0].header
    hdr2D = coordinates.get_header2d(hdr)
    wcs2D = WCS(hdr2D)
    w,y,x = cube.shape

    #Load params and get QSO position
    if args.pos!=None:

        qso_pos = tuple(float(x) for x in args.pos.split(','))

    elif args.par!=None:

        #Open CWITools params
        p = parameters.load_params(args.par)

        #Get position if not provided as a separate argument
        qso_pos = wcs2D.all_world2pix(p["TARGET_RA"], p["TARGET_DEC"], 0)

    #If no CWITools parameters or qso_pos provided
    else:

        print("No target position provided (see -h menu for details.) White-light subtraction cannot be performed.")
        qso_pos = None

    if args.var!=None:
        varcube = fits.getdata(args.var)
    else:
        varcube = variance.estimate_variance(infits)


    if args.fg_mask == None and args.fg_reg == None:
        fg_mask = []

    elif args.fg_reg != None:
        reg = pyregion.open(args.fg_reg)
        fg_mask = pyregion.get_mask(reg, hdr2D)

    else:
        fg_mask = np.getdata(args.src_mask)

    #Get width of NB
    if args.dv!=None: bandwidth = ( args.cWav*args.dv/3e5 )
    elif args.dw!=None: bandwidth = args.dw
    else: print("Error. Paramters dv or dw must be provided in proper format. See -h menu for help."); sys.exit()

    #print("%s,"%args.cube.split('/')[-2], end='')
    NB, WL, NB_var, WL_var = imaging.get_pseudo_nb(
        infits,
        args.wav,
        bandwidth,
        pos = qso_pos,
        cwing = args.cwidth,
        fit_rad = args.fit_rad,
        sub_rad = args.sub_rad,
        smooth = args.smooth,
        mask_psf = args.mask_psf,
        fg_mask = fg_mask,
        var = varcube
    )

    #Add info to header
    hdr2D["pNBW0"] = args.wav
    hdr2D["pNBDW"] = bandwidth

    # Save NB
    pnb_out = args.cube.replace(".fits", args.ext)
    pnb_fits = fits.HDUList([fits.PrimaryHDU(NB)])
    pnb_fits[0].header = hdr2D
    pnb_fits.writeto(pnb_out, overwrite=True)
    print("Saved %s" % pnb_out)

    var_out = pnb_out.replace(".fits", ".var.fits")
    var_fits = fits.HDUList([fits.PrimaryHDU(NB_var)])
    var_fits[0].header = hdr2D
    var_fits.writeto(var_out, overwrite=True)
    print("Saved %s"%var_out)

    #Save WL image
    wl_out = pnb_out.replace(".fits", ".WL.fits")
    wl_fits = fits.HDUList([fits.PrimaryHDU(WL)])
    wl_fits[0].header = hdr2D
    wl_fits.writeto(wl_out, overwrite=True)
    print("Saved %s"%wl_out)


    wl_var_out = pnb_out.replace(".fits", ".WL.var.fits")
    wl_var_fits = fits.HDUList([fits.PrimaryHDU(WL_var)])
    wl_var_fits[0].header = hdr2D
    wl_var_fits.writeto(wl_var_out, overwrite=True)
    print("Saved %s" % wl_var_out)

    #Save SNR
    SNR = NB/np.sqrt(NB_var)
    snr_out = pnb_out.replace('.fits', '.SNR.fits')
    snr_fits = fits.HDUList([fits.PrimaryHDU(SNR)])
    snr_fits[0].header = hdr2D
    snr_fits.writeto(snr_out, overwrite=True)
    print("Saved %s"%snr_out)

if __name__=="__main__": main()
