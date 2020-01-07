from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from scipy.stats import sigmaclip

from cwitools import libs
import argparse
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import sys

import time

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Make channel maps of an input cube around a specified emission line.')
mainGroup = parser.add_argument_group(title="Main",description="Basic input")
mainGroup.add_argument('cube',
                    type=str,
                    metavar='cube',
                    help='The input data cube.'
)
nbGroup = parser.add_argument_group(title="pseudo-NB Parameters")
nbGroup.add_argument('-cWav',
                    type=float,
                    metavar='float',
                    help='Central (observed) wavelength to use for pseudo NB. Default: None.',
                    default=None

)
widthGroup = parser.add_mutually_exclusive_group(required=True)
widthGroup.add_argument('-dv',
                    type=float,
                    metavar='float',
                    help='Velocity full width (in km/s) to use for NB image.',
                    default=None

)
widthGroup.add_argument('-dw',
                    type=float,
                    metavar='float',
                    help='Wavelength full width (in Angstrom) to use for NB image.',
                    default=None

)

nbGroup.add_argument('-var',
                    type=str,
                    metavar='var',
                    help='Variance cube for calculating SNR.',
                    default=None
)

nbGroup.add_argument('-par',
                    type=str,
                    metavar='path',
                    help='CWITools parameter file for target (used for target position).'
)
nbGroup.add_argument('-pos',
                    type=str,
                    metavar='str',
                    help='Position of your source as an \'x,y\' tuple. Overrrides CWITools param file position if given. If neither -pos nor -par are provided, continuum subtraction will not be performed. Default: None',
                    default=None

)
nbGroup.add_argument('-snrMode',
                    type=str,
                    metavar='bool',
                    help='True = Make SNR map. False = Make SB map. (Default:False)',
                    default=False

)
nbGroup.add_argument('-cWidth',
                    type=float,
                    metavar='float',
                    help='Velocity width of wings used to create continuum image.',
                    default=2000

)
nbGroup.add_argument('-fitRadius',
                    type=float,
                    metavar='float',
                    help='Radius (px) around QSO to use for fitting WL image. Default: 3px',
                    default=3

)
nbGroup.add_argument('-maskPSF',
                    type=str,
                    metavar='bool',
                    help='True = Mask PSF core. False = No masking. (Default:True)',
                    default=True

)
nbGroup.add_argument('-smooth',
                    type=float,
                    metavar='float',
                    help='Standard deviation of 2D Gaussian spatial smoothing kernel. Default: 1.5px',
                    default=None

)
nbGroup.add_argument('-ext',
                    type=str,
                    metavar='string',
                    help='Extension added to input filename for output image. Default: \'.pNB.fits\' ',
                    default='.pNB.fits'

)
nbGroup.add_argument('-saveSpec',
                    type=str,
                    metavar='bool',
                    help='Save spectrum of nebular emission.',
                    default='True'

)
nbGroup.add_argument('-reg',
                    type=str,
                    metavar='REG file',
                    help='Path to ds9 region file of continuum sources.',
                    default='True'

)
args = parser.parse_args()

args.saveSpec = (args.saveSpec=='True')
args.snrMode= (args.snrMode=='True')
args.maskPSF= (args.maskPSF=='True')

#Constants
c = 3e5 #Speed of light in km/s

#Load data
infits = fits.open(args.cube)
cube,hdr = infits[0].data, infits[0].header
hdr2D = libs.cubes.get_header2d(hdr)
wcs2D = WCS(hdr2D)
w,y,x = cube.shape

#Load params and get QSO position
if args.pos!=None:

    qsoPos = tuple(float(x) for x in args.pos.split(','))

elif args.par!=None:

    #Open CWITools params
    p = libs.params.loadparams_old(args.par)

    #Get position if not provided as a separate argument
    qsoPos = wcs2D.all_world2pix(p["RA"],p["DEC"],0)

#If no CWITools parameters or qsoPos provided
else:

    print("No target position provided (see -h menu for details.) White-light subtraction cannot be performed.")
    qsoPos = None

z = args.cWav/1215.7 - 1

if args.var!=None: varcube = fits.getdata(args.var)
else: varcube = []

#Get width of NB
if args.dv!=None: bandwidth = ( args.cWav*args.dv/c )
elif args.dw!=None: bandwidth = args.dw
else: print("Error. Paramters dv or dw must be provided in proper format. See -h menu for help."); sys.exit()

cWidth = args.cWav*args.cWidth/c
pkpc_per_px = libs.science.get_pkpc_px(wcs2D,z)


WL, NB, NBsub, NBsub_var, mask = libs.science.pseudo_nb_2(infits, args.cWav, bandwidth,
    pos=qsoPos,
    cwing=cWidth,
    fitRad=args.fitRadius,
    smooth=args.smooth,
    maskPSF=args.maskPSF,
    reg = args.reg,
    var = varcube
)

#Add info to header
hdr2D["pNBw0"] = args.cWav
hdr2D["pNBdw"] = bandwidth

#1 Save NB
pnb_out = args.cube.replace(".fits",args.ext)
pnb_fits = libs.cubes.make_fits(NBsub, hdr2D)
pnb_fits.writeto(pnb_out, overwrite=True)
print("Saved %s"%pnb_out)

#2 Save variance
var_out = pnb_out.replace(".fits", ".var.fits")
var_fits = libs.cubes.make_fits(NBsub_var, hdr2D)
var_fits.writeto(var_out, overwrite=True)
print("Saved %s"%var_out)

#Save SNR
SNR = NBsub/np.sqrt(NBsub_var)
snr_out = pnb_out.replace('.fits', '.SNR.fits')
snr_fits = libs.cubes.make_fits(SNR, hdr2D)
snr_fits.writeto(snr_out, overwrite=True)
print("Saved %s"%snr_out)

#2 Save non-subtracted NB
pnb_out_nosub = pnb_out.replace(".fits", ".nosub.fits")
pnb_nosub_fits = libs.cubes.make_fits(NB, hdr2D)
pnb_nosub_fits.writeto(pnb_out_nosub, overwrite=True)
print("Saved %s"%pnb_out_nosub)

#2 Save WL image
wl_out = pnb_out.replace(".fits", ".WL.fits")
wl_fits = libs.cubes.make_fits(WL, hdr2D)
wl_fits.writeto(wl_out, overwrite=True)
print("Saved %s"%wl_out)

#2 Save src mask
msk_out = pnb_out.replace(".fits", ".msk.fits")
msk_fits = libs.cubes.make_fits(mask, hdr2D)
msk_fits.writeto(msk_out, overwrite=True)
print("Saved %s" % msk_out)
