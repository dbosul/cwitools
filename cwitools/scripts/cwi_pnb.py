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
nbGroup.add_argument('-wav',
                    type=float,
                    metavar='float',
                    help='Central wavelength to use for pseudo NB. Default: None.',
                    default=None

)
widthGroup = parser.add_mutually_exclusive_group(required=True)
widthGroup.add_argument('-dv',
                    type=float,
                    metavar='float',
                    help='Pseudo-NB width in km/s.',
                    default=None

)
widthGroup.add_argument('-dw',
                    type=float,
                    metavar='float',
                    help='Pseudo-NB width in Angstrom.',
                    default=None

)

nbGroup.add_argument('-var',
                    type=str,
                    metavar='var',
                    help='Variance cube for SNR calculations. Estimated from cube otherwise.',
                    default=None
)

nbGroup.add_argument('-par',
                    type=str,
                    metavar='path',
                    help='CWITools parameter file (used for target position).'
)
nbGroup.add_argument('-pos',
                    type=str,
                    metavar='str',
                    help='Position of your source as an \'x,y\' tuple. Overrrides CWITools param file position if given. If neither -pos nor -par are provided, continuum subtraction will not be performed. Default: None',
                    default=None

)
nbGroup.add_argument('-cwidth',
                    type=float,
                    metavar='float',
                    help='Bandwidth (in km/s) to use for continuum image.',
                    default=2000

)
nbGroup.add_argument('-creg',
                    type=str,
                    metavar='REG file',
                    help='Region file of continuum sources to mask.',
                    default='True'

)
nbGroup.add_argument('-fitradius',
                    type=float,
                    metavar='float',
                    help='Radius (px) around QSO to use for fitting WL image. Default: 3px',
                    default=3

)

fileIOGroup.add_argument('-maskpsf',
                        help="Mask the region used to scale PSF for subtraction.",
                        action='store_true'
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

args = parser.parse_args()

#Load data
infits = fits.open(args.cube)
cube, hdr = infits[0].data, infits[0].header
hdr2D = libs.cubes.get_header2d(hdr)
wcs2D = WCS(hdr2D)
w,y,x = cube.shape

#Load params and get QSO position
if args.pos!=None:

    qso_pos = tuple(float(x) for x in args.pos.split(','))

elif args.par!=None:

    #Open CWITools params
    p = libs.params.loadparams(args.par)

    #Get position if not provided as a separate argument
    qso_pos = wcs2D.all_world2pix(p["RA"],p["DEC"],0)

#If no CWITools parameters or qso_pos provided
else:

    print("No target position provided (see -h menu for details.) White-light subtraction cannot be performed.")
    qso_pos = None

if args.var!=None: varcube = fits.getdata(args.var)
else: varcube = []

#Get width of NB
if args.dv!=None: bandwidth = ( args.cWav*args.dv/3e5 )
elif args.dw!=None: bandwidth = args.dw
else: print("Error. Paramters dv or dw must be provided in proper format. See -h menu for help."); sys.exit()

cwidth = args.wav*args.cwidth/c


WL, NB, NBsub, NBsub_var, mask = libs.science.pseudo_nb_2(infits, args.cWav, bandwidth,
    pos=qso_pos,
    cwing=cwidth,
    fitRad=args.fitradius,
    smooth=args.smooth,
    maskpsf=args.maskpsf,
    reg = args.creg,
    var = varcube
)

#Add info to header
hdr2D["pNBW0"] = args.cWav
hdr2D["pNBDW"] = bandwidth

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
