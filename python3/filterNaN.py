from astropy.io import fits as fits
import argparse
import numpy as np
import sys
import libs
import matplotlib.pyplot as plt

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Replace NAN voxels in a cube with some other value.')


parser.add_argument('cube', 
                    type=str, 
                    metavar='str',             
                    help='Cube to filter.)'
)
parser.add_argument('-replaceVal', 
                    type=str, 
                    metavar='str',             
                    help='Value to replace NANs with. Can be INF, ZERO, MEDIAN or a numerical value',
                    default='0'      
)
parser.add_argument('-ext', 
                    type=str, 
                    metavar='str',             
                    help='Extension to append to output file. Default \'.nf.fits\'',
                    default='.nf.fits'      
)

args = parser.parse_args()

#Try to load the fits file
try: F = fits.open(args.cube)
except: print(("Error: could not open '%s'\nExiting."%args.cube));sys.exit()

#Get replacement value
args.replaceVal = args.replaceVal.upper()
if args.replaceVal=="INF": args.replaceVal = np.inf
elif args.replaceVal=="ZERO": args.replaceVal = 0
elif args.replaceVal=="MEDIAN": args.replaceVal = np.median(F[0].data)
else:
    try: args.replaceVal=float(args.replaceVal)
    except: print("Error parsing -replaceVal parameter. Should be floating point value if not INF/ZERO/MEDIAN");sys.exit()
    F[0].data[F[0].data==np.nan] = args.replaceVal
    F.writeto(args.cube.replace('.fits',args.ext),overwrite=True)
