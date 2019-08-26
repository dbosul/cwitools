from astropy.io import fits as fits
from astropy import units

import argparse
import numpy as np
import sys
import libs
import time

# Timer start
tStart = time.time()

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Use RA/DEC and Wavelength reference points to adjust WCS.')


parser.add_argument('cube', 
                    type=str, 
                    metavar='path',             
                    help='Input cube to crop.)'
)
parser.add_argument('wavPair', 
                    type=str, 
                    metavar='float tuple',             
                    help='Wavelength range (in angstrom) to crop to (e.g. 4160,4180)'
)
parser.add_argument('-ext', 
                    type=str, 
                    metavar='str',             
                    help='Extension to add to cropped cube filename (default: .wcrop.fits)',
                    default=".wcrop.fits"
)
args = parser.parse_args()


#Try to load the fits file
try: F = fits.open(args.cube)
except: print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()

#Try to parse wavelength tuple
try: w0,w1 = (float(x) for x in args.wavPair.split(','))
except:
    print("Could not parse wavelengths from input. Please check syntax (should be comma-separated tuple of floats representing upper/lower bound in wavelength for cropped cube.")
    sys.exit();

#Get indices of upper and lower bound
a,b = libs.cubes.getband(w0,w1,F[0].header)

#Crop cube
F[0].data = F[0].data[a:b]

#Update header
F[0].header["CRPIX3"] -= a

#Get output name and save
outFile = args.cube.replace('.fits',args.ext)
F.writeto(outFile,overwrite=True)
print("Saved %s."%outFile)


#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))        
