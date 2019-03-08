from astropy.io import fits 
from scipy.signal import medfilt
from scipy.ndimage.filters import generic_filter
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

import libs

#Timer start
tStart = time.time()

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Clip systematic residuals from bad sky lines.')

mainGroup = parser.add_argument_group(title="Main",description="Basic input")
mainGroup.add_argument('cube', 
                    type=str, 
                    metavar='cube',             
                    help='The cube to be subtracted.'
)

methodGroup = parser.add_argument_group(title="Methods",description="Parameters related to BKG Subtraction methods.")
methodGroup.add_argument('-zPairs',
                    type=str,
                    metavar='str',
                    help='Tuples of z-ranges over which to perform sigma-clipping. Each pair separated by commas, each range given as a colon-separated tuple of floats or integers. E.g. 4350:4360,4400:4410'

)
methodGroup.add_argument('-zunit',
                    type=str,
                    metavar='str',
                    help='Unit of input for zmask. Can be Angstrom (A) or Pixels (px) (Default: A)',
                    default='A',
                    choices=['A','px']
)
methodGroup.add_argument('-sigclip',
                    type=float,
                    metavar='float',
                    help='Threshold to use when sigma-clipping',
                    default=3
)
fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")
fileIOGroup.add_argument('-ext',
                    type=str,
                    metavar='File Extension',
                    help='Extension to append to input cube for output cube (.bs.fits)',
                    default='.sc.fits'
)
args = parser.parse_args()

#Try to load the fits file
try: F = fits.open(args.cube)
except: print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()

D = F[0].data
H = F[0].header
w,y,x = D.shape
k = 11

#Try to parse the wavelength mask tuple
if ',' not in args.zPairs and ':' not in args.zPairs:
    print("Syntax Error: Please give z-ranges as list of the form z0:z1,z2:z3,z3:z4,...")
    sys.exit()
else:

    pairs = args.zPairs.split(',')

    for p in pairs:
        
        try: z0,z1 = tuple( float(x) for x in p.split(':') )
        except:
            print("Syntax Error: Please give z-ranges as list of the form z0:z1,z2:z3,z3:z4,..")
            sys.exit()
        
        if args.zunit=='A': z0,z1 = libs.cubes.getband(z0,z1,H)
        
        z1+=1
        
        if z0<k or z1>D.shape[0]-k: continue
        
        z00 = z0-k
        z11 = z1+k
        
        zSlice = D[z00:z11].copy()

        zSlice_Filt = medfilt(zSlice,kernel_size=(2*k+1,1,1))
        

        F[0].data[z0:z1] = zSlice_Filt[k:-k]

        
F.writeto(args.cube.replace('.fits',args.ext),overwrite=True)
        
        
