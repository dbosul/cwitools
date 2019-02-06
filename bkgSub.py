from astropy.io import fits 
from astropy.modeling import models,fitting
from scipy.signal import medfilt
from scipy.ndimage.filters import generic_filter
import argparse
import numpy as np
import os
import sys
import time

import libs

#Timer start
tStart = time.time()

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Perform background subtraction on a data cube.')

mainGroup = parser.add_argument_group(title="Main",description="Basic input")
mainGroup.add_argument('cube', 
                    type=str, 
                    metavar='cube',             
                    help='The cube to be subtracted.'
)

methodGroup = parser.add_argument_group(title="Methods",description="Parameters related to BKG Subtraction methods.")
methodGroup.add_argument('-method',
                    type=str,
                    metavar='Method',
                    help='Which method to use for subtraction. Polynomial fit or median filter. (\'medfilt\' or \'polyFit\')',
                    choices=['medfilt','polyfit'],
                    default='medfilt'
)
methodGroup.add_argument('-k',
                    type=int,  
                    metavar='Polynomial Degree',  
                    help='Degree of polynomial (if using polynomial sutbraction method).',
                    default=1
)
methodGroup.add_argument('-w',
                    type=int,
                    metavar='MedFilt Window',
                    help='Size of window (if using median filtering method).',
                    default=31
)
methodGroup.add_argument('-zmask',
                    type=str,
                    metavar='Wav Mask',
                    help='Z-indices to mask when fitting or median filtering (e.g. \'21,32\')',
                    default='0,0'
)
fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")
fileIOGroup.add_argument('-save',
                    type=str,
                    metavar='Save Model',
                    help='Set to True to output background model cube (.bg.fits)',
                    choices = ["True","False"],
                    default = "False"
)
fileIOGroup.add_argument('-ext',
                    type=str,
                    metavar='File Extension',
                    help='Extension to append to input cube for output cube (.bs.fits)',
                    default='.bs.fits'
)
args = parser.parse_args()

#Try to load the fits file
try: F = fits.open(args.cube)
except: print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()

#Try to parse the wavelength mask tuple
try: z0,z1 = tuple(int(x) for x in args.zmask.split(','))
except: print("Could not parse zmask argument. Should be two comma-separated integers (e.g. 21,32)");sys.exit()

#Parse arg.save from str to bool
arg.save = True if arg.save=="True" else False

#Output info to user
print("""
CWITools Background Subtraction
--------------------------------------
Input Cube: {0}
Method: {1}""".format(args.cube,args.method))
if args.method=='polyfit': print("Degree: {0}".format(args.k))
elif args.method=='medfilt': print("Window size: {0}".format(args.w))
print("--------------------------------------")

#Load header and data
header = F[0].header
cube   = F[0].data
W      = libs.cubes.getWavAxis(header)
useW   = np.ones_like(W,dtype=bool)   
maskZ  = False

#If using polynomial subtraction, initialize fitter, model and mask    
if args.method=='polyfit':
    
    useW[z0:z1] = 0
    fitter  = fitting.LinearLSQFitter()
    pModel0 = models.Polynomial1D(degree=args.k)

elif args.method=='medfilt' and z1>0:

    #Get +/- 5px windows around masked region
    a = max(0,z0-6)
    b = min(cube.shape[0],z1+6)
    
    #Use 'useW' to select this wing region
    useW[a:z0] = 0
    useW[z1:b] = 0
    
    #Warn user in the rare case there aren't at least 5 pixels in this region total
    if np.count_nonzero(useW)<5: print("Warning: masked region too large to get local median around it.")
    
    #Set maskZ variable to True
    maskZ = True
#Run through spaxels and subtract low-order polynomial
for yi in range(cube.shape[1]):
    for xi in range(cube.shape[2]):
        
        #Extract spectrum at this location
        spectrum = cube[:,yi,xi].copy()
        
        #Median filtering method
        if args.method=='medfilt':
            
            #Replace masked region median of 5px window either side
            if maskZ: spectrum[z0:z1] = np.median(spectrum[useW==0])
            
            #Get median filtered spectrum as background model
            bgModel = generic_filter(spectrum,np.median,size=args.w,mode='reflect')
            
        else:
        
            #Fit polynomial to data, ignoring masked pixels
            pModel1 = fitter(pModel0,W[useW],spectrum[useW])
            
            #Get background model
            bgModel = pModel1(W)
            
        F[0].data[:,yi,xi] -= bgModel
        
#Write out PSF-subtracted fits
outFile = args.cube.replace('.fits',args.ext)
F.writeto(outFile,overwrite=True)
print("Saved %s" % outFile)

#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))                    
