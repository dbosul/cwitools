from astropy.io import fits 
from astropy.stats import SigmaClip
from astropy.modeling import models,fitting
from photutils import Background2D, MedianBackground
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
                    help='Z-indices to mask when fitting or median filtering (e.g. \'21,32\' or \'4140,4170\')',
                    default='0,0'
)
methodGroup.add_argument('-zunit',
                    type=str,
                    metavar='Wav Mask',
                    help='Unit of input for zmask. Can be Angstrom (A) or Pixels (px) (Default: A)',
                    default='A',
                    choices=['A','px']
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
fileIOGroup.add_argument('-bgExt',
                    type=str,
                    metavar='File Extension',
                    help='Extension to append to input cube for output cube (.bs.fits)',
                    default='.bgModel.fits'
)
args = parser.parse_args()

#Try to load the fits file
try: F = fits.open(args.cube)
except: print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()

#Try to parse the wavelength mask tuple
try: z0,z1 = tuple(int(x) for x in args.zmask.split(','))
except: print("Could not parse zmask argument. Should be two comma-separated integers (e.g. 21,32)");sys.exit()


#Parse arg.save from str to bool
args.save = True if args.save=="True" else False

#Output info to user
print("""
CWITools Background Subtraction
--------------------------------------
Input Cube: {0}
Method: {1}""".format(args.cube,args.method))
if args.method=='polyfit': print("Degree: {0}".format(args.k))
elif args.method=='medfilt': print("Window size: {0}".format(args.w))

#Load header and data
header = F[0].header
cube   = F[0].data
W      = libs.cubes.getWavAxis(header)
useW   = np.ones_like(W,dtype=bool)   
maskZ  = False
modelC = np.zeros_like(cube)

#Convert zmask to pixels if given in angstrom
if args.zunit=='A': z0,z1 = libs.cubes.getband(z0,z1,header)

print("Zmask (px): %i,%i"%(z0,z1))
print("--------------------------------------")


#If using polynomial subtraction, initialize fitter, model and mask    
if args.method=='polyfit':
    
    useW[z0:z1] = 0
    fitter  = fitting.LinearLSQFitter()
    pModel0 = models.Polynomial1D(degree=args.k)

    #Track progress % using n
    xySize = cube[0].size
    n = 0

    #Run through spaxels and subtract low-order polynomial
    for yi in range(cube.shape[1]):
        for xi in range(cube.shape[2]):
        
            n+=1
            p = 100*float(n)/xySize
            sys.stdout.write('%5.2f percent complete\r'%p)
            sys.stdout.flush()   
         
            #Extract spectrum at this location
            spectrum = cube[:,yi,xi].copy()        

            #Fit polynomial to data, ignoring masked pixels
            pModel1 = fitter(pModel0,W[useW],spectrum[useW])
            
            #Get background model
            bgModel = pModel1(W)
                             
            F[0].data[:,yi,xi] -= bgModel

            #Add to model
            modelC[:,yi,xi] += bgModel     
                   
elif args.method=='medfilt':

    #Get +/- 5px windows around masked region, if mask is set
    if z1>0:
    
        #Get size of window region used to interpolate (minimum 5 to get median)
        nw = max(5,(z1-z0))
        
        #Get left and right index of window regions
        a = max(0,z0-nw)
        b = min(cube.shape[0],z1+nw)
        
        #Get two z mid-points which we will use for calculating line slope/intercept
        ZA = (a+z0)/2.0
        ZB = (b+z1)/2.0

        maskZ = True


    #Track progress % using n
    xySize = cube[0].size
    n = 0
            
    for yi in range(cube.shape[1]):
        for xi in range(cube.shape[2]):
            n+=1
            p = 100*float(n)/xySize
            sys.stdout.write('%5.2f percent complete\r'%p)
            sys.stdout.flush()   
         
            #Extract spectrum at this location
            spectrum = cube[:,yi,xi].copy()

            #Fill in masked region with smooth linear interpolation           
            if maskZ:

                #Calculate slope and intercept
                YA = np.mean(spectrum[a:z0]) if (z0-a)<5 else np.median(spectrum[a:z0])
                YB = np.mean(spectrum[z1:b]) if (b-z1)<5 else np.median(spectrum[z1:b])         
                m  = (YB-YA)/(ZB-ZA)
                c  = YA - m*ZA
                
                #Get domain for masked pixels
                ZZ = np.arange(z0,z1+1)
                spectrum[z0:z1+1] = m*ZZ + c
            
            #Get median filtered spectrum as background model
            bgModel = generic_filter(spectrum,np.median,size=args.w,mode='reflect')
                 
            #Subtract from data
            F[0].data[:,yi,xi] -= bgModel        
            
            #Add to model
            modelC[:,yi,xi] += bgModel
        
#Write out PSF-subtracted fits
outFile = args.cube.replace('.fits',args.ext)
F.writeto(outFile,overwrite=True)
print("Saved %s" % outFile)

if args.save:
    outFile2 = args.cube.replace('.fits',args.bgExt)
    M = fits.HDUList([fits.PrimaryHDU(modelC)])
    M[0].header = F[0].header
    M.writeto(outFile2,overwrite=True)
    print("Saved %s" % outFile2)
        
    
    
#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))                    
