"""bgSubtract: Model & subtract extended continuum emission or scattered light.
"""
from .. imports libs

from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.modeling import models,fitting
from scipy.signal import medfilt
from scipy.stats import tstd
from scipy.ndimage.filters import generic_filter


import argparse
import numpy as np
import sys


def run(cubePath,method='polyfit',polyK=1,medfiltWindow=31,zMask=(0,0),
        zUnit='A',saveModel=False,fileExt='.bs.fits'):
    """
    Subtracts extended continuum emission / scattered light from a cube

    Args:
        cubePath (str): Path to the data cube to be subtracted.
        method (str): Which method to use to model background
            'polyfit': Fits polynomial to the spectrum in each spaxel (default.)
            'median': Subtract the spatial median of each wavelength layer.
            'medfilt': Model spectrum in each spaxel by median filtering it.
            'noiseFit': Model noise in each z-layer and subtract mean.
        polyK (int): The degree of polynomial to use for background modeling.
        medfiltWindow (int): The filter window size to use if median filtering.
        zMask (int tuple): Wavelength region to mask, given as tuple of indices.
        zUnit (str): If using zmask, indices are given in these units.
            'A': Angstrom (default)
            'px': pixels
        saveModel (bool): Set to TRUE to save background model cube.
        fileExt (str): File extension to use for output (Default: .bs.fits)
    """

    #Try to load the fits file
    try: F = fits.open(cubePath)
    except: print("Error: could not open '%s'\nExiting."%cubePath);sys.exit()

    #Try to parse the wavelength mask tuple
    try: z0,z1 = tuple(int(x) for x in zmask.split(','))
    except: print("Could not parse zmask argument. Should be two comma-separated integers (e.g. 21,32)");sys.exit()

    #Output info to user
    print("""
    CWITools Background Subtraction
    --------------------------------------
    Input Cube: {0}
    Method: {1}""".format(cubePath,method))
    if method=='polyfit': print("Degree: {0}".format(polyK))
    elif method=='medfilt': print("Window size: {0}".format(medfiltWindow))

    #Load header and data
    header = F[0].header
    cube   = F[0].data
    W      = libs.cubes.getWavAxis(header)
    useW   = np.ones_like(W,dtype=bool)
    maskZ  = False
    modelC = np.zeros_like(cube)

    #Get empty regions mask
    mask2D = np.sum(cube,axis=0)==0

    #Convert zmask to pixels if given in angstrom
    if zUnit=='A': z0,z1 = libs.cubes.getband(z0,z1,header)

    print("Zmask (px): %i,%i"%(z0,z1))
    print("--------------------------------------")


    #Subtract background by fitting a low-order polynomial
    if method=='polyfit':

        useW[z0:z1] = 0
        fitter  = fitting.LinearLSQFitter()
        pModel0 = models.Polynomial1D(degree=polyK)

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

                if mask2D[yi,xi]==0:

                    F[0].data[:,yi,xi] -= bgModel

                    #Add to model
                    modelC[:,yi,xi] += bgModel

    #Subtract background by estimating it with a median filter
    elif method=='medfilt':

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

                    #Apply mask
                    spectrum[z0:z1+1] = m*ZZ + c

                #Get median filtered spectrum as background model
                bgModel = generic_filter(spectrum,np.median,size=medfiltWindow,mode='reflect')

                if mask2D[yi,xi]==0:

                    #Subtract from data
                    F[0].data[:,yi,xi] -= bgModel

                    #Add to model
                    modelC[:,yi,xi] += bgModel

    #Subtract layer-by-layer by fitting noise profile
    elif method=='noiseFit':
        fitter = fitting.SimplexLSQFitter()
        medians = []
        for wi in range(cube.shape[0]):

            #Extract layer
            layer = cube[wi]
            layerNonZ = layer[~mask2D]

            #Get median
            median = np.median(layerNonZ)
            stddev = np.std(layerNonZ)
            trimmed_stddev = tstd(layerNonZ,limits=(-3*stddev,3*stddev))
            trimmed_median = np.median(layerNonZ[np.abs(layerNonZ-median)<3*trimmed_stddev])

            medians.append(trimmed_median)
        medians = np.array(medians)
        bgModel0 = models.Polynomial1D(degree=2)
        useW[z0:z1] = 0
        bgModel1 = fitter(bgModel0,W[useW],medians[useW])
        for i,wi in enumerate(W): F[0].data[i][~mask2D] -= bgModel1(wi)

    #Subtract using simple layer-by-layer median value
    elif method=="median":

        sigclip = SigmaClip(sigma=2)
        dataclipped = sigclip(cube)
        medianModel = np.zeros_like(cube)
        for wi in range(dataclipped.shape[0]):
            medianModel[wi][mask2D==0] = np.median(dataclipped[wi][mask2D==0])
        medianModel = medfilt(medianModel,kernel_size=(3,1,1))
        F[0].data -= medianModel

    #Write out PSF-subtracted fits
    outFile = cubePath.replace('.fits',fileExt)
    F.writeto(outFile,overwrite=True)
    print("Saved %s" % outFile)

    if saveModel:
        outFile2 = outFile.replace('.fits','.bg_model.fits')
        M = fits.HDUList([fits.PrimaryHDU(modelC)])
        M[0].header = F[0].header
        M.writeto(outFile2,overwrite=True)
        print("Saved %s" % outFile2)

if __name__=="__main__":

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
                        choices=['medfilt','polyfit','noiseFit','median'],
                        default='medfilt'
    )
    methodGroup.add_argument('-k',
                        type=int,
                        metavar='Polynomial Degree',
                        help='Degree of polynomial (if using polynomial sutbraction method).',
                        default=1
    )
    methodGroup.add_argument('-window',
                        type=int,
                        metavar='MedFilt Window',
                        help='Size of median window (if using median filtering method).',
                        default=31
    )
    methodGroup.add_argument('-zMask',
                        type=str,
                        metavar='Wav Mask',
                        help='Z-indices to mask when fitting or median filtering (e.g. \'21,32\' or \'4140,4170\')',
                        default='0,0'
    )
    methodGroup.add_argument('-zUnit',
                        type=str,
                        metavar='Wav Mask',
                        help='Unit of input for zmask. Can be Angstrom (A) or Pixels (px) (Default: A)',
                        default='A',
                        choices=['A','px']
    )
    fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")
    fileIOGroup.add_argument('-saveModel',
                        type=str,
                        metavar='Save BG Model',
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

    #Parse arg.save from str to bool
    saveModel = True if args.save=="True" else False

    run(args.cube,
        method=args.method,
        polyK=args.k,
        medfiltWindow=args.window,
        zmask=args.zMask,
        zunit=args.zUnit,
        saveModel=args.saveModel,
        fileExt=args.ext
    )
