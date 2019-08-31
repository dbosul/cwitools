"""CWITools Science Functions Library.

This module contains functions for use in scientific analysis or the generation
of scientific products from data cubes (such as pseudo-NarrowBand images).

"""
from cwitools.libs import cubes

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.constants import G
from astropy.convolution import Gaussian1DKernel,Box1DKernel
from astropy.cosmology import WMAP9 as cosmo
from astropy.modeling import models,fitting
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales as getPxScales
from scipy.ndimage import gaussian_filter as GaussND
from scipy.stats import sigmaclip
from scipy.optimize import differential_evolution

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np


def nonpos2inf(cube,level=0):
    """Replace values below a certain threshold with infinity.

    Args:
        cube (numpy.ndarray): The cube to process.
        thresh (float): The threshold below which to replace values (Default:0)

    Returns:
        numpy.ndarray: The modified cube.

    """
    newcube = cube.copy()
    newcube[newcube<=level] = np.inf
    return newcube

def fwhm2sigma(fwhm):
    """Convert a gaussian Full-Width at Half Maximum to a standard deviation.

    Args:
        fwhm (float): Full-Width at Half Maximum of a Gaussian function.

    Returns:
        float: The standard deviation of the equivalent Gaussian.

    """
    return fwhm/(2*np.sqrt(2*np.log(2)))

#Return a pseudo-Narrowband image (either SB units or SNR)
def pseudo_nb(inpFits,center,bandwidth, wlsub=True,pos=None,cwing=20,
            fitRad=2,subRad=None,maskPSF=True,smooth=None):

    """Create a pseudo-Narrow-Band (pNB) image from a data cube.

    Args:
        inpFits (astropy.io.fits.HDUList): The input FITS file.
        center (float): The central wavelength of the narrow-band in Angstrom.
        bandwidth (float): The width of the narrow-band image in Angstorm.
        wlsub (bool): TRUE = subtract continuum emission.
            If a source is provided with the pos argument, it will be used to
            scale the white-light image for subtraction. Otherwise the entire
            image will be used to find a scaling factor.
        pos (float tuple): The location the main continuum source in x,y.
        fitRad (float): The radius around the source to use for fitting.
        subRad (float): The radius around the soruce to use when subtracting.
        maskPSF (bool): TRUE = mask source after white-light subtraction.
        smooth (float): FWHM of a Gaussian kernel to use for smoothing the NB.

    Returns:
        numpy.ndarray: The pseudo-narrowband image, in surface-brightness units.

    """

    cube,header = fits.getdata(inpFits,header=True)
    #Prep: get some useful structures numbers
    wcs2D = WCS(cubes.get_header2d(header)) #Astropy world-coord sys
    pxScls = getPxScales(wcs2D)*3600 #Pixel scales in degrees (x3600 to arcsec)
    pxArea = pxScls[0]*pxScls[1] #Pixel size in arcsec2
    dwav  = header["CD3_3"] #Spectral plate scale in angstrom/px
    bandwidth /= 2.0

    #Create plain narrow-band
    A,B = cubes.get_indices(center-bandwidth,center+bandwidth,header)
    NB = np.sum(cube[A:B],axis=0)

    #If requested NB is not in range of cube, return none type
    if B<=0 or A>=cube.shape[0]-1: return np.zeros_like(cube[0])
    elif A<0: A = 0
    elif B>cube.shape[0]-1: B=-1

    #fMask is the 2D mask of pixels used for fitting (by default the whole NB)
    fMask = np.ones_like(cube[0],dtype=bool)
    sMask = np.ones_like(cube[1],dtype=bool)

    #Create white-light image and subtract if requested
    if wlsub:

        #Calculate wing indices and get WL image
        a,b = cubes.get_indices(center-bandwidth-cwing,center+bandwidth+cwing,header)
        WL = np.sum(cube[a:A],axis=0) + np.sum(cube[B:b],axis=0)

        #If source position given, get mask
        if pos!=None:
            yy,xx = np.indices(cube[0].shape)
            rr = np.sqrt( (yy-pos[1])**2 + (xx-pos[0])**2 )
            fMask = rr<=fitRad

            #If subtraction radius given, update sMask
            if subRad!=None: sMask = rr<=subRad

        scalingFactors = NB[fMask]/WL[fMask]
        scalingFactors_Clipped = sigmaclip(scalingFactors,high=2.5,low=2.5)
        scalingFactors_Mean = np.median(scalingFactors)
        S = scalingFactors_Mean

        sMask[WL<=0] = 0

        #Subtract WL image
        NB[sMask] -=  S*WL[sMask]

        if maskPSF: NB[fMask] = 0

        NB -= np.median(NB)

    #If smoothing requested
    if smooth!=None: NB = smooth3d(NB, smooth, axes=(0,1), ktype='gaussian')

    #Convert to SB units
    NB *= dwav/pxArea

    #Return SB map
    return NB

def get_pkpc_px(wcs2d,redshift=0):
    """Return the physical size of pixels in proper kpc. Assumes 1:1 aspect.

    Args:
        wcs2D (astropy.wcs.WCS): A 2D astropy WCS object.
        redshift (float): Cosmological redshift of the field/target.

    Returns:
        float: Proper kiloparsecs per pixel

    """
    #Get platescale in arcsec/px (assumed to be 1:1 aspect)
    pxScale = getPxScales(wcs2d)[0]*3600

    #Get pkpc/arcsec from cosmology
    pkpcScale = cosmo.kpc_proper_per_arcmin(redshift)/60.0

    #Get pkpc/pixel by combining
    pkpc_per_px = (pkpcScale*pxScale).value

    return pkpc_per_px

#Function to smooth along wavelength axis
def smooth3d(cube,scale,axes=(0,1,2),ktype='gaussian',var=False):
    """Smooth along all/any axes of a data cube with a box or gaussian kernel.

    Args:
        cube (numpy.ndarray): The input datacube.
        scale (float): The smoothing scale.
            For a gaussian kernel, this is full-width at half-maximum (FWHM)
            For a box kernel, this is the width of the box.
        axes (int tuple): The axes to smooth along.
        ktype (str): The kernel type ('gaussian' or 'box')
        var (bool): Set to TRUE when smoothing variance data.

    Returns:
        numpy.ndarray: The smoothed data cube.

    """
    #Make copy - do not modify input cube directly
    cubeFilt = cube.copy()

    #Set kernel type
    if ktype=='box': kernel = Box1DKernel(scale)
    elif ktype=='gaussian': kernel = Gaussian1DKernel(fwhm2sigma(scale))
    else: raise ValueError("Kernel type ('%s') not found"%ktype)

    #Square kernel if needed
    kernel = np.array(kernel)

    if var==True: kernel = np.power(kernel,2)

    #Apply kernel
    for a in axes: cubeFilt = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=a, arr=cubeFilt.copy())


    #Return
    return cubeFilt
