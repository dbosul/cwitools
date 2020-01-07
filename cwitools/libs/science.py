"""CWITools Science Functions Library.

This module contains functions for use in scientific analysis or the generation
of scientific products from data cubes (such as pseudo-NarrowBand images).

"""
from cwitools.libs import cubes

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.constants import G
from astropy.convolution import Gaussian1DKernel,Box1DKernel,Gaussian2DKernel,convolve
from astropy.cosmology import WMAP9 as cosmo
from astropy.modeling import models,fitting
#from astropy.modeling.functional_models import Moffat2D

from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales as getPxScales
from scipy.ndimage import gaussian_filter as GaussND
from scipy.stats import sigmaclip
from scipy.optimize import differential_evolution

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pyregion

def first_moment(x, y):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)

    Returns:
        float: The first moment in x.

    """
    return np.sum(x*y)/np.sum(y)

def second_moment(x, y, mu):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)

    Returns:
        float: The second moment in x.

    """
    return np.sum(y*(x-mu)**2 )/np.sum(y)

#Basic moments calculatio
def basic_moments(x, y, pos_thresh=False):
    """Calculate first and second moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        pos_thresh (bool): Set to TRUE to exclude negative weights.

    Returns:
        float: The first moment in x.
        float: The second moment in x.

    """
    if pos_thresh:
        x = x[y > 0]
        y = y[y > 0]

    mu_1 = first_moment(x, y)
    mu_2 = second_moment(x, y, mu_1)

    return mu_1, mu_2

#Convergent method for moments calculation
def closing_window_moments(x, y, mu1_init = None, window_max=25, window_min=10,
     window_step_size=1 ):
    """Calculate first and second moments using the 'closing-window method' (O'Sullivan et al. 2020).

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        mu1_init (float): Initial guess for first moment, used to center window.
            If none given, center value of x will be used.
        window_max (float): Starting window size for calculation (in same unit as x)
        window_min (float): Minimum window size for calculation. (same units as x)
        window_step_size (float): Decrease in window size for each step (same units as x).

    Returns:
        float: The first moment in x.
        float: The second moment in x.

    """
    #Take user input as initial guess or use center of x-range
    if mu1_init == None: mu_1 = x[int(len(x)/2)]
    else: mu_1 = mu1_init

    #Initialize window at maximum size
    window = window_max

    # Loop over window size
    while window > window_min:

        #Get indices of values to use for this calculation
        usex = ( np.abs(x - mu_1) < window/2 ) & (y > 0)

        #Calculate moments (i.e. update window center)
        mu_1 = first_moment(x[usex], y[usex])
        mu_2 = second_moment(x[usex], y[usex], mu_1 )

        #Update window size
        window -= window_step_size

    return mu_1, mu_2

def akaike_weights(aic_list):
    """Get weights representing relative likelihood of models based on AIC values.

    Args:
        
    """
    delta_i = aic_list - np.min(aic_list) #Minimum AIC value of set
    rel_L = np.exp(-0.5*delta_i) #Proportional likelihood term
    weights = rel_L/np.sum(rel_L)
    return weights

def rss(data, model): return np.sum(np.power(data-model, 2))

def aic(rss, k, n): return n*np.log(rss/n) + 2*k

def aic_c(rss, k, n):
    aic0 = aic(rss, k, n)
    correction = (2*k*k + 2*k)/(n - k - 1)
    return aic0 + correction

def bic(rss, k, n): return n*np.log(rss/n) + k*np.log(n)

def gauss1D(x,par):
    """A simple one-dimensional gaussian.

    Args:
        x (scalar or np.array): The domain for the gaussian.
        par (list/tuple): A list/tuple of amplitude, mean, standard-deviation.

    Returns:
        scalar or numpy.array: The value of the gaussian function at/over x.
    """
    return par[0]*np.exp(-0.5*np.power(x-par[1],2)/par[2] )

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
def pseudo_nb(inpFits, center, bandwidth, wlsub=True, pos=None, cwing=20,
            fitRad=2, subRad=None, maskPSF=True, smooth=None):

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

    cube,header = inpFits[0].data, inpFits[0].header

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

#Return a pseudo-Narrowband image (either SB units or SNR)
def pseudo_nb_2(inpFits, center, bandwidth, wlsub=True, pos=None, cwing=20,
            fitRad=2, subRad=None, maskPSF=True, smooth=None, reg=None, var=[]):

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
        reg (string): Path to DS9 region file containing continuum sources.

    Returns:
        numpy.ndarray: A white-light image with the given center/width
        numpy.ndarray: The PSF model which was subtracted from the data
        numpy.ndarray: The mask used to cover foreground sources
        numpy.ndarray: The pseudo-narrowband image, in surface-brightness units.

    """

    cube,header = inpFits[0].data, inpFits[0].header

    if var==[]: usevar=False
    else: usevar=True

    #Prep: get some useful structures numbers
    hdr2D = cubes.get_header2d(header)
    wcs2D = WCS(hdr2D) #Astropy world-coord sys
    pxScls = getPxScales(wcs2D)*3600 #Pixel scales in degrees (x3600 to arcsec)
    pxArea = pxScls[0]*pxScls[1] #Pixel size in arcsec2
    wavbin  = header["CD3_3"] #Spectral plate scale in angstrom/px
    halfwidth = bandwidth/2.0

    xQ, yQ = pos

    #Create plain narrow-band
    A,B = cubes.get_indices(center-halfwidth, center+halfwidth, header)

    #If requested NB is not in range of cube, return none type
    if B<=0 or A>=cube.shape[0]-1: return np.zeros_like(cube[0])
    #If it is clipped on left, adjust size
    elif A<0: A = 0
    #Clipped on right, adjust size
    elif B>cube.shape[0]-1: B=-1

    #Get indices of WL image
    a,b = cubes.get_indices(center-bandwidth-cwing,center+bandwidth+cwing,header)

    #Get WL image in SB units
    WL = np.sum(cube[a:A],axis=0) + np.sum(cube[B:b],axis=0)
    WL *= wavbin/pxArea

    #Create narrowband image
    NB = np.sum(cube[A:B], axis=0)
    NB *= wavbin/pxArea

    if usevar:

        #White-light variance image
        WL_var = np.sum(var[a:A],axis=0) + np.sum(var[B:b],axis=0)
        WL_var *= (wavbin/pxArea)**2

        #NB variance image
        NB_var = np.sum(var[A:B], axis=0)
        NB_var *= (wavbin/pxArea)**2

    #fMask is the 2D mask of pixels used for fitting (by default the whole NB)
    fMask = np.ones_like(cube[0],dtype=bool)
    sMask = np.ones_like(cube[1],dtype=bool)

    #Now build up source mask and model if reg file given
    if reg != None:

        box_rad = 3
        fit_rad = 2
        psf_fitter = fitting.LevMarLSQFitter()

        reg_wcs = pyregion.open(reg)
        reg_img = reg_wcs.as_imagecoord(header=hdr2D)

        hdu2D = fits.PrimaryHDU(NB)
        hdu2D.header = hdr2D

        src_model = np.zeros_like(WL)
        src_mask = np.zeros_like(WL) #Mask of continuum (foreground) sources

        qso_mask = np.zeros_like(WL) #Keep QSO mask separate till the end

        yy, xx = np.indices(WL.shape)

        for i, src in enumerate(reg_wcs):

            x, y, rad = reg_img[i].coord_list
            x -= 1 #Adjust to 0-indexing
            y -= 1

            #Get distance mesh
            rr = np.sqrt( (xx-x)**2 + (yy-y)**2 )

            #Get distance from QSO
            dist = np.sqrt( (x-xQ)**2 + (y-yQ)**2)

            #Cast to integers for indexing
            y = int(round(y))
            x = int(round(x))

            #If outside FOV, ignore
            if np.min(rr) > 1: continue

            elif rad > 6: src_mask[ rr < rad ] = 1

            #Otherwise, process:
            else:

                #Extract box around source

                boxL, boxR = max(0, y-box_rad), min(WL.shape[0]-1, y+box_rad+1)
                boxB, boxT = max(0, x-box_rad), min(WL.shape[1]-1, x+box_rad+1)
                box = WL[boxL:boxR, boxB:boxT]

                #Get indices/axes of box
                box_xx, box_yy = np.indices(box.shape, dtype=float)
                box_xx -= (boxT-boxB)/2.0
                box_yy -= (boxR-boxL)/2.0

                #Set bounds on model
                fit_bounds = {'amplitude':(0, 5*np.max(box)),
                                 'x_mean':(-fit_rad, fit_rad),
                                 'y_mean':(-fit_rad, fit_rad),
                                 'x_stddev':(1.5, 4),
                                 'y_stddev':(1.5, 4)
                                }

                #make initial guess
                model_guess = models.Gaussian2D(amplitude=np.max(box),
                                              x_mean = 0,
                                              y_mean = 0,
                                              x_stddev = 2,
                                              y_stddev = 2,
                                              bounds = fit_bounds

                )

                #Fit model to data
                model_fit = psf_fitter(model_guess, box_yy, box_xx, box)

                #Adjust center of PSF back to global coords
                model_fit.x_mean += x
                model_fit.y_mean += y

                #Create elliptical source mask from fitted PSF
                mask_model = models.Ellipse2D(x_0 = model_fit.x_mean,
                                        y_0 = model_fit.y_mean,
                                        theta = model_fit.theta,
                                        a = 2*model_fit.x_stddev,
                                        b = 2*model_fit.y_stddev
                )

                #If this is the central source, add to QSO mask
                if dist < 1:

                    #Set fitting mask for later
                    fMask = rr<=fitRad

                    #If subtraction radius given, update sMask
                    if subRad!=None: sMask = rr<=subRad


                    #Add to QSO mask
                    qso_mask += mask_model(xx, yy)

                else:

                    #Add to source model and mask
                    src_model += model_fit(xx, yy)
                    src_mask  += mask_model(xx, yy)

        src_mask[src_mask > 0] = 1
        src_mask[qso_mask > 0] = 2 #2 indicates QSO or central source

    else:
        src_model = np.zeros_like(NB)
        src_mask = np.zeros_like(NB)


    #Median subtract both
    # NB -= np.median(NB[ src_mask == 0 ])
    # WL -= np.median(WL[ src_mask == 0 ])

    # # Vertical median correction
    # for xi in range(NB.shape[1]):
    #     NB[:, xi] -= np.median( NB[src_mask[:, xi]==0, xi] )
    #     WL[:, xi] -= np.median( WL[src_mask[:, xi]==0, xi] )
    #
    # # # Horizontal median correction
    for yi in range(NB.shape[0]):
        NB[yi, :] -= np.median( sigmaclip(NB[yi, src_mask[yi, :]==0 ], low=3, high=3).clipped )
        WL[yi, :] -= np.median( sigmaclip(WL[yi, src_mask[yi, :]==0 ], low=3, high=3).clipped)



    #If smoothing requested
    if smooth!=None:
        NB = smooth2d(NB, smooth, ktype='gaussian')
        WL = smooth2d(WL, smooth, ktype='gaussian')
        # NB_sub = smooth2d(NB_sub, smooth, ktype='gaussian')
        # NB_sub_var = smooth2d(NB_sub_var, smooth, ktype='gaussian', var=True)
        NB_var =  smooth2d(NB_var, smooth, ktype='gaussian', var=True)



    scalingFactors = NB[fMask]/WL[fMask]
    scalingFactors_Clipped = sigmaclip(scalingFactors,high=2.5,low=2.5)
    scalingFactors_Mean = np.median(scalingFactors)
    S = scalingFactors_Mean

    sMask[WL<=0] = 1 #Do not subtract negative values
    sMask[src_mask == 1] = 0 #Do not subtract over other sources
    #Subtract WL image

    NB_sub = NB.copy()
    NB_sub[sMask] -=  S*WL[sMask]
    NB_sub[src_mask == 1] = 0
    NB_sub[fMask == 1] = 0

    NB_sub_var = NB_var.copy()
    NB_sub_var[sMask] += (S**2)*WL_var[sMask]

    NB_sub = noisefit_bgsubtract(NB_sub, src_mask)




    NB_sub = noisefit_bgsubtract(NB_sub, src_mask)
    NB_sub[src_mask == 1] = 0

    #Scale variance to match noise in NB image
    snr = NB_sub/np.sqrt(NB_sub_var)
    n, edges = np.histogram(snr, range=(-5,5), bins=40)
    centers = np.array([ (edges[i]+edges[i+1])/2.0 for i in range(edges.size-1)])
    fitter = fitting.SimplexLSQFitter()
    fit_inds = centers <= 1
    snrmodel0 = models.Gaussian1D(amplitude=n.max(), mean=0, stddev=1)
    snrmodel1 = fitter(snrmodel0, centers[fit_inds], n[fit_inds])


    NB_sub_var *= (snrmodel1.stddev.value**2)

    print(snrmodel1.stddev.value**2)
    if 0:
        snr2 = NB_sub/np.sqrt(NB_sub_var)
        n2, edges2 = np.histogram(snr2, range=(-5,5), bins=40)
        centers2 = np.array([ (edges2[i]+edges2[i+1])/2.0 for i in range(edges2.size-1)])
        snrmodel2 = models.Gaussian1D(amplitude=n2.max(), mean=0, stddev=1)
        plt.figure()
        plt.plot(centers, n, 'k.')
        plt.plot(centers, snrmodel0(centers), 'g-')
        plt.plot(centers, snrmodel1(centers), 'r-')
        plt.plot(centers2, n2, 'b.')
        plt.plot(centers, snrmodel2(centers), 'b-')
        plt.show()



    #Return SB map
    return WL, NB, NB_sub, NB_sub_var, src_mask

def noisefit_bgsubtract(data, mask=[], plot=False):
    if mask == []: mask=np.zeros_like(data)
    bg_pixels = data[mask == 0].flatten()

    n, edges = np.histogram(bg_pixels, bins=40)
    centers = np.array([ (edges[i]+edges[i+1])/2.0 for i in range(edges.size-1)])


    noisefitter = fitting.SimplexLSQFitter()
    noisemodel0 = models.Gaussian1D(amplitude=n.max(), mean=0, stddev=np.std(bg_pixels))


    #Fit noise
    noisemodel1 = noisefitter(noisemodel0, centers, n)

    data_sub = data.copy() - noisemodel1.mean.value

    if plot:
        plt.figure()
        plt.plot(centers, n, 'k.--')
        plt.plot(centers, noisemodel0(centers), 'g-')
        plt.plot(centers, noisemodel1(centers), 'r-')
        n2, edges2 = np.histogram(data_sub[mask==0], bins=40)
        centers2 = np.array([ (edges2[i]+edges2[i+1])/2.0 for i in range(edges2.size-1)])
        plt.plot(centers2, n2, 'b.--')
        plt.show()

    return data_sub

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
    if ktype=='box':
        kernel = Box1DKernel(scale)
    elif ktype=='gaussian':
        sig = fwhm2sigma(scale)
        kernel = np.array(Gaussian1DKernel(sig))
        kernel_vol =  np.sum(kernel)#2*np.pi*np.sqrt(np.max(kernel))*(sig**2)

    else: raise ValueError("Kernel type ('%s') not found"%ktype)

    kernel = np.array(kernel)

    if var: kernel = np.power(kernel,2)

    #Apply kernel
    for a in axes:
        cubeFilt = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'),
                                       axis=a,
                                       arr=cubeFilt.copy()
        )

    #Re-normalize data

    #if var=True: kernel

    #Return
    return cubeFilt

#Function to smooth along wavelength axis
def smooth2d(img, scale, ktype='gaussian', var=False):
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
    imgFilt = img.copy()

    #Set kernel type
    if ktype=='box': kernel = Box2DKernel(scale)
    elif ktype=='gaussian': kernel = Gaussian2DKernel(fwhm2sigma(scale))
    else: raise ValueError("Kernel type ('%s') not found"%ktype)

    print(np.sum(np.array(kernel)))

    if var: kernel = np.power(np.array(kernel), 2)

    imgFilt = convolve(img, kernel)

    #Return
    return imgFilt

def get_box_old(fitsIN,paramsIN,boxSize,fill=0):

    wcs2D = WCS(cubes.get_header2d(fitsIN[0].header))

    xC,yC = wcs2D.all_world2pix(paramsIN["RA"],paramsIN["DEC"],0)

    pkpc_per_px = get_pkpc_px(wcs2D,paramsIN["Z"])
    boxSize_px = boxSize/pkpc_per_px

    NBCutout = Cutout2D(fitsIN[0].data,(xC,yC),boxSize_px,wcs2D,mode='partial',fill_value=fill)

    return NBCutout.data
