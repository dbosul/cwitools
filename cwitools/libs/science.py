"""CWITools Science Functions Library.

This module contains functions for use in scientific analysis or the generation
of scientific products from data cubes (such as pseudo-NarrowBand images).

"""

from cwitools.libs import cubes

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.constants import G
from astropy.convolution import Gaussian1DKernel,Box1DKernel,Gaussian2DKernel,Box2DKernel,convolve
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

import matplotlib
matplotlib.use('TkAgg')
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

def first_moment_err(x, y, y_var=[]):

    A = np.sum(x*y) #Numerator of moments calculation
    B = np.sum(y) #Denominator of moments calculation

    if y_var == []: y_var = np.var(y) #Estimate if no variance given

    return  np.sqrt( np.sum( y_var*np.power(B*x - A, 2)/np.power(B, 4) ) )

def second_moment_err(x, y, mu_1, y_var=[]):

    A = np.sum( x*y ) #Numerator of first moment calculation
    B = np.sum( y ) #Denominator of first/second moment calculation
    C = np.sum( np.power(x-mu_1, 2)*y ) #Numerator of second moment calc.
    R = np.sum( (x - mu_1)*y ) #Term needed for eq.
    dmu2_dIj = (B*x - A)/(B**2) #Another term needed
    mu_2 = np.sqrt(C/B) #Second moment

    if y_var == []: y_var = np.var(y) #Estimate if no variance given

    #Two squared terms that are multiplied by variance
    term1 = (1/(2*B*B*mu_2))**2
    term2 = ( B*np.power(x - mu_1, 2) + 2*B*dmu2_dIj*R - C )**2

    return np.sqrt( term1*np.sum(y_var*term2) )

#Basic moments calculatio
def basic_moments(x, y, y_var=[], pos_thresh=False, mu1_init=None, window=30):
    """Calculate first and second moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        pos_thresh (bool): Set to TRUE to exclude negative weights.

    Returns:
        float: The first moment in x.
        float: The second moment in x.

    """

    if mu1_init != None:
        usex = np.abs(x-mu1_init) <= window/2
        x = x[usex]
        y = y[usex]
        if y_var!=[]: y_var = y_var[usex]

    #Points with y=0 add noise without influencing result - so remove
    x = x[y != 0]
    if y_var!=[]: y_var = y_var[ y!= 0]
    y = y[y != 0]

    if pos_thresh:
        x = x[y > 0]
        if y_var!=[]: y_var = y_var[ y > 0]
        y = y[y > 0]

    #plt.figure();plt.plot(x, y, 'k.');plt.show()

    mu_1 = first_moment(x, y)
    mu_2 = second_moment(x, y, mu_1)

    mu_1_err = first_moment_err(x, y, y_var)
    mu_2_err = second_moment_err(x, y, mu_1, y_var)

    return mu_1, mu_2, mu_1_err, mu_2_err

#Convergent method for moments calculation
def closing_window_moments(x, y, mu1_init = None, window_max=25, window_min=15,
     window_step_size=1, y_var=[]):
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
        use = ( np.abs(x - mu_1) < window/2 ) & (y > 0)

        usex = x[use]
        usey = y[use]
        usevar = y_var if y_var==[] else y_var[use]

        #Calculate moments (i.e. update window center)
        mu_1 = first_moment(usex, usey)
        mu_2 = second_moment(usex, usey, mu_1 )

        mu_1_err = first_moment_err(usex, usey, usevar)
        mu_2_err = second_moment_err(usex, usey, mu_1, usevar)
        #Update window size
        window -= window_step_size

    return mu_1, mu_2, mu_1_err, mu_2_err

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
def pseudo_nb_2(inpFits, center, bandwidth, wlsub=True, pos=None, cwing=50,
            fitRad=2, subRad=None, maskPSF=True, smooth=None, reg=None, var=[],
            plot=False, r_inner=100, r_outer=200, redshift=2.5,
            h_corr=0, v_corr=0, medsub=1):

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
    pkpc_per_px = get_pkpc_px(wcs2D, redshift)
    r_inner /= pkpc_per_px
    r_outer /= pkpc_per_px
    pxScls = getPxScales(wcs2D)*3600 #Pixel scales in degrees (x3600 to arcsec)
    pxArea = pxScls[0]*pxScls[1] #Pixel size in arcsec2
    wavbin  = header["CD3_3"] #Spectral plate scale in angstrom/px
    halfwidth = bandwidth/2.0
    wav_axis = cubes.get_wavaxis(header)

    cube[ np.abs(wav_axis-4569)<=12] = 0 #TEMP
    #cube[ wav_axis-4569)<=13] = 0 #TEMP
    xQ, yQ = pos
    #yQ -= 2
    #Create plain narrow-band
    A,B = cubes.get_indices(center-halfwidth, center+halfwidth, header)

    #If requested NB is not in range of cube, return none type
    if B<=0 or A>=cube.shape[0]-1: return [np.zeros_like(cube[0])]*5
    #If it is clipped on left, adjust size
    elif A<0: A = 0
    #Clipped on right, adjust size
    elif B>cube.shape[0]-1: B=-1

    #Get indices of WL image
    a,b = cubes.get_indices(center-halfwidth-cwing,center+halfwidth+cwing,header)

    #TEMPORARY
    cube[np.abs(wav_axis-4359) < 3 ] = 0
    var[np.abs(wav_axis-4359) < 3 ] = 0

    #Get WL image in SB units
    WL = np.sum(cube[a:A],axis=0) +np.sum(cube[B:b],axis=0)
    WL *= wavbin/pxArea

    WL_backup = WL.copy()

    #Create narrowband image
    NB = np.sum(cube[A:B], axis=0)
    NB *= wavbin/pxArea

    if usevar:
        var[np.isnan(var)] = 0
        #White-light variance image
        WL_var = np.sum(var[a:A],axis=0) + np.sum(var[B:b],axis=0)
        WL_var *= (wavbin/pxArea)**2

        #NB variance image
        NB_var = np.sum(var[A:B], axis=0)
        NB_var *= (wavbin/pxArea)**2


    yy, xx = np.indices(WL.shape)
    rr_qso = np.sqrt( (xx-xQ)**2 + (yy-yQ)**2 )

    annulus_mask = (rr_qso > r_inner) & (rr_qso <= r_outer)
    #fMask is the 2D mask of pixels used for fitting (by default the whole NB)
    fMask = np.ones_like(cube[0],dtype=bool)
    sMask = np.ones_like(cube[1],dtype=bool)

    #Now build up source mask and model if reg file given
    if reg != None:

        box_rad = 3 #Size of box around each source to extract
        fit_rad = 2 #Radius to use for fitting
        psf_fitter = fitting.LevMarLSQFitter() #Fitter for PSF modeling
        reg_wcs = pyregion.open(reg) #World coordinate regions
        reg_img = reg_wcs.as_imagecoord(header=hdr2D) #Image coordinate regions
        src_model = np.zeros_like(WL) #Model of continuum sources
        src_mask = np.zeros_like(WL) #Mask of continuum (foreground) sources
        qso_mask = np.zeros_like(WL) #Keep QSO mask separate till the end

        for i, src in enumerate(reg_wcs):

            x, y, rad = reg_img[i].coord_list
            x -= 1
            y -= 1
            rr = np.sqrt( (xx-x)**2 + (yy-y)**2 ) #Get distance mesh
            dist = np.sqrt( (x-xQ)**2 + (y-yQ)**2) #Get distance from QSO

            y = int(round(y)) #Cast to integers for indexing
            x = int(round(x))

            #If outside FOV, ignore
            if np.min(rr) > 1: continue
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
                                        a = 3*model_fit.x_stddev,
                                        b = 3*model_fit.y_stddev
                )

                #If this is the central source, add to QSO mask
                if dist < 1:
                    fMask = rr<=fitRad #Set fitting mask for later
                    if subRad!=None: sMask = rr<=subRad #If subtraction radius given, update sMask
                    qso_mask += mask_model(xx, yy) #Add to QSO mask
                else:
                    src_model += model_fit(xx, yy) #Add to source model and mask
                    if rad > 6: src_mask[ rr < rad ] = 1
                    else: src_mask  += mask_model(xx, yy)

        src_mask[src_mask > 0] = 1
        src_mask[qso_mask > 0] = 2 #2 indicates QSO or central source

    else:
        src_model = np.zeros_like(NB)
        src_mask = np.zeros_like(NB)

    #Limit background to annular region (excluding sources)
    bg_mask = (src_mask == 0) & (annulus_mask == 1)

    #Median subtract both
    scval = 3
    if medsub:
        NB -= np.median(sigmaclip(NB[bg_mask == 1].copy(), low=scval, high=scval).clipped)
        WL -= np.median(sigmaclip(WL[bg_mask == 1].copy(), low=scval, high=scval).clipped)

    if h_corr:
        for yi in range(NB.shape[0]):
            NB[yi, :] -= np.median( sigmaclip(NB[yi, src_mask[yi, :]==0 ], low=scval, high=scval).clipped )
            WL[yi, :] -= np.median( sigmaclip(WL[yi, src_mask[yi, :]==0 ], low=scval, high=scval).clipped)


    if v_corr:
        for xi in range(NB.shape[1]):
            NB[:, xi] -= np.median( sigmaclip(NB[src_mask[:, xi]==0, xi], low=scval, high=scval).clipped )
            WL[:, xi] -= np.median( sigmaclip(WL[src_mask[:, xi]==0, xi], low=scval, high=scval).clipped )

    ###
    #### Scale NB Variance to match noise in NB
    ####
    fitter = fitting.LevMarLSQFitter()

    NB_snr = NB/np.sqrt(NB_var)
    NB_n, edges = np.histogram(NB_snr[bg_mask == 1], range=(-10,10), bins=100)
    centers = np.array([ (edges[i]+edges[i+1])/2.0 for i in range(edges.size-1)])
    use_centers = (centers <= 2) & (centers >= -4)
    NB_snrmodel_0 = models.Gaussian1D(amplitude=NB_n.max(), mean=0, stddev=1)
    NB_snrmodel_1 = fitter(NB_snrmodel_0, centers[use_centers], NB_n[use_centers])

    NB_var_rescaleF = (NB_snrmodel_1.stddev.value**2)
    NB_var *= NB_var_rescaleF

    ####
    #### Scale WL Variance to match noise in WL image
    ####
    WL_snr = WL/np.sqrt(WL_var)
    WL_n, WL_edges = np.histogram(WL_snr[bg_mask == 1], range=(-10,10), bins=100)

    WL_snrmodel_0 = models.Gaussian1D(amplitude=WL_n.max(), mean=0, stddev=1)
    WL_snrmodel_1 = fitter(WL_snrmodel_0, centers[use_centers], WL_n[use_centers])

    WL_var_rescaleF = (WL_snrmodel_1.stddev.value**2)
    #WL_var *= WL_var_rescaleF

    ####
    ####
    ####

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(10,3))
        ax1, ax2, ax3 = axes
        ax1.set_aspect('equal')
        ax1.pcolor(WL, cmap=plt.cm.binary)
        ax1.contour(bg_mask, levels=[0.5], colors=['k'])
        ax2.set_title("NB Background SNR")
        ax2.step( centers, NB_n, 'r')
        ax2.plot(centers, NB_snrmodel_1(centers), 'k-')
        ax3.set_title("WL Background SNR")
        ax3.step( centers, WL_n, 'b')
        ax3.set_xlim([-3.5,3.5])
        ax2.set_xlim(ax3.get_xlim())
        ax3.plot(centers, WL_snrmodel_1(centers), 'k-')
        fig.tight_layout()
        fig.show()
        plt.waitforbuttonpress()
        plt.close()

    #If smoothing requested
    if smooth!=None:
        smoothtype = 'box'
        NB = smooth2d(NB, smooth, ktype=smoothtype)
        WL = smooth2d(WL, smooth, ktype=smoothtype)
        NB_var =  smooth2d(NB_var, smooth, ktype=smoothtype, var=True)
        WL_var =  smooth2d(WL_var, smooth, ktype=smoothtype, var=True)

    #Median subtract both
    if medsub:
        NB -= np.median(sigmaclip(NB[bg_mask == 1].copy(), low=2, high=2).clipped)
        WL -= np.median(sigmaclip(WL[bg_mask == 1].copy(), low=2, high=2).clipped)


    #Calculate scaling factor
    scalingFactors = NB[fMask]/WL[fMask]

    scalingFactors_Clipped = sigmaclip(scalingFactors,high=2.5,low=2.5)
    scalingFactors_Mean = np.median(scalingFactors)
    S = scalingFactors_Mean

    #Scale WL image and variance
    WL *= S
    WL_var *= (S**2)


    sMask[WL<=0] = 0 #Do not subtract negative values
    sMask[src_mask == 1] = 0 #Do not subtract over other sources

    #Create NB-S*WL subtraction image
    NB_sub = NB.copy()
    NB_sub[sMask] -=  WL[sMask]

    NB_sub_bgmean = noiseFitMean(NB_sub[bg_mask == 1], plot=plot)
    NB_sub -= NB_sub_bgmean

    NB_sub_var = NB_var.copy()
    NB_sub_var[sMask] += WL_var[sMask]
    NB_sub[src_mask == 1] = 0

    NB_sub[fMask == 1] = 0

    if smooth!=None:
        NB_sub_var_rescaleF = 6.053 #TEMPORARY \
        NB_sub_var *= NB_sub_var_rescaleF

    if plot:
        fig, axes = plt.subplots(1, 2)
        ax1, ax2 = axes
        ax1.pcolor(NB_sub, cmap=plt.cm.binary)
        ax1.contour(bg_mask, levels=[0.5], colors=['r'] )
        ax1.set_aspect('equal')
        ax2.hist(NB_sub_snr_bg, range=(-10,10), bins=100, facecolor='r', alpha=0.75)
        ax2.plot(centers, NB_sub_snrmodel_1(centers), 'k-')
        fig.show()
        plt.waitforbuttonpress()
        plt.close()

    #Return SB map
    return WL, NB, NB_sub, NB_sub_var, src_mask

def noiseFitMean(data, plot=False):


    data_clipped = sigmaclip(data, low=3, high=3).clipped
    x_guess = np.mean(data_clipped)
    s_guess = np.std(data_clipped)
    R = (x_guess-3*s_guess, x_guess+3*s_guess)
    B = 100
    n, edges = np.histogram(data_clipped, range=R, bins=B)
    cens = np.array( [(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])

    A_guess = n.max()*0.75
    fit_inds = (cens < x_guess + 2*s_guess)

    fitter = fitting.LevMarLSQFitter()
    model0 = models.Gaussian1D(amplitude=A_guess, mean=x_guess, stddev=s_guess)
    model1 = fitter(model0, cens[fit_inds], n[fit_inds])

    if plot:
        fig, ax = plt.subplots(1,1)
        ax.hist(data, range=R, bins=B, facecolor='b', alpha=0.5)
        ax.step(cens, n, 'k-')
        ax.plot(cens, model1(cens), 'r-')
        fig.show()
        plt.waitforbuttonpress()
        plt.close()
    fit_mean = (model1.mean.value)
    return fit_mean

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

    #Get kernel
    if ktype=='box': K = Box2DKernel(scale)
    elif ktype=='gaussian': K = Gaussian2DKernel(fwhm2sigma(scale))
    else: raise ValueError("Kernel type ('%s') not found"%ktype)

    #Un-normalize kernel so max value = 1
    K = K.array
    K /= np.max(K)

    #Filter variance by K^2 and intensity data by K
    if var: imgFilt = convolve(img, np.power(np.array(K), 2), normalize_kernel=False)
    else: imgFilt = convolve(img, K, normalize_kernel=False)

    if var: imgFilt /= np.power(np.sum(K), 2) #Divide variance data by sum(K)^2
    else: imgFilt /= np.sum(K) #Divide intensity data by sum(K)

    #Return
    return imgFilt

def get_box_old(fitsIN,paramsIN,boxSize,fill=0):

    wcs2D = WCS(cubes.get_header2d(fitsIN[0].header))

    xC,yC = wcs2D.all_world2pix(paramsIN["RA"],paramsIN["DEC"],0)

    pkpc_per_px = get_pkpc_px(wcs2D,paramsIN["Z"])
    boxSize_px = boxSize/pkpc_per_px

    NBCutout = Cutout2D(fitsIN[0].data,(xC,yC),boxSize_px,wcs2D,mode='partial',fill_value=fill)

    return NBCutout.data
