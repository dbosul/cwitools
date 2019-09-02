"""CWITools Data Analysis Functions"""
from cwitools.libs import cubes

from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.modeling import models, fitting
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import SqrtStretch
from astropy.stats import sigma_clip
from photutils import DAOStarFinder
from scipy import ndimage
from scipy.signal import medfilt
from scipy.stats import tstd
from scipy.ndimage.filters import generic_filter
from scipy.stats import sigmaclip
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pyregion
import sys
import warnings


def rebin(inputfits, xybin=1, zbin=1, vardata=False):
    """Re-bin a data cube along the spatial (x,y) and wavelength (z) axes.

    Args:
        inputfits (astropy FITS object): Input FITS to be rebinned.
        xybin (int): Integer binning factor for x,y axes. (Def: 1)
        zbin (int): Integer binning factor for z axis. (Def: 1)
        vardata (bool): Set to TRUE if rebinning variance data. (Def: True)
        fileExt (str): File extension for output (Def: .binned.fits)

    """


    #Extract useful structures
    data = inputfits[0].data.copy()
    head = inputfits[0].header.copy()

    #Get dimensions & Wav array
    z, y, x = data.shape
    wav = cubes.get_wavaxis(head)

    #Get new sizes
    znew = int(z/zbin)  + 1 if zbin > 1 else z
    ynew = int(y/xybin) + 1 if xybin > 1 else y
    xnew = int(x/xybin) + 1 if xybin > 1 else x

    #Perform wavelenght-binning first, if bin provided
    if zbin > 1:

        #Get new bin size in Angstrom
        zbinSize = zbin*head["CD3_3"]

        #Create new data cube shape
        data_zbinned = np.zeros((znew, y, x))

        #Run through all input wavelength layers and add to new cube
        for zi in range(z): data_zbinned[int(zi/zbin)] += data[zi]

        #Normalize so that units remain as "erg/s/cm2/A"
        if vardata: data_zbinned /= zbin**2
        else: data_zbinned /= zbin

        #Update central reference and pixel scales
        head["CD3_3"] *= zbin
        head["CRPIX3"] /= zbin

    else: data_zbinned = data

    #Perform spatial binning next
    if xybin > 1:

        #Get new shape
        data_xybinned = np.zeros((znew, ynew, xnew))

        #Run through spatial pixels and add
        for yi in range(y):
            for xi in range(x):
                data_xybinned[:, int(yi/xybin), int(xi/xybin)] += data_zbinned[:, yi, xi]

        #
        # No normalization needed for binning spatial pixels.
        # Units remain as 'per pixel' but pixel size changes.
        #

        #Update reference pixel
        head["CRPIX1"] /= float(xybin)
        head["CRPIX2"] /= float(xybin)

        #Update pixel scales
        for key in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]: head[key] *= xybin

    else: data_xybinned = data_zbinned

    binnedFits = fits.HDUList([fits.PrimaryHDU(data_zbinned)])
    binnedFits[0].header = head

    return binnedFits

def psf_subtract(inputfits, rmin=1.5, rmax=5.0, reg=None, pos=None, recenter=True, \
 auto=7, wl_window=200, local_window=0, scalemask=1.0, zmask=(0, 0), zunit='A', verbose=False):
    """Models and subtracts point-sources in a 3D data cube.

    Args:
        inputfits (astrop FITS object): Input data cube/FITS.
        rmin (float): Inner radius, used for fitting PSF.
        rmax (float): Outer radius, used to subtract PSF.
        reg (str): Path to a DS9 region file containing sources to subtract.
        pos (float tuple): (x,y) position of the source to subtract.
        auto (float): SNR above which to automatically detect/subtract sources.
            Note: One of the parameters reg, pos, or auto must be provided.
        wl_window (int): Size of white-light window (in Angstrom) to use.
            This is the window used to form a white-light image centered
            on each wavelength layer. Default: 200A.
        local_window (int): Size of local window (in Angstrom) to use.
            This is the window used around each wavelength layer to form the
            local narrowband image. Default: 0 (i.e. single layer only)
        scalemask (float): Scaling factor for output PSF mask (Default: 1)
        zmask (int tuple): Wavelength region to exclude from white-light images.
        zunit (str): Unit of argument zmask.
            Can be Angstrom ('A') or pixels ('px'). Default: 'A'.
        verbose (bool): Set to True to display progress and information.

    Returns:
        numpy.ndarray: PSF-subtracted data cube
        numpy.ndarray: PSF model cube
        numpy.ndarray: 2D mask of sources

    Raises:
        FileNotFoundError: If region file is not found.

    """

    #Open fits image and extract info
    z0, z1 = zmask
    cube = inputfits[0].data
    header = inputfits[0].header
    w, y, x = cube.shape
    W, Y, X = np.arange(w), np.arange(y), np.arange(x)
    wav = cubes.get_wavaxis(header)

    #Remove NaN values
    cube = np.nan_to_num(cube,nan=0.0,posinf=0,neginf=0)

    #Create cube for subtracted cube and for model of psf
    psf_cube = np.zeros_like(cube)
    sub_cube = cube.copy()

    #Make mask canvas
    msk2D = np.zeros((y, x))

    #Create white-light image
    if zunit == 'A': z0, z1 = cubes.get_indices(z0, z1, header)
    wlcube = cube.copy()
    wlcube[z0:z1] = 0
    wlImg = np.sum(wlcube, axis=0)

    #Get WCS information
    wcs = WCS(header)
    pxScales = proj_plane_pixel_scales(wcs)

    #Get mask of spaxels with no data
    emptyPxMask2D = (np.sum(cube, axis=0) == 0)

    #Convert plate scale to arcseconds
    xScale, yScale = (pxScales[:2]*u.deg).to(u.arcsecond)
    zScale = (pxScales[2]*u.meter).to(u.angstrom)

    #Convert fitting & subtracting radii from arcsecond/Angstrom to pixels
    rmin_px = rmin/xScale.value
    rmax_px = rmax/xScale.value
    delZ_px = int(round(0.5*wl_window/zScale.value))

    #Get fitter for PSF fit
    boxSize = 3*int(round(rmax_px))
    psfFitter = fitting.LevMarLSQFitter()
    psfModel = models.Gaussian2D(amplitude=1, x_mean=boxSize/2, y_mean=boxSize/2)

    #Get box size and indices for fitting
    yy, xx = np.mgrid[:boxSize, :boxSize]

    #Get sources from region file or position input
    sources = []
    if reg != None:

        if verbose: print("Using region file to locate sources.")

        if os.path.isfile(reg): regFile = pyregion.open(reg)
        else:
            raise FileNotFoundError("Region file could not be found: %s"%reg)

        for src in regFile:
            ra, dec, pa = src.coord_list
            xP, yP, wP = wcs.all_world2pix(ra, dec, header["CRVAL3"], 0)
            sources.append((xP, yP))

    elif pos != None:
        if verbose: print("Using provided source position.")
        sources = [pos]
    else:

        if verbose: print("Automatically detecting sources with photutils. (SNR>%.1f)"%auto)

        stddev = np.std(wlImg[wlImg <= 10*np.std(wlImg)])

        #Run source finder
        daofind = DAOStarFinder(fwhm=8.0, threshold=auto*stddev)
        autoSrcs = daofind(wlImg)

        #Get list of peak values
        peaks = list(autoSrcs['peak'])

        #Make list of sources
        sources = []
        for i in range(len(autoSrcs['xcentroid'])):
            sources.append((autoSrcs['xcentroid'][i], autoSrcs['ycentroid'][i]))

        #Sort according to peak value (this will be ascending)
        peaks, sources = zip(*sorted(zip(peaks, sources)))

        #Cast back to list
        sources = list(sources)

        #Reverse to get descending order (brightest sources first)
        sources.reverse()

    if verbose:
        print("Subtracting %i source(s)."%len(sources))
        pbar = tqdm(total=len(sources))

    #Run through sources
    for (xP, yP) in sources:

        #Get meshgrid of distance from P
        YY, XX = np.meshgrid(X-xP, Y-yP)
        RR = np.sqrt(XX**2 + YY**2)

        if np.min(RR) > rmin_px: continue
        else:

            #Get cut-out around source
            psfBox = Cutout2D(wlImg, (xP, yP), (boxSize, boxSize), mode='partial', fill_value=-99)
            psfBox = psfBox.data

            #Get useable spaxels
            fitXY = np.array(psfBox != -99, dtype=int)

            #Run fit
            psfFit = psfFitter(psfModel, yy, xx, psfBox, weights=fitXY)

            #Get sigma/fwhm
            xfwhm, yfwhm = 2.355*psfFit.x_stddev.value, 2.355*psfFit.y_stddev.value

            #We take larger of the two for our purposes
            fwhm = max(xfwhm, yfwhm)

            #Only continue with well-fit, high-snr sources
            if 1 or (psfFitter.fit_info['nfev'] < 100 and fwhm < 10):

                if recenter:
                    yP = psfFit.x_mean.value+yP-boxSize/2
                    xP = psfFit.y_mean.value+xP-boxSize/2

                #Update meshgrid of distance from P
                YY, XX = np.meshgrid(X-xP, Y-yP)
                RR = np.sqrt(XX**2 + YY**2)

                #Get half-width-half-max
                hwhm = fwhm/2.0

                #Add source to mask
                msk2D[RR <= scalemask*hwhm] = 1

                #Get boolean masks for
                fitPx = RR <= rmin_px
                subPx = (RR <= rmax_px) & (~emptyPxMask2D)

                meanRs = []
                #Run through wavelength layers
                for wi in range(w):

                    #Get this wavelength layer and subtract any median residual
                    wl1, wl2 = max(0, wi-local_window), min(w, wi+local_window)+1
                    layer = np.mean(cube[wl1:wl2], axis=0)

                    #Get upper and lower-bounds for creating WL image
                    a = max(0, wi-delZ_px)
                    b = min(w, a+delZ_px)

                    #Create PSF image
                    psfImg = np.sum(wlcube[a:b], axis=0)
                    psfImg[psfImg < 0] = 0

                    scalingFactors = layer[fitPx]/psfImg[fitPx]
                    scalingFactors_Clipped = sigmaclip(scalingFactors, high=3.5, low=3.5)
                    scalingFactors_Mean = np.mean(scalingFactors)
                    A = scalingFactors_Mean
                    if A < 0: A = 0


                    #Subtract fit from data
                    sub_cube[wi][subPx] -= A*psfImg[subPx]

                    #Add to PSF model
                    psf_cube[wi][subPx] += A*psfImg[subPx]

                #Update WL cube and image after subtracting this source
                wlImg = np.sum(sub_cube[:z0], axis=0)+np.sum(sub_cube[z1:], axis=0)

        if verbose: pbar.update(1)

    if verbose: pbar.close()

    return sub_cube, psf_cube, msk2D



def estimate_variance(inputfits, zwindow=10, rescale=True, sigmaclip=4, zmask=(0, 0), fmin=0.9, fmax=10):
    """Estimates the 3D variance cube of an input cube.

    Args:
        cubePath (str): Path to the input cube.
        zWindow (int): Size of z-axis bins to use for 2D variance estimation. Default: 10.
        zmask (int tuple): Wavelength layers to exclude while estimating variance.
        rescale (bool): Set to TRUE to perform layer-by-layer rescaling of 2D variance.
        fMin (float): The minimum rescaling factor (Default 0.9)
        fMax (float): The maximum rescaling factor (Default: 10)
        fileExt (str): The extension to use for the output cube (Default .var.fits)

    Returns:

        NumPy ndarray: Estimated variance cube

    """

    cube = inputfits[0].data
    z0, z1 = zmask
    dz = zwindow

    #Output warning
    if z1-z0 >= dz:
        warnings.warn("""Your z-mask is large relative to your zwindow size.\
        \nVariance estimate may be unreliable.""")

    #Parse boolean input
    rescale = True if rescale == "True" else False

    #Run sigma-clip if set
    if sigmaclip > 0: cube = sigma_clip(cube, sigma=sigmaclip).data

    #Make first estimate by binning data
    varcube = np.zeros_like(cube)
    i = 0
    a, b = (i*dz), (i+1)*dz
    while b < cube.shape[0]:
        varcube[a:b] = np.var(cube[a:b], axis=0)
        i += 1
        a, b = (i*dz), (i+1)*dz
    varcube[a:] = np.var(cube[a:], axis=0)

    #Adjust first estimate by rescaling, if set to do so
    if rescale:
        for wi in range(len(varcube)):

            sig = np.sqrt(varcube[wi])

            useXY = sig > 0

            varNorm = np.var(cube[wi][useXY]/sig[useXY])

            #Normalize so that variance of layer as a whole is ~1
            #
            # Note: this assumes most of the 3D field is empty of real signal.
            # Z and XY Masks should be supplied if that is not the case
            #

            rsFactor = (1/varNorm)

            rsFactor = max(rsFactor, fmin)
            rsFactor = min(rsFactor, fmax)

            varcube[wi] *= rsFactor

    return varcube


def bg_subtract(inputfits, method='polyfit', poly_k=1, median_window=31, zmask=(0, 0), zunit='A'):
    """
    Subtracts extended continuum emission / scattered light from a cube

    Args:
        cubePath (str): Path to the data cube to be subtracted.
        method (str): Which method to use to model background
            'polyfit': Fits polynomial to the spectrum in each spaxel (default.)
            'median': Subtract the spatial median of each wavelength layer.
            'medfilt': Model spectrum in each spaxel by median filtering it.
            'noiseFit': Model noise in each z-layer and subtract mean.
        poly_k (int): The degree of polynomial to use for background modeling.
        median_window (int): The filter window size to use if median filtering.
        zmask (int tuple): Wavelength region to mask, given as tuple of indices.
        zunit (str): If using zmask, indices are given in these units.
            'A': Angstrom (default)
            'px': pixels
        saveModel (bool): Set to TRUE to save background model cube.
        fileExt (str): File extension to use for output (Default: .bs.fits)

    """

    #Load header and data
    header = inputfits[0].header.copy()
    cube = inputfits[0].data.copy()
    W = cubes.get_wavaxis(header)
    z, y, x = cube.shape
    xySize = cube[0].size
    useZ = np.ones_like(W, dtype=bool)
    maskZ = False
    modelC = np.zeros_like(cube)

    #Get empty regions mask
    mask2D = np.sum(cube, axis=0) == 0

    #Convert zmask to pixels if given in angstrom
    z0, z1 = zmask
    if zunit == 'A': z0, z1 = cubes.get_indices(z0, z1, header)

    #Subtract background by fitting a low-order polynomial
    if method == 'polyfit':

        useZ[z0:z1] = 0
        fitter = fitting.LinearLSQFitter()
        pModel0 = models.Polynomial1D(degree=poly_k)

        #Track progress % using n
        n = 0

        #Run through spaxels and subtract low-order polynomial
        for yi in range(y):
            for xi in range(x):

                n += 1
                p = 100*float(n)/xySize
                sys.stdout.write('%5.2f percent complete\r'%p)
                sys.stdout.flush()

                #Extract spectrum at this location
                spectrum = cube[:, yi, xi].copy()

                #Fit polynomial to data, ignoring masked pixels
                pModel1 = fitter(pModel0, W[useZ], spectrum[useZ])

                #Get background model
                bgModel = pModel1(W)

                if mask2D[yi, xi] == 0:

                    cube[:, yi, xi] -= bgModel

                    #Add to model
                    modelC[:, yi, xi] += bgModel

    #Subtract background by estimating it with a median filter
    elif method == 'medfilt':

        #Get +/- 5px windows around masked region, if mask is set
        if z1 > 0:

            #Get size of window region used to interpolate (minimum 5 to get median)
            nw = max(5, (z1-z0))

            #Get left and right index of window regions
            a = max(0, z0-nw)
            b = min(z, z1+nw)

            #Get two z mid-points which we will use for calculating line slope/intercept
            ZA = (a+z0)/2.0
            ZB = (b+z1)/2.0

            maskZ = True

        #Track progress % using n
        n = 0

        for yi in range(y):
            for xi in range(x):
                n += 1
                p = 100*float(n)/xySize
                sys.stdout.write('%5.2f percent complete\r'%p)
                sys.stdout.flush()

                #Extract spectrum at this location
                spectrum = cube[:, yi, xi].copy()

                #Fill in masked region with smooth linear interpolation
                if maskZ:

                    #Calculate slope and intercept
                    YA = np.mean(spectrum[a:z0]) if (z0-a) < 5 else np.median(spectrum[a:z0])
                    YB = np.mean(spectrum[z1:b]) if (b-z1) < 5 else np.median(spectrum[z1:b])
                    m = (YB-YA)/(ZB-ZA)
                    c = YA - m*ZA

                    #Get domain for masked pixels
                    ZZ = np.arange(z0, z1+1)

                    #Apply mask
                    spectrum[z0:z1+1] = m*ZZ + c

                #Get median filtered spectrum as background model
                bgModel = generic_filter(spectrum, np.median, size=median_window, mode='reflect')

                if mask2D[yi, xi] == 0:

                    #Subtract from data
                    cube[:, yi, xi] -= bgModel

                    #Add to model
                    modelC[:, yi, xi] += bgModel

    #Subtract layer-by-layer by fitting noise profile
    elif method == 'noiseFit':
        fitter = fitting.SimplexLSQFitter()
        medians = []
        for zi in range(z):

            #Extract layer
            layer = cube[zi]
            layerNonZ = layer[~mask2D]

            #Get median
            median = np.median(layerNonZ)
            stddev = np.std(layerNonZ)
            trimmed_stddev = tstd(layerNonZ, limits=(-3*stddev, 3*stddev))
            trimmed_median = np.median(layerNonZ[np.abs(layerNonZ-median) < 3*trimmed_stddev])

            medians.append(trimmed_median)
        medians = np.array(medians)
        bgModel0 = models.Polynomial1D(degree=2)
        useZ[z0:z1] = 0
        bgModel1 = fitter(bgModel0, W[useZ], medians[useZ])
        for i, wi in enumerate(W):
            cube[i][~mask2D] -= bgModel1(wi)
            modelC[i][~mask2D] = bgModel1(wi)

    #Subtract using simple layer-by-layer median value
    elif method == "median":

        sigclip = SigmaClip(sigma=2)
        dataclipped = sigclip(cube)

        for zi in range(z):
            modelC[zi][mask2D == 0] = np.median(dataclipped[zi][mask2D == 0])
        modelC = medfilt(modelC, kernel_size=(3, 1, 1))
        cube -= modelC

    return cube, modelC
