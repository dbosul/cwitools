from astropy import units as u
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from cwitools import coordinates
from photutils import DAOStarFinder
from scipy.ndimage.filters import generic_filter
from scipy.stats import sigmaclip, tstd
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import os

def psf_subtract_byslice(fits_in, pos, wmasks=[], fit_rad=2, sub_rad=15,
wl_window=150, slice_rad=4, inst="KCWI"):

    #Extract data + meta-data
    cube, hdr = fits_in[0].data, fits_in[0].header
    cube = np.nan_to_num(cube, nan=0, posinf=0, neginf=0)
    z, y, x = cube.shape
    Z, Y, X = np.arange(z), np.arange(y), np.arange(x)
    x0, y0 = pos
    cd3_3 = hdr["CD3_3"]
    wav_axis = coordinates.get_wav_axis(hdr)
    psf_model = np.zeros_like(cube)

    #Create wavelength masked based on input
    zmask = np.ones_like(wav_axis, dtype=bool)
    for (w0, w1) in wmasks:
        zmask[(wav_axis > w0) & (wav_axis < w1)] = 0

    #Create masks for fitting and subtracting PSF
    fit_mask = np.abs(Y - y0) <= fit_rad
    sub_mask = np.abs(Y - y0) <= sub_rad

    #Get minimum and maximum slices for subtraction
    slice_min = max(0, x0 - slice_rad)
    slice_max = min(x - 1, x0 + slice_rad + 1)


    #Loop over wavelength first to minimize repetition of wl-mask calculation
    for j, wav_j in enumerate(wav_axis):

        #Get initial width of white-light bandpass in px
        wl_width_px = wl_window / cd3_3

        #Create initial white-light mask, centered on j with above width
        wl_mask = zmask & (np.abs(Z - j) <= wl_width_px / 2)

        #Grow until minimum number of valid wavelength layers included
        while np.count_nonzero(wl_mask) < wl_window / cd3_3:
            wl_width_px += 2
            wl_mask = zmask & (np.abs(Z - j) <= wl_width_px / 2)

        #Iterate over slices and perform subtraction
        for slice_i in range(slice_min, slice_max):

            #Get 1D profile of current slice/wav and WL profile
            j_prof = cube[j, :, slice_i].copy()

            wl_prof = np.mean(cube[wl_mask, :, slice_i], axis=0)
            wl_prof -= np.median(wl_prof[~sub_mask])

            if np.sum(wl_prof[fit_mask]) / np.std(wl_prof) < 7:
                continue


            #Exclude negative WL pixels from scaling fit
            fit_mask_i = fit_mask & (wl_prof > 0)

            #Calculate scaling factor
            scale_factors = j_prof[fit_mask_i] / wl_prof[fit_mask_i]
            scale_factor_med = np.mean(scale_factors)
            scale_factor = scale_factor_med

            #Create WL model
            wl_model = scale_factor * wl_prof

            psf_model[j, sub_mask, slice_i] += wl_model[sub_mask]

            if 0:
                fig, axes = plt.subplots(2, 1, figsize=(8,8))
                axes[0].plot(Y, j_prof, 'k.-')
                axes[0].plot(Y[fit_mask], j_prof[fit_mask], 'ko')
                axes[0].plot(Y, wl_model, 'r--')
                axes[1].plot(Y, j_prof - wl_model, 'b.-')
                fig.show()
                input("")#plt.waitforbuttonpress()
                plt.close()
    #Subtract PSF model
    cube -= psf_model

    #Return both
    return cube, psf_model


def psf_subtract(inputfits, rmin=1.5, rmax=5.0, reg=None, pos=None,
recenter=True, auto=7, wl_window=200, local_window=0, scalemask=1.0,
zmasks=((0, 0)), zunit='A', verbose=False ):
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

    Examples:

        To subtract point sources from an input cube using a DS9 region file:

        >>> from astropy.io import fits
        >>> from cwitools import psf_subtract
        >>> myregfile = "mysources.reg"
        >>> myfits = fits.open("mydata.fits")
        >>> sub_cube, psf_model, mask_2d = psf_subtract(myfits, reg = myregfile)

        To subtract using automatic source detection with photutils, and a
        source S/N ratio >5:

        >>> sub_cube, psf_model, mask_2d = psf_subtract(myfits, auto = 5)

        Or to subtract a single source from a specific location (x,y)=(21.1,34.6):

        >>> sub_cube, psf_model, mask_2d = psf_subtract(myfits, pos=(21.1, 34.6))

    """

    #Open fits image and extract info
    cube = inputfits[0].data
    header = inputfits[0].header
    w, y, x = cube.shape
    W, Y, X = np.arange(w), np.arange(y), np.arange(x)
    wav = coordinates.get_wav_axis(header)

    #Remove NaN values
    cube = np.nan_to_num(cube,nan=0.0,posinf=0,neginf=0)

    #Create cube for subtracted cube and for model of psf
    psf_cube = np.zeros_like(cube)
    sub_cube = cube.copy()

    #Make mask canvas
    msk2D = np.zeros((y, x))

    #Create white-light image
    maskwav = np.zeros_like(wav, dtype=bool)
    for (z0, z1) in zmasks:

        if zunit == 'A': z0, z1 = coordinates.get_indices(z0, z1, header)
        maskwav[z0:z1] = 1

    wlcube = cube.copy()
    wlcube[maskwav] = 0
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
                    if A < 0 or np.isinf(A) or np.isnan(A): A = 0

                    #Subtract fit from data
                    sub_cube[wi][subPx] -= A*psfImg[subPx]

                    #Add to PSF model
                    psf_cube[wi][subPx] += A*psfImg[subPx]

                #Update WL cube and image after subtracting this source
                wlImg = np.sum(sub_cube[maskwav == 0], axis=0)

        if verbose: pbar.update(1)

    if verbose: pbar.close()

    return sub_cube, psf_cube, msk2D

def bg_subtract(inputfits, method='polyfit', poly_k=1, median_window=31, zmasks=[(0, 0)], zunit='A'):
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

    Returns:
        NumPy.ndarray: Background-subtracted cube
        NumPy.ndarray: Cube containing background model which was subtracted.

    Examples:

        To model the background with 1D polynomials for each spaxel's spectrum,
        using a quadratic polynomial (k=2):

        >>> from cwitools import bg_subtract
        >>> from astropy.io import fits
        >>> myfits = fits.open("mydata.fits")
        >>> bgsub_cube, bgmodel_cube = bg_subtract(myfits, method='polfit', poly_k=2)

    """

    #Load header and data
    header = inputfits[0].header.copy()
    cube = inputfits[0].data.copy()
    W = coordinates.get_wav_axis(header)
    z, y, x = cube.shape
    xySize = cube[0].size
    maskZ = False
    modelC = np.zeros_like(cube)

    #Get empty regions mask
    mask2D = np.sum(cube, axis=0) == 0

    #Convert zmask to pixels if given in angstrom
    useZ = np.ones_like(W, dtype=bool)
    for (z0, z1) in zmasks:
        print(z0,z1)
        if zunit == 'A': z0, z1 = coordinates.get_indices(z0, z1, header)
        useZ[z0:z1] = 0

    #Subtract background by fitting a low-order polynomial
    if method == 'polyfit':

        fitter = fitting.LevMarLSQFitter()
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

        bgModel1 = fitter(bgModel0, W[useZ], medians[useZ])
        for i, wi in enumerate(W):
            cube[i][~mask2D] -= bgModel1(wi)
            modelC[i][~mask2D] = bgModel1(wi)

    #Subtract using simple layer-by-layer median value
    elif method == "median":

        dataclipped = sigmaclip(cube).clipped

        for zi in range(z):
            modelC[zi][mask2D == 0] = np.median(dataclipped[zi][mask2D == 0])
        modelC = medfilt(modelC, kernel_size=(3, 1, 1))
        cube -= modelC

    return cube, modelC
