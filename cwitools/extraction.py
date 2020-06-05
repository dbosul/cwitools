"""Tools for extracting extended emission from a cube."""
from astropy import units as u
from astropy import convolution
from astropy.cosmology import WMAP9
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from cwitools import coordinates, utils
from cwitools.modeling import sigma2fwhm, fwhm2sigma
from photutils import DAOStarFinder
from scipy.ndimage.filters import generic_filter
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import medfilt
from scipy.ndimage import convolve as sc_ndi_convolve
from scipy.stats import sigmaclip, tstd
from skimage import measure
from tqdm import tqdm

import numpy as np
import os
import pyregion
import sys
import warnings

def apply_mask(data, mask, label=0, fill=0):
    """Apply a binary or label mask to data.

    Args:
        data (numpy.ndarray): The data to be masked
        mask (numpy.ndarray): The mask to apply
        label (int): The mask label to isolate. Default is 0, meaning all
            non-zero pixels are masked. If mask_in is an object mask containing
            IDs, then label=3 would mask all pixels where mask is NOT 3.
        fill (float): The value to replace masked pixels with.

    Returns:
        numpy.ndarray: The masked data
    """
    data_masked = data.copy()

    #Check shape
    if data.shape == mask.shape:
        data_masked[ mask==1 ] = fill
    elif mask.shape == data[0].shape:
        for zi in range(data.shape[0]):
            data_masked[zi][mask==1] = fill
    else:
        raise ValueError("Mask should either match dimensions of data or data[0]")

    #Return masked data
    return data_masked

def detect_lines(obj_fits, lines=None, z = 0, dv=500):
    """Associate detected 3D objects with known emission lines.

    Args:
        obj_fits (HDU or HDUList): The input 3D object mask.
        lines (float list): Optional list of rest-frame emission lines to compare
            against, in units of Angstrom. Over-rides default line list.
        z (float): The redshift of the emission.
        dv (float): The velocity window of each line, in km/s,  within
            which objects are considered to be associated. (+/- dv)

    Returns:
        dict: A dictionary of the format {<line>:<obj_ids>} where
            <line> is a given input line, and <obj_ids> is a list of integer
            labels for the objects.
    """
    hdu = utils.extractHDU(obj_fits)
    obj_mask, header = hdu.data, hdu.header
    wav_axis = coordinates.get_wav_axis(header)

    if lines is None:
        line_data = utils.get_neblines(wav_axis[0], wav_axis[-1], z)
        labels, lines = line_data['ION'], line_data['WAV']
    else:
        labels = ["custom{0} {1}".format(i, l) for i, l in enumerate(lines)]

    candidates = {label:[] for label in labels}

    zmask = np.zeros_like(wav_axis, dtype=int)

    #Calculate windows
    for i, line in enumerate(lines):
        wav_lo = line * (1 - dv/3e5)
        wav_hi = line * (1 + dv/3e5)
        zmask_line = (wav_axis > wav_lo) & (wav_axis < wav_hi)
        candidate_mask = obj_mask.copy()
        candidate_mask[~zmask_line] = 0
        candidate_ids = np.unique(candidate_mask)
        for cid in candidate_ids:
            if cid > 0:
                candidates[labels[i]].append(cid)

    #Remove empty keys
    empty_keys = []
    for key, item in candidates.items():
        if len(item) == 0:
            empty_keys.append(key)
    for key in empty_keys:
        del candidates[key]

    #Return the dictionary
    return candidates


def cutout(fits_in, pos, box_size, redshift=None, fill=0, unit='px',
postype='img', cosmo=WMAP9):
    """Extract a spatial box around a central position from 2D or 3D data.

    Returned data has same dimensions as input data. Return type (HDU/HDUList)
    also matches input type. First HDU is used if input is HDUList.

    Args:
        fits_in (astropy HDU or HDUList): HDU or HDUList with 2D or 3D data.
        pos (float tuple): Center of cutout, given as (axis0, axis1) coordinate
            by default (if postype is set to'image') or as an (RA,DEC) tuple, if
            postype is set to 'radec'
        box_size (float): The size of the box, in units determined by `unit'.
        redshift (float): Cosmological redshift of the source. Required to get
            conversion to units of kiloparsec.
        fill (string): The fill value for box regions outside the image bounds.
            Default: 0.
        unit (str): The unit of the box_size argument
            'px' - pixels
            'arcsec' - arcseconds
            'pkpc' - proper kiloparsecs (requires redshift)
            'ckpc' - comoving kiloparsecs (requires redshift)
        postype (str): The type of coordinate given for the 'pos' argument.
            'radec' - a tuple of (RA, DEC) coordinates, in decimal degrees
            'image' - a tuple of image coordinates, in pixels
    Returns:
        HDU or HDUList The FITS with cutout data and header.

    Examples:

        To extract a 2D region of size 250x250 pkpc^2 around the QSO from a
        pseudo-Narrowband image:

        >>> from cwitools import imaging
        >>> from cwitools import parameters
        >>> from astropy.io import fits
        >>> qso_nb_fits = fits.open("QSO_123.fits")
        >>> target_params = parameters.load_params("QSO_123.param")
        >>> qso_cutout = extraction.get_cutout(qso_nb_fits, target_params, 250)

        This method assumes a 1:1 aspect ratio for the spatial axes of the
        input.
    """
    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data, hdu.header

    #Get 2D WCS information from cube regardless of 2D or 3D input
    if header["NAXIS"] == 2:
        header2d = header
        data2d = data.copy()
    elif header["NAXIS"] == 3:
        header2d = coordinates.get_header2d(fits_in[0].header)
        data2d = np.sum(data, axis=0)
    else:
        raise ValueError("2D or 3D input only for get_cutout.")

    #If RA/DEC position given, convert to image coordinates
    if postype == 'radec':
        ra, dec = pos
        wcs2d = WCS(header2d)
        pos = tuple(float(x) for x in wcs2d.all_world2pix(pos[0], pos[1], 0))
    elif postype == 'image':
        ra, dec = tuple(float(x) for x in wcs2d.all_pix2world(pos[0], pos[1], 0))
    else:
        raise ValueError("postype argument must be 'image' or 'radec'")


    #Get box size, either as scalar or angular Astropy quantity
    if unit in ['pkpc', 'ckpc']:
        ktype = 'proper' if unit == 'pkpc' else 'comoving'
        kpc_per_px = coordinates.get_kpc_per_px(header,
            redshift=redshift,
            type=ktype,
            cosmo=cosmo
        )
        box_size = box_size / kpc_per_px
    elif unit == 'arcsec':
        box_size = box_size * u.arcsec #Cutout2D accepts angular Quantities
    elif unit != 'px':
        raise ValueError("Unit must be px, arcsec, pkpc or ckpc.")

    #Create modified fits and update spatial axes WCS
    fits_out = fits_in.copy()
    if header["NAXIS"] == 2:
        cutout = Cutout2D(fits_in[0].data, pos, box_size, wcs2d,
            mode='partial',
            fill_value=fill
        )
        fits_out[0].data = cutout.data
    else:
        new_cube = []
        for i in range(len(fits_in[0].data)):
            layer_cutout = Cutout2D(fits_in[0].data[i], pos, box_size, wcs2d,
                mode='partial',
                fill_value=fill
            )
            new_cube.append(layer_cutout.data)
        fits_out[0].data = np.array(new_cube)

    #Update spatial axes of WCS
    fits_out[0].header["CRVAL1"] = ra
    fits_out[0].header["CRVAL2"] = dec
    fits_out[0].header["CRPIX1"] = pos[0]
    fits_out[0].header["CRPIX2"] = pos[1]
    fits_out[0].header["NAXIS1"] = fits_out[0].data.shape[2]
    fits_out[0].header["NAXIS2"] = fits_out[0].data.shape[1]

    #Return
    return fits_out

def reg2mask(fits_in, reg):
    """Convert a DS9 region file into a 2D binary mask of sources.

    Return type (HDU or HDUList) will match input type. Mask is always 2D.

    Args:
        fits_in (HDU or HDUList): HDU or HDUList with 2D/3D data.
        reg (string): The path to the DS9 region file

    Returns:
        HDU/HDUList: A 2D mask with source regions labelled sequentially.

    """
    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data, hdu.header

    ndims = len(data.shape)
    if ndims == 3:
        image = np.sum(data, axis=0)
        header2d = coordinates.get_header2d(header)
    elif ndims == 2:
        image = data.copy()
        header2d = header
    else:
        raise ValueError("Input data must be 2D or 3D")

    if os.path.isfile(reg):
        reg_wcs = pyregion.open(reg) #World coordinates
        reg_img = reg_wcs.as_imagecoord(header=header2d) #Image coordinates

    else:
        raise FileNotFoundError("%s does not exist." % reg)

    #Mask of continuum (foreground) sources
    src_mask = np.zeros_like(image)

    #Run through sources
    yy, xx = np.indices(src_mask.shape)
    for i, src in enumerate(reg_img):
        x, y, rad = src.coord_list
        x -= 1
        y -= 1
        rr = np.sqrt((xx - x)**2 + (yy - y)**2)
        src_mask[rr <= rad] = 1

    hdu_out = utils.matchHDUType(fits_in, src_mask, header2d)
    return hdu_out


def psf_sub(inputfits, pos, fit_rad=1.5, sub_rad=5.0, wl_window=200,
wmasks=[], recenter=True, recenter_rad=5, var_cube=[], maskpsf=False):

    """Models and subtracts a single point-source in a 3D data cube.

    Args:
        inputfits (astrop FITS object): Input data cube/FITS.
        fit_rad (float): Inner radius, in arcsec, used for fitting PSF.
        sub_rad (float): Outer radius, in arcsec, used to subtract PSF.
        pos (float tuple): Position of the source to subtract in image coords.
        recenter (bool): Recenter the input (x, y) using the centroid within a
            box of size recenter_box, arcseconds.
        recenter_rad(float): Radius of circle used to recenter PSF, in arcsec.
        wl_window (int): Size of white-light window (in Angstrom) to use.
            This is the window used to form a white-light image centered
            on each wavelength layer. Default: 200A.
        wmasks (list): List of wavelength tuples to exclude when making
            white-light images. Use to exclude nebular emission or sky lines.
        var_cube (numpy.ndarray): Variance cube associated with input. Optional.
            Method returns propagated variance if given.

    Returns:
        numpy.ndarray: PSF-subtracted data cube
        numpy.ndarray: PSF model cube
        numpy.ndarray: (if var_cube given) Propagated variance cube

    """

    #Open fits image and extract info
    cube = inputfits[0].data
    header = inputfits[0].header
    z, y, x = cube.shape
    Z, Y, X = np.arange(z), np.arange(y), np.arange(x)
    wav = coordinates.get_wav_axis(header)
    cd3_3 = header["CD3_3"]
    usevar = (var_cube != [])

    #Get plate scales in arcseconds and Angstrom
    rr_arcsec = coordinates.get_rgrid(inputfits, pos, unit='arcsec')

    #Convert WL window size from Ang to px
    wl_window_px = int(round(wl_window / cd3_3))

    #Remove NaN values
    cube = np.nan_to_num(cube,nan=0.0,posinf=0,neginf=0)

    #Create cube for subtracted cube and for model of psf
    psf_cube = np.zeros_like(cube)

    #Make mask canvas
    msk2D = np.zeros((y, x))

    #Create white-light image
    zmask = np.ones_like(wav, dtype=bool)
    for (w0, w1) in wmasks:
        zmask[(wav >= w0) & (wav <= w1)] = 0
    nzmax = np.count_nonzero(zmask)

    #Create WL image
    wl_img = np.sum(cube[zmask], axis=0)

    #Reposition source if requested
    if recenter:
        recenter_img = wl_img.copy()
        recenter_img[rr_arcsec > recenter_rad] = 0
        pos = center_of_mass(recenter_img)
        rr_arcsec = coordinates.get_rgrid(inputfits, pos, unit='arcsec')


    #Get boolean masks for
    fit_mask = (rr_arcsec <= fit_rad)
    sub_mask = (rr_arcsec <= sub_rad)

    #Run through wavelength layers
    for i, wav_i in enumerate(wav):

        #Get initial width of white-light bandpass in px
        wl_width_px = wl_window / cd3_3

        #Create initial white-light mask, centered on j with above width
        wl_mask = zmask & (np.abs(Z - i) <= wl_width_px / 2)

        #Grow mask until minimum number of valid wavelength layers is included
        while np.count_nonzero(wl_mask) < min(nzmax, wl_window / cd3_3):
            wl_width_px += 2
            wl_mask = zmask & (np.abs(Z - i) <= wl_width_px / 2)

        #Get current layer and do median subtraction (after sigclipping)
        layer_i = cube[i].copy()



        #Get white-light image and do the same thing
        N_wl = np.count_nonzero(wl_mask)
        wlimg_i = np.sum(cube[wl_mask], axis=0) / N_wl

        #Try to remove any elevated background levels
        layer_i -= np.median(sigmaclip(layer_i, low=3, high=3).clipped)
        wlimg_i -= np.median(sigmaclip(wlimg_i, low=3, high=3).clipped)

        #Calculate scaling factors
        sfactors = layer_i[fit_mask] / wlimg_i[fit_mask]

        #Get scaling factor, A
        A = np.median(sfactors)

        #Set to zero for bad values
        if A < 0 or np.isinf(A) or np.isnan(A): A = 0

        #Add to PSF model
        psf_cube[i][sub_mask] += A * wlimg_i[sub_mask]

        if usevar:
            wlimg_i_var = np.sum(var_cube[wl_mask], axis=0) / N_wl**2
            var_cube[i][sub_mask] += (A**2) * wlimg_i_var[sub_mask]


    #Subtract 3D PSF model
    sub_cube = cube - psf_cube

    if maskpsf:
        var_img = np.var(sub_cube, axis=0)
        var_mean = np.mean(var_img)
        var_std = np.std(var_img)
        var_mask = ((var_img - var_mean) > 4*var_std) & sub_mask

        sub_cube = sub_cube.T
        sub_cube[fit_mask.T] = 0
        sub_cube[var_mask.T] = 0
        sub_cube = sub_cube.T



    #Return subtracted data alongside model
    if usevar:
        return sub_cube, psf_cube, var_cube
    else:
        return sub_cube, psf_cube

def psf_sub_all(inputfits, fit_rad=1.5, sub_rad=5.0, reg=None, pos=None,
recenter=True, auto=7, wl_window=200, wmasks=[], slice_axis=2, method='2d',
 var_cube=[], maskpsf=False):

    """Models and subtracts multiple point-sources in a 3D data cube.

    Args:
        inputfits (astrop FITS object): Input data cube/FITS.
        fit_rad (float): Inner radius, in arcsec, used for fitting PSF.
        sub_rad (float): Outer radius, in arcsec, used to subtract PSF.
        reg (str): Path to a DS9 region file containing sources to subtract.
        pos (float tuple): Position of the source to subtract.
        auto (float): SNR above which to automatically detect/subtract sources.
            Note: One of the parameters reg, pos, or auto must be provided.
        wl_window (int): Size of white-light window (in Angstrom) to use.
            This is the window used to form a white-light image centered
            on each wavelength layer. Default: 200A.
        wmasks (int tuple): Wavelength regions to exclude from white-light images.
        var_cube (numpy.ndarray): Variance cube associated with input. Optional.
            Method returns propagated variance if given.

    Returns:
        numpy.ndarray: PSF-subtracted data cube
        numpy.ndarray: PSF model cube
        numpy.ndarray: (if var_cube given) Propagated variance cube


    Examples:

        To subtract point sources from an input cube using a DS9 region file:

        >>> from astropy.io import fits
        >>> from cwitools import psf_subtract
        >>> myregfile = "mysources.reg"
        >>> myfits = fits.open("mydata.fits")
        >>> sub_cube, psf_model = psf_subtract(myfits, reg = myregfile)

        To subtract using automatic source detection with photutils, and a
        source S/N ratio >5:

        >>> sub_cube, psf_model = psf_subtract(myfits, auto = 5)

        Or to subtract a single source from a specific location (x,y)=(21.1,34.6):

        >>> sub_cube, psf_model = psf_subtract(myfits, pos=(21.1, 34.6))

    """
    #Open fits image and extract info
    cube = inputfits[0].data
    header = inputfits[0].header
    w, y, x = cube.shape
    W, Y, X = np.arange(w), np.arange(y), np.arange(x)
    wav = coordinates.get_wav_axis(header)

    usevar = var_cube != []

    #Get WCS information
    wcs = WCS(header)
    pxScales = proj_plane_pixel_scales(wcs)

    #Remove NaN values
    cube = np.nan_to_num(cube,nan=0.0,posinf=0,neginf=0)

    #Create cube for subtracted cube and for model of psf
    psf_cube = np.zeros_like(cube)
    sub_cube = cube.copy()

    #Make mask canvas
    msk2D = np.zeros((y, x))

    #Create wavelength mask for white-light image
    zmask = np.ones_like(wav, dtype=bool)
    for (w0, w1) in wmasks:
        zmask[(wav >= w0) & (wav <= w1)] = 0

    #Create white-light image
    wlImg = np.sum(cube[zmask], axis=0)

    #Get list of sources, method depending on input
    sources = []

    #If a single source is given, use that.
    if pos != None:
        sources = [pos]

    #If region file is given
    elif reg != None:

        if os.path.isfile(reg):
            regFile = pyregion.open(reg)
        else:
            raise FileNotFoundError("Region file could not be found: %s"%reg)

        for src in regFile:
            ra, dec, pa = src.coord_list
            xP, yP, wP = wcs.all_world2pix(ra, dec, header["CRVAL3"], 0)
            xP = float(xP)
            yP = float(yP)
            sources.append((yP, xP))

    #Otherwise, use the automatic method
    else:

        #Get standard deviation in WL image (sigmaclip to remove bright sources)
        stddev = np.std(sigmaclip(wlImg, low=3, high=3).clipped)

        #Run source finder
        daofind = DAOStarFinder(fwhm=8.0, threshold=auto * stddev)
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

    #Run through sources
    psf_model = np.zeros_like(cube)
    sub_cube = cube.copy()

    for pos in sources:

        res = psf_sub(inputfits,
            pos = pos,
            fit_rad = fit_rad,
            sub_rad = sub_rad,
            wl_window = wl_window,
            wmasks = wmasks,
            var_cube = var_cube,
            maskpsf = maskpsf
        )

        if usevar:
            sub_cube, model_P, var_cube = res

        else:
            sub_cube, model_P = res

        #Update FITS data and model cube
        inputfits[0].data = sub_cube
        psf_model += model_P

    if usevar:
        return sub_cube, psf_model, var_cube
    else:
        return sub_cube, psf_model

def bg_sub(inputfits, method='polyfit', poly_k=1, median_window=31, wmasks=[]):

    """Subtracts extended continuum emission / scattered light from a cube

    Args:
        inputfits (Astropy HDUList): FITS object to be subtracted
        method (str): Which method to use to model background
            'polyfit': Fits polynomial to the spectrum in each spaxel (default.)
            'median': Subtract the spatial median of each wavelength layer.
            'medfilt': Model spectrum in each spaxel by median filtering it.
            'noiseFit': Model noise in each z-layer and subtract mean.
        poly_k (int): The degree of polynomial to use for background modeling.
        median_window (int): The filter window size to use if median filtering.
        wmasks (int tuple): Wavelength regions to exclude from white-light images.
        saveModel (bool): Set to TRUE to save background model cube.
        fileExt (str): File extension to use for output (Default: .bs.fits)

    Returns:
        NumPy.ndarray: Background-subtracted cube
        NumPy.ndarray: Cube containing background model which was subtracted.

    """

    #Load header and data
    header = inputfits[0].header.copy()
    cube = inputfits[0].data.copy()
    varcube = np.zeros_like(cube)

    W = coordinates.get_wav_axis(header)
    z, y, x = cube.shape
    xySize = cube[0].size
    maskZ = False
    modelC = np.zeros_like(cube)

    #Get empty regions mask
    mask2D = np.sum(cube, axis=0) == 0

    #Create wavelength mask for white-light image
    zmask = np.ones_like(W, dtype=bool)
    for (w0, w1) in wmasks:
        zmask[(W >= w0) & (W <= w1)] = 0

    #Subtract background by fitting a low-order polynomial
    if method == 'polyfit':

        #Track progress % using n
        n = 0

        #Run through spaxels and subtract low-order polynomial
        for yi in tqdm(range(y)):
            for xi in range(x):

                n += 1

                #Extract spectrum at this location
                spectrum = cube[:, yi, xi].copy()

                coeff, covar = np.polyfit(W[zmask], spectrum[zmask], poly_k,
                    full=False,
                    cov=True
                )

                polymodel = np.poly1d(coeff)

                #Get background model
                bgModel = polymodel(W)

                if mask2D[yi, xi] == 0:

                    cube[:, yi, xi] -= bgModel

                    #Add to model
                    modelC[:, yi, xi] += bgModel

                    for m in range(covar.shape[0]):
                        var_m = 0
                        for l in range(covar.shape[1]):
                            var_m += np.power(W, poly_k - l) * covar[l, m] / np.sqrt(covar[m, m])
                        varcube[:, yi, xi] += var_m**2

    #Subtract background by estimating it with a median filter
    elif method == 'medfilt':


        if np.count_nonzero(zmask) > 0:
            warnings.warn("Wavelength masking not yet included in median filter method. Mask will not be applied.")

        #Get median filtered spectrum as background model
        bgModel = medfilt(cube, kernel_size=(median_window, 1, 1))

        bgModel_T = bgModel.T
        bgModel_T[mask2D.T] = 0
        bgModel = bgModel_T.T

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

        bgModel1 = fitter(bgModel0, W[zmask], medians[zmask])
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

    return cube, modelC, varcube

def smooth_cube_wavelength(data, scale, ktype='gaussian', var=False):
    """Smooth 3D data spatially by a specified 2D kernel.

    Args:
        data (numpy.ndarray): The input data to be smoothed.
        scale (float): The smoothing scale.
            For a gaussian kernel, this is full-width at half-maximum (FWHM)
            For a box kernel, this is the width of the box.
        ktype (str): The kernel type ('gaussian' or 'box')
        var (bool): Set to TRUE when smoothing variance data.

    Returns:
        numpy.ndarray: The smoothed data cube.

    """
    #Make copy - do not modify input cube directly
    data_copy = data.copy()


    if ktype=='box':
        kernel = convolution.Box1DKernel(scale)
    elif ktype=='gaussian':
        sigma = fwhm2sigma(scale)
        kernel = convolution.Gaussian1DKernel(sigma)
    else:
        err = "No kernel type '%s' for %iD smoothing" % (ktype, naxes)
        raise ValueError(err)

    kernel = np.array([[kernel.array]]).T

    if var:
        kernel = np.power(kernel, 2)

    data_copy = sc_ndi_convolve(data, kernel)
    return data_copy

def smooth_cube_spatial(data, scale, ktype='gaussian', var=False):
    """Smooth 3D data spatially by a specified 2D kernel.

    Args:
        data (numpy.ndarray): The input data to be smoothed.
        scale (float): The smoothing scale.
            For a gaussian kernel, this is full-width at half-maximum (FWHM)
            For a box kernel, this is the width of the box.
        ktype (str): The kernel type ('gaussian' or 'box')
        var (bool): Set to TRUE when smoothing variance data.

    Returns:
        numpy.ndarray: The smoothed data cube.

    """
    #Make copy - do not modify input cube directly
    data_copy = data.copy()


    if ktype=='box':
        kernel = convolution.Box2DKernel(scale)
    elif ktype=='gaussian':
        sigma = fwhm2sigma(scale)
        kernel = convolution.Gaussian2DKernel(sigma)
    else:
        err = "No kernel type '%s' for %iD smoothing" % (ktype, naxes)
        raise ValueError(err)

    kernel = np.array([kernel.array])

    if var:
        kernel = np.power(kernel, 2)

    data_copy = sc_ndi_convolve(data, kernel)
    return data_copy


def smooth_nd(data, scale, axes=None, ktype='gaussian', var=False):
    """Smooth along all/any axes of a data cube with a box or gaussian kernel.

    Args:
        data (numpy.ndarray): The input data to be smoothed.
        scale (float): The smoothing scale.
            For a gaussian kernel, this is full-width at half-maximum (FWHM)
            For a box kernel, this is the width of the box.
        axes (int tuple): The axes to smooth along. Default is all input axes.
        ktype (str): The kernel type ('gaussian' or 'box')
        var (bool): Set to TRUE when smoothing variance data.

    Returns:
        numpy.ndarray: The smoothed data cube.

    """
    #Make copy - do not modify input cube directly
    data_copy = data.copy()

    if axes == None:
        axes = range(len(data.shape))

    axes = np.array(axes)
    naxes = len(axes)
    ndims = len(data.shape)

    if naxes > ndims or np.any(axes >= ndims):
        raise ValueError("Requested axis greater than dimensions of data.")

    if ndims < 1 or ndims > 3:
        raise ValueError("smooth_nd only works for 1-3 dimensional data.")

    elif ndims == 1 or ndims == 3:

        if ktype=='box':
            kernel = convolution.Box1DKernel(scale)
        elif ktype=='gaussian':
            sigma = fwhm2sigma(scale)
            kernel = convolution.Gaussian1DKernel(sigma)
        else:
            err = "No kernel type '%s' for %iD smoothing" % (ktype, naxes)
            raise ValueError(err)

        kernel = np.power(np.array(kernel), 2) if var else np.array(kernel)

        for a in axes:
            data_copy = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode='same'),
                axis=a,
                arr=data_copy.copy()
            )

        return data_copy

    else:

        if ktype == 'box':
            kernel = convolution.Box2DKernel(scale)
        elif ktype == 'gaussian':
            sigma = fwhm2sigma(scale)
            kernel = convolution.Gaussian2DKernel(sigma)
        else:
            err = "No kernel type '%s' for %iD smoothing" % (ktype, naxes)
            raise ValueError(err)

        kernel = np.power(np.array(kernel), 2) if var else np.array(kernel)
        data_copy = convolution.convolve(data_copy, kernel)

        return data_copy

def obj2binary(obj_mask, obj_id):
    """Get a binary mask of specific objects in a labelled object mask.

    Args:
        obj_mask (numpy.ndarray): Data cube containing labelled regions.
        obj_id (int or list): Object ID or list of object IDs to include.

    Returns:
        numpy.ndarray: The binary mask, where 1 = object, and 0 = background.

    """
    #Create 3D mask from object cube and IDs
    bin_cube = np.zeros_like(obj_mask, dtype=bool)

    if type(obj_id) == int:
        bin_cube = obj_mask == obj_id
    elif type(obj_id) == list and np.all(np.array([type(x) for x in obj_id]) == int):
        bin_cube = np.zeros_like(obj_mask, dtype=bool)
        for oid in obj_id:
            bin_cube[obj_mask == oid] = 1
    else:
        raise TypeError("obj_id must be an integer or list of integers.")
    return bin_cube

def segment(fits_in, var, snrmin=3, includes=None, excludes=None, nmin=10, pad=0):
    """Segment cube into 3D regions above a threshold.

    Args:
        fits_in (NumPy.ndarray): The input data.
        var (NumPy.ndarray): The input variance
        snrmin (float): The minimum SNR for detection
        nmin (int): The minimum 3D object size, in voxels.
        includes (list): List of int tuples indicating which wavelength ranges
            to include, in units of Angstrom. e.g. [(4100,4200), (4350,4400)]
        excludes (list): List of tuples indicating which wavelength ranges to
            exclude from segmentation process.
        pad (int): Number of pixels on xy axes to ignore, useful for excluding
            edge artifacts,
    Returns:
        numpy.ndarray: An object mask with labelled regions

    """
    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data.copy(), hdu.header

    #Create wavelength masked based on input
    wav_axis = coordinates.get_wav_axis(header)

    #Use all indices if no mask ranges given
    if includes is None or includes == []:
        include_mask = np.ones_like(wav_axis, dtype=bool)
    else:
        include_mask = np.zeros_like(wav_axis, dtype=bool)
        for (w0, w1) in includes:
            include_mask[(wav_axis > w0) & (wav_axis < w1)] = 1

    exclude_mask = np.zeros_like(wav_axis, dtype=bool)
    if excludes is not None:
        for (w0, w1) in excludes:
            exclude_mask[(wav_axis > w0) & (wav_axis < w1)] = 1
    import matplotlib.pyplot as plt

    use_mask = include_mask & ~exclude_mask

    plt.figure()
    plt.step(wav_axis, exclude_mask, 'r-')
    plt.step(wav_axis, include_mask, 'g-')
    plt.step(wav_axis, ~use_mask, 'k--')
    plt.show()
    #Limit to zmask


    #Apply XY padding
    data = data.T
    data[pad:-pad, pad:-pad] = 0
    data = data.T

    snr = data / np.sqrt(var)
    snr[~use_mask] = snrmin - 1
    det = (snr >= snrmin)
    reg = measure.label(det)
    reg_props = measure.regionprops_table(reg, properties=['area', 'label'])
    large_reg = reg_props['area'] > nmin

    obj_mask = np.zeros_like(data, dtype=int)
    n = 1
    for reg_label in reg_props['label'][large_reg]:
        obj_mask[reg == reg_label] = n
        n += 1

    obj_out = utils.matchHDUType(fits_in, obj_mask, header)
    obj_out[0].header["BUNIT"] = "OBJ_ID"

    return obj_out
