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
from scipy.stats import sigmaclip, tstd
from skimage import measure
from tqdm import tqdm

import numpy as np
import os
import pyregion
import sys
import warnings

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
        w0, w1 = wav_axis[0] / (1 + z), wav_axis[-1] / (1 + z)
        line_data = utils.get_gallines(w0, w1)
        lines, ions = line_data['WAV'], line_data['ION']
        labels = ["{0}_{1:.0f}".format(ion, lines[i]) for i, ion in enumerate(ions)]
    else:
        labels = ["custom{0} {1}".format(i, l) for i, l in enumerate(lines)]

    candidates = {label:[] for label in labels}

    zmask = np.zeros_like(wav_axis, dtype=int)


    #Calculate windows
    for i, line in enumerate(lines):
        line_obs = line * (1 + z)
        wav_lo = line_obs * (1 - dv/3e5)
        wav_hi = line_obs * (1 + dv/3e5)
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


def psf_sub_1d(fits_in, pos, fit_rad=2, sub_rad=5, wl_window=150, wmasks=[],
var_cube=[], slice_axis=2):

    """Models and subtracts a point-source on a slice-by-slice basis.

    Args:
        fits_in (astropy FITS object): Input data cube/FITS.
        pos (float tuple): (x,y) position of the source to subtract.
        fit_rad (float): Inner radius, in arcsec, to use for fitting PSF within a slice.
        sub_rad (float): Outer radius, in arcsec, over which to subtract PSF within a slice.
        wl_window (float): Size of white-light window (in Angstrom) to use.
            This is the window used to form a white-light image centered
            on each wavelength layer. Default: 150A.
        wmasks (list): List of wavelength tuples to exclude when making
            white-light images. Use to exclude nebular emission or sky lines.
        slice_axis (int): Which axis represents the slices of the image.
            For KCWI default data cubes, slice_axis = 2. For PCWI data cubes,
            slice_axis = 1.
        var_cube (numpy.ndarray): Variance cube associated with input. Optional.
            Method returns propagated variance if given.
    Returns:
        numpy.ndarray: PSF-subtracted data cube
        numpy.ndarray: PSF model data
        numpy.ndarray: (if var_cube given) Propagated variance cube

    """
    #Extract data + meta-data
    cube, hdr = fits_in[0].data, fits_in[0].header
    cube = np.nan_to_num(cube, nan=0, posinf=0, neginf=0)
    usevar = var_cube != []
    cd3_3 = hdr["CD3_3"]
    wav_axis = coordinates.get_wav_axis(hdr)
    y0, x0 = pos

    #Get distance meshgrid in arcsec
    rr_arcsec = coordinates.get_rgrid(fits_in, x0, y0, unit='arcsec')

    #Rotate so slice axis is 2 (Clockwise by 90 deg)
    if slice_axis == 1:
        cube = np.rot90(cube, axes=(1, 2), k=1)
        rr_arcsec = np.rot90(rr_arcsec, k=1)
        tempy0 = y0
        y0 = x0
        x0 = tempy0

    #Some useful shapes etc.
    z, y, x = cube.shape
    Z, Y, X = np.arange(z), np.arange(y), np.arange(x)
    x0 = int(round(x0))
    y0 = int(round(y0))
    psf_model = np.zeros_like(cube)

    #Create wavelength masked based on input
    zmask = np.ones_like(wav_axis, dtype=bool)
    for (w0, w1) in wmasks:
        zmask[(wav_axis > w0) & (wav_axis < w1)] = 0
    nzmax = np.count_nonzero(zmask)

    #Create masks for fitting and subtracting PSF
    fit_mask = rr_arcsec <= fit_rad
    sub_mask = rr_arcsec <= sub_rad

    #Loop over wavelength first to minimize repetition of wl-mask calculation
    for j, wav_j in enumerate(wav_axis):

        #Get initial width of white-light bandpass in px
        wl_width_px = wl_window / cd3_3

        #Create initial white-light mask, centered on j with above width
        wl_mask = zmask & (np.abs(Z - j) <= wl_width_px / 2)

        #Grow until minimum number of valid wavelength layers included
        while np.count_nonzero(wl_mask) < min(nzmax, wl_window / cd3_3):
            wl_width_px += 2
            wl_mask = zmask & (np.abs(Z - j) <= wl_width_px / 2)

        #Iterate over slices and perform subtraction
        for slice_i in X:

            #Skip slices where no subtraction needed
            if np.count_nonzero(sub_mask[:, slice_i]) < 10: continue

            #Get 1D profile of current slice/wav and WL profile
            j_prof = cube[j, :, slice_i].copy()

            #Exclude negative WL pixels from scaling fit
            sub_mask_i = sub_mask[:, slice_i]
            fit_mask_i = np.zeros_like(sub_mask_i)
            fit_mask_i[x0-1:x0+2] = 1

            #If there aren't pixels on either side to be subtracted
            if np.count_nonzero(sub_mask_i) < 3:
                continue


            #Make WL profile
            N_wl = np.count_nonzero(wl_mask)

            wl_prof = np.sum(cube[wl_mask, :, slice_i], axis=0) / N_wl
            if np.count_nonzero(~sub_mask_i) > 5:
                wl_prof -= np.median(wl_prof[~sub_mask_i])
            if np.sum(wl_prof[fit_mask_i]) / np.std(wl_prof) < 7:
                continue

            #Calculate scaling factor
            scale_factors = j_prof[fit_mask_i] / wl_prof[fit_mask_i]

            scale_factor_med = np.mean(scale_factors)
            scale_factor = scale_factor_med

            #Create WL model
            wl_model = scale_factor * wl_prof

            psf_model[j, sub_mask_i, slice_i] += wl_model[sub_mask_i]

            if usevar:
                wl_var = np.sum(var_cube[wl_mask, :, slice_i], axis=0) / N_wl**2
                wl_model_var = (scale_factor**2) * wl_var
                var_cube[j, sub_mask_i, slice_i] += wl_model_var[sub_mask_i]


    #Subtract PSF model
    cube -= psf_model

    #Rotate back if data was rotated at beginning (Clockwise by 270 deg)
    if slice_axis == 1:
        cube = np.rot90(cube, axes=(1, 2), k=3)
        psf_model = np.rot90(cube, axes=(1, 2), k=3)

    if usevar:
        return cube, psf_model, var_cube

    else:
        return cube, psf_model

def psf_sub_2d(inputfits, pos, fit_rad=1.5, sub_rad=5.0, wl_window=200,
wmasks=[], recenter=True, recenter_rad=5, var_cube=[], maskpsf=False):

    """Models and subtracts point-sources in a 3D data cube.

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
        layer_i = cube[i]

        #Get white-light image and do the same thing
        N_wl = np.count_nonzero(wl_mask)
        wlimg_i = np.sum(cube[wl_mask], axis=0) / N_wl

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

    """Models and subtracts point-sources in a 3D data cube.

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
        method (str): Method of PSF subtraction.
            '1d': Subtract PSFs on slice-by-slice basis with 1D models.
            '2d': Subtract PSFs using a 2D PSF model.
        wmasks (int tuple): Wavelength regions to exclude from white-light images.
        slice_axis (int): Which axis represents the slices of the image.
            For KCWI default data cubes, slice_axis = 2. For PCWI data cubes,
            slice_axis = 1. Only relevant if using 1d subtraction.
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

        if method == '1d':

            res = psf_sub_1d(inputfits,
                pos = pos,
                fit_rad = fit_rad,
                sub_rad = sub_rad,
                wl_window = wl_window,
                wmasks = wmasks,
                slice_axis = slice_axis,
                var_cube = var_cube
            )

        elif method == '2d':

            res = psf_sub_2d(inputfits,
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

def smooth_nd(data, scale, axes=None, ktype='gaussian', var=False):
    """Smooth along all/any axes of a data cube with a box or gaussian kernel.

    Args:
        fits_in (HDU or HDUList): The input data to be smoothed.
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

def segment(fits_in, var, snrmin=3, wranges=[], nmin=10):
    """Segment cube into 3D regions above a threshold.

    Args:
        fits_in (NumPy.ndarray): The input data.
        var (NumPy.ndarray): The input variance
        snrmin (float): The minimum SNR for detection
        nmin (int): The minimum 3D object size, in voxels.
        zrange (int tuple): The z-axis range to consider


    Returns:
        numpy.ndarray: An object mask with labelled regions

    """
    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data.copy(), hdu.header

    #Create wavelength masked based on input
    wav_axis = coordinates.get_wav_axis(header)
    zmask = np.ones_like(wav_axis, dtype=bool)
    for (w0, w1) in wranges:
        zmask[(wav_axis > w0) & (wav_axis < w1)] = 0
    data[zmask] = 0

    snr = data / np.sqrt(var)
    det = (snr >= snrmin)
    lab = measure.label(det)
    labs_unique = np.unique(lab[lab > 0])
    for i, lab_i in enumerate(labs_unique):
        region = lab == lab_i
        if np.count_nonzero(region) < nmin:
            lab[region] = 0
    lab_new = measure.label(lab)
    obj_out = utils.matchHDUType(fits_in, lab_new, header)
    obj_out[0].header["BUNIT"] = "OBJ_ID"
    return obj_out
