"""Tools for extracting extended emission from a cube."""
#Standard Imports
import os

#Third-party Imports
from astropy import units as u
from astropy import convolution
from astropy.cosmology import WMAP9
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from photutils import DAOStarFinder
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import medfilt
from scipy.ndimage import convolve as sc_ndi_convolve
from scipy.stats import sigmaclip, tstd
from skimage import measure, morphology
from tqdm import tqdm

import numpy as np
import pyregion

#Local Imports
from cwitools import coordinates, utils, modeling
from cwitools.modeling import fwhm2sigma
from cwitools.reduction.variance import scale_variance

def apply_mask(data, mask, fill=0):
    """Apply a binary or label mask to data.

    Args:
        data (numpy.ndarray): The data to be masked.
        mask (numpy.ndarray): The mask to apply (1s are masked).
        fill (float): The value to replace mask==1 pixels with.

    Returns:
        numpy.ndarray: The masked data
    """
    data_masked = data.copy()

    #Check shape
    if data.shape == mask.shape:
        data_masked[mask == 1] = fill
    elif mask.shape == data[0].shape:
        for layer in range(data.shape[0]):
            data_masked[layer][mask == 1] = fill
    else:
        raise ValueError("Mask should either match dimensions of data or data[0]")

    #Return masked data
    return data_masked

def detect_lines(obj_fits, lines=None, redshift=0, vwidth=500):
    """Associate detected 3D objects with known emission lines.

    Args:
        obj_fits (HDU or HDUList): The input 3D object mask.
        lines (float list): Optional list of rest-frame emission lines to compare
            against, in units of Angstrom. Over-rides default line list.
        redshift (float): The redshift of the emission.
        vwidth (float): The velocity window of each line, in km/s,  within
            which objects are considered to be associated. (+/- vwidth)

    Returns:
        dict: A dictionary of the format {<line>:<obj_ids>} where
            <line> is a given input line, and <obj_ids> is a list of integer
            labels for the objects.
    """
    hdu = utils.extract_hdu(obj_fits)
    obj_mask, header = hdu.data.copy(), hdu.header
    wav_axis = coordinates.get_wav_axis(header)

    if lines is None:
        line_data = utils.get_neblines(wav_axis[0], wav_axis[-1], redshift)
        labels, lines = line_data['ION'], line_data['WAV']
    else:
        labels = ["custom{0} {1}".format(i, l) for i, l in enumerate(lines)]

    candidates = {label:[] for label in labels}

    #Calculate windows
    for i, line in enumerate(lines):
        wav_lo = line * (1 - vwidth / 3e5)
        wav_hi = line * (1 + vwidth / 3e5)
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
           pos_type='img', cosmo=WMAP9):
    """Extract a spatial box around a central position from 2D or 3D data.

    Returned data has same dimensions as input data. Return type (HDU/HDUList)
    also matches input type. First HDU is used if input is HDUList.

    Args:
        fits_in (astropy HDU or HDUList): HDU or HDUList with 2D or 3D data.
        pos (float tuple): Center of cutout, given as (axis0, axis1) coordinate
            by default (if pos_type is set to'image') or as an (RA,DEC) tuple, if
            pos_type is set to 'radec'
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
        pos_type (str): The type of coordinate given for the 'pos' argument.
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
    hdu = utils.extract_hdu(fits_in)
    header = hdu.header.copy()

    #Get 2D WCS information from cube regardless of 2D or 3D input
    if header["NAXIS"] == 2:
        header2d = header
    elif header["NAXIS"] == 3:
        header2d = coordinates.get_header2d(fits_in[0].header)
    else:
        raise ValueError("2D or 3D input only for get_cutout.")

    #If RA/DEC position given, convert to image coordinates
    if pos_type == 'radec':
        ra, dec = pos
        wcs2d = WCS(header2d)
        pos = tuple(float(x) for x in wcs2d.all_world2pix(pos[0], pos[1], 0))
    elif pos_type == 'image':
        ra, dec = tuple(float(x) for x in wcs2d.all_pix2world(pos[0], pos[1], 0))
    else:
        raise ValueError("pos_type argument must be 'image' or 'radec'")


    #Get box size, either as scalar or angular Astropy quantity
    if unit in ['pkpc', 'ckpc']:
        kpc_per_px = coordinates.get_kpc_per_px(header, redshift=redshift, unit=unit, cosmo=cosmo)
        box_size = box_size / kpc_per_px
    elif unit == 'arcsec':
        box_size = box_size * u.arcsec #Cutout2D accepts angular Quantities
    elif unit != 'px':
        raise ValueError("Unit must be px, arcsec, pkpc or ckpc.")

    #Create modified fits and update spatial axes WCS
    fits_out = fits_in.copy()
    if header["NAXIS"] == 2:
        fits_out[0].data = Cutout2D(fits_in[0].data, pos, box_size, wcs2d,
                                    mode='partial',
                                    fill_value=fill
                                    ).data
    else:
        new_cube = []
        for i in range(len(fits_in[0].data)):
            new_cube.append(Cutout2D(fits_in[0].data[i], pos, box_size, wcs2d,
                                     mode='partial',
                                     fill_value=fill
                                     ).data
                           )
        fits_out[0].data = np.array(new_cube)

    #Update spatial axes of WCS
    fits_out[0].header["CRVAL1"] = ra
    fits_out[0].header["CRVAL2"] = dec
    fits_out[0].header["CRPIX1"] = pos[0]
    fits_out[0].header["CRPIX2"] = pos[1]
    fits_out[0].header["NAXIS1"] = fits_out[0].data.shape[-1]
    fits_out[0].header["NAXIS2"] = fits_out[0].data.shape[-2]

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
    hdu = utils.extract_hdu(fits_in)

    if len(hdu.data.shape) == 3:
        src_mask = np.zeros_like(hdu.data[0])
        header2d = coordinates.get_header2d(hdu.header)
    elif len(hdu.data.shape) == 2:
        src_mask = np.zeros_like(hdu.data)
        header2d = hdu.header.copy()
    else:
        raise ValueError("Input data must be 2D or 3D")

    if os.path.isfile(reg):
        #Open region file in image coordinates
        reg_img = pyregion.open(reg).as_imagecoord(header=header2d)

    else:
        raise FileNotFoundError("%s does not exist." % reg)

    #Run through sources
    ygrid, xgrid = np.indices(src_mask.shape)
    for src in reg_img:
        src_x, src_y, rad = src.coord_list
        src_x -= 1
        src_y -= 1
        src_rr = np.sqrt((xgrid - src_x)**2 + (ygrid - src_y)**2)
        src_mask[src_rr <= rad] = 1

    hdu_out = utils.match_hdu_type(fits_in, src_mask, header2d)
    return hdu_out


def psf_sub(inputfits, pos, r_fit=1.5, r_sub=5.0, wl_window=200, wmasks=None, recenter=True,
            var=None, maskpsf=False, use_model=None):
    """Models and subtracts a single point-source in a 3D data cube.

    Args:
        inputfits (astrop FITS object): Input data cube/FITS.
        r_fit (float): Inner radius, in arcsec, used for fitting PSF.
        r_sub (float): Outer radius, in arcsec, used to subtract PSF.
        pos (float tuple): Position of the source to subtract in image coords.
        recenter (bool): Recenter (x, y) using the centroid within a radius of 2''.
        wl_window (int): Size of white-light window (in Angstrom) to use. This is the window used
            to form a white-light image centered on each wavelength layer. Default: 200A.
        wmasks (list): List of wavelength tuples to exclude when making
            white-light images. Use to exclude nebular emission or sky lines.
        var (numpy.ndarray): Variance cube associated with input. Optional.
            Method returns propagated variance if given.
        use_model (str): Set to 'moffat' or 'gauss' to replace the empirical PSF
            model with a 2D Moffat or 2D Gaussian estimate instead. Default
            is None (i.e. standard empirical PSF model is used). This takes significantly longer,
            but can help when subtracting blended sources.

    Returns:
        numpy.ndarray: PSF-subtracted data cube
        numpy.ndarray: PSF model cube
        numpy.ndarray: (if var_cube given) Propagated variance cube

    """

    #Open fits image and extract info
    cube = inputfits[0].data.copy()
    header = inputfits[0].header
    zrange = np.arange(cube.shape[0])
    wav = coordinates.get_wav_axis(header)
    cd3_3 = header["CD3_3"]

    if var is not None:
        var_cube = var.copy()
        usevar = True

    #Get plate scales in arcseconds and Angstrom
    rr_arcsec = coordinates.get_rgrid(inputfits, pos, unit='arcsec')

    #Remove NaN values
    cube = np.nan_to_num(cube, nan=0.0, posinf=0, neginf=0)

    #Create cube for subtracted cube and for model of psf and assoc. variance
    psf_cube = np.zeros_like(cube)
    psf_cube_var = np.zeros_like(psf_cube)

    #Create white-light image
    zmask = np.ones_like(wav, dtype=bool)
    for (wav0, wav1) in wmasks:
        zmask[(wav >= wav0) & (wav <= wav1)] = 0
    nzmax = np.count_nonzero(zmask)

    #Create WL image
    wl_img = np.sum(cube[zmask], axis=0)

    ygrid, xgrid = np.indices(wl_img.shape)

    #Reposition source using data within 1'' if requested
    if recenter:
        recenter_img = wl_img.copy()
        recenter_img[rr_arcsec > 2.0] = 0
        pos = center_of_mass(recenter_img)
        rr_arcsec = coordinates.get_rgrid(inputfits, pos, unit='arcsec')

    #Get boolean masks for
    fit_mask = (rr_arcsec <= r_fit)
    sub_mask = (rr_arcsec <= r_sub)

    fitx = xgrid[np.where(fit_mask == 1)]
    fity = ygrid[np.where(fit_mask == 1)]
    xmin, xmax = fitx.min() - 1, fitx.max() + 2
    ymin, ymax = fity.min() - 1, fity.max() + 2

    #Run through wavelength layers
    for i in tqdm(range(wav.size)):

        #Get current layer and do median subtraction (after sigclipping)
        layer_i = cube[i].copy()
        layer_i -= np.median(sigmaclip(layer_i, low=2, high=2).clipped)

        #Construct empirical PSF model if using this method
        if use_model is None:

            #Get initial width of white-light bandpass in px
            wl_width_px = wl_window / cd3_3

            #Create initial white-light mask, centered on j with above width
            wl_mask = zmask & (np.abs(zrange - i) <= wl_width_px / 2)

            #Grow mask until minimum number of valid wavelength layers included
            while np.count_nonzero(wl_mask) < min(nzmax, wl_window / cd3_3):
                wl_width_px += 2
                wl_mask = zmask & (np.abs(zrange - i) <= wl_width_px / 2)

            #Get white-light image and do the same thing
            n_wl = np.count_nonzero(wl_mask)
            wl_img_i = np.sum(cube[wl_mask], axis=0) / n_wl

            #Try to remove any elevated background levels
            wl_img_i -= np.median(sigmaclip(wl_img_i, low=2, high=2).clipped)

            #Calculate scaling factor for PSF model
            scale_factor = np.median(layer_i[fit_mask] / wl_img_i[fit_mask])

            #Set to zero for bad values
            if scale_factor < 0 or np.isinf(scale_factor) or np.isnan(scale_factor):
                scale_factor = 0

            #Create empirical PSF model by scaling WL image
            psf_model = scale_factor * wl_img_i

            #Propagate variance on this
            if usevar:
                psf_model_var = (scale_factor / n_wl)**2 * np.sum(var_cube[wl_mask], axis=0)
                psf_cube_var[i][sub_mask] += psf_model_var[sub_mask]

        #If user wants to fit analytical model instead of empirical PSF
        else:
            if 'moffat' in use_model.lower():
                model_func = modeling.moffat2d
                model_bounds = [
                    (0, layer_i[fit_mask].max() * 3),
                    (pos[0], pos[0]),
                    (pos[1], pos[1]),
                    (0.1, 15.0),
                    (0.1, 15.0)
                ]

            elif 'gauss' in use_model.lower():
                model_func = modeling.gauss2d
                model_bounds = [
                    (0, layer_i[fit_mask].max() * 3),
                    (pos[1], pos[1]),
                    (pos[0], pos[0]),
                    (0.5, 2.0),
                    (0.5, 2.0),
                    (0, 0)
                ]
            else:
                raise ValueError("use_model must be 'gauss' or 'moffat'")

            model_fit = modeling.fit_model2d(
                model_func,
                model_bounds,
                ygrid[ymin:ymax, xmin:xmax],
                xgrid[ymin:ymax, xmin:xmax],
                layer_i[ymin:ymax, xmin:xmax]
            )
            #Create analytical model
            psf_model = model_func(model_fit.x, ygrid, xgrid)

        #Add 2D PSF model to psf cube, within subtraction mask
        psf_cube[i][sub_mask] += psf_model[sub_mask]

    #Subtract 3D PSF model
    cube -= psf_cube

    if maskpsf:
        cube = cube.T
        cube[fit_mask.T] = 0
        cube = cube.T

    #Return subtracted data alongside model
    if usevar:
        var_cube += psf_cube_var
        return cube, psf_cube, var_cube

    return cube, psf_cube

def psf_sub_all(inputfits, r_fit=1.5, r_sub=5.0, reg=None, pos=None,
                recenter=True, auto=7, wl_window=200, wmasks=None, var_cube=None,
                maskpsf=False):
    """Models and subtracts multiple point-sources in a 3D data cube.

    Args:
        inputfits (astrop FITS object): Input data cube/FITS.
        r_fit (float): Inner radius, in arcsec, used for fitting PSF.
        r_sub (float): Outer radius, in arcsec, used to subtract PSF.
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
    cube = np.nan_to_num(cube, nan=0.0, posinf=0, neginf=0)
    header = inputfits[0].header
    wav = coordinates.get_wav_axis(header)
    usevar = var_cube is not None

    #Get WCS information
    wcs = WCS(header)

    #Create cube for subtracted cube and for model of psf
    sub_cube = cube.copy()

    #Create wavelength mask for white-light image
    zmask = np.ones_like(wav, dtype=bool)
    for (wav0, wav1) in wmasks:
        zmask[(wav >= wav0) & (wav <= wav1)] = 0

    #Create white-light image
    wl_img = np.sum(cube[zmask], axis=0)

    #Get list of sources, method depending on input
    sources = []

    #If a single source is given, use that.
    if pos is not None:
        sources = [pos]

    #If region file is given
    elif reg is not None:

        if os.path.isfile(reg):
            reg_file = pyregion.open(reg)
        else:
            raise FileNotFoundError("Region file could not be found: %s"%reg)

        for src in reg_file:
            src_ra, src_dec = src.coord_list[:2]
            src_x, src_y, _ = wcs.all_world2pix(src_ra, src_dec, header["CRVAL3"], 0)
            src_x = float(src_x)
            src_y = float(src_y)
            sources.append((src_y, src_x))

    #Otherwise, use the automatic method
    else:

        #Get standard deviation in WL image (sigmaclip to remove bright sources)
        stddev = np.std(sigmaclip(wl_img, low=3, high=3).clipped)

        #Run source finder
        daofind = DAOStarFinder(fwhm=8.0, threshold=auto * stddev)
        auto_srcs = daofind(wl_img)

        #Get list of peak values
        peaks = list(auto_srcs['peak'])

        #Make list of sources
        sources = []
        for i in range(len(auto_srcs['xcentroid'])):
            sources.append((auto_srcs['xcentroid'][i], auto_srcs['ycentroid'][i]))

        #Sort according to peak value (this will be ascending)
        peaks, sources = zip(*sorted(zip(peaks, sources)))

        #Cast back to list
        sources = list(sources)

        #Reverse to get descending order (brightest sources first)
        sources.reverse()

    #Run through sources
    psf_model = np.zeros_like(cube)
    sub_cube = cube.copy()

    for src_pos in sources:

        res = psf_sub(inputfits,
                      pos=src_pos,
                      r_fit=r_fit,
                      r_sub=r_sub,
                      wl_window=wl_window,
                      wmasks=wmasks,
                      var=var_cube,
                      maskpsf=maskpsf,
                      recenter=recenter
                      )
        if len(res) == 3:
            sub_cube, model_p, var_cube = res

        elif len(res) == 2:
            sub_cube, model_p = res

        #Update FITS data and model cube
        inputfits[0].data = sub_cube
        psf_model += model_p

    if usevar:
        return sub_cube, psf_model, var_cube

    return sub_cube, psf_model

def bg_sub(inputfits, method='polyfit', poly_k=1, median_window=31, wmasks=None,
           mask_reg=None, var=None):
    """Subtracts extended continuum emission / scattered light from a cube

    Args:
        inputfits (Astropy HDUList): FITS object to be subtracted
        method (str): Which method to use to model background
            'polyfit': Fits polynomial to the spectrum in each spaxel (default.)
            'median': Subtract the spatial median of each wavelength layer.
            'medfilt': Model spectrum in each spaxel by median filtering it.
            'noisefit': Model noise in each z-layer and subtract mean.
        poly_k (int): The degree of polynomial to use for background modeling.
        median_window (int): The filter window size to use if median filtering.
        wmasks (int tuple): Wavelength regions to exclude from white-light images.
        mask_reg (str): Path to a DS9 region file to use to exclude regions
            when using 'median' method of bg subtraction.
        var (numpy.ndarray): Variance cube associated with input data.
            NOTE: Variance is only formally propagated for 'polyfit'. For  other methods, the input
            variance is re-scaled empirically using reduction.variance.scale_variance.

    Returns:
        numpy.ndarray: Background-subtracted cube
        numpy.ndarray: Cube containing background model which was subtracted.
        numpy.ndarray: (if var provided) Cube containing updated variance estimate

    """

    #Load header and data
    header = inputfits[0].header.copy()
    cube = inputfits[0].data.copy()
    var_out = None if var is None else var.copy()
    wav = coordinates.get_wav_axis(header)
    model_cube = np.zeros_like(cube)

    #Get empty regions mask
    mask_2d = np.sum(cube, axis=0) == 0

    #Create wavelength mask for white-light image
    zmask = np.ones_like(wav, dtype=bool)
    if wmasks is not None:
        for (wav0, wav1) in wmasks:
            zmask[(wav >= wav0) & (wav <= wav1)] = 0

    #Subtract background by fitting a low-order polynomial
    if method == 'polyfit':

        #Run through spaxels and subtract low-order polynomial
        for y_ind in tqdm(range(cube.shape[1])):
            for x_ind in range(cube.shape[2]):

                #Extract spectrum at this location
                spectrum = cube[:, y_ind, x_ind].copy()

                coeff, covar = np.polyfit(wav[zmask], spectrum[zmask], poly_k, full=False, cov=True)

                polymodel = np.poly1d(coeff)

                #Get background model
                bg_model = polymodel(wav)

                if mask_2d[y_ind, x_ind] != 0:
                    continue

                cube[:, y_ind, x_ind] -= bg_model

                #Add to model
                model_cube[:, y_ind, x_ind] += bg_model

                if var is None:
                    continue

                for m in range(covar.shape[0]):
                    var_m = 0
                    for l in range(covar.shape[1]):
                        var_m += np.power(wav, poly_k - l) * covar[l, m] / np.sqrt(covar[m, m])
                    var_out[:, y_ind, x_ind] += var_m**2

    #Subtract background by estimating it with a median filter
    elif method == 'medfilt':

        #Get median filtered spectrum as background model
        bg_model = medfilt(cube, kernel_size=(median_window, 1, 1))

        bg_model_t = bg_model.T
        bg_model_t[mask_2d.T] = 0
        bg_model = bg_model_t.T

        #Run through spaxels and subtract low-order polynomial
        for y_ind in tqdm(range(cube.shape[1])):
            for x_ind in range(cube.shape[2]):
                cube[:, y_ind, x_ind] -= bg_model
                model_cube[:, y_ind, x_ind] += bg_model

        if var is not None:
            var_out = scale_variance(cube, var)

    #Subtract layer-by-layer by fitting noise profile
    elif method == 'noisefit':
        fitter = fitting.SimplexLSQFitter()
        medians = []
        for z_ind in range(cube.shape[0]):

            #Extract layer
            layer = cube[z_ind]
            layer_bg = layer[~mask_2d]

            #Get trimmed/sigmaclipped background median
            med = np.median(layer_bg)
            std = np.std(layer_bg)
            t_std = tstd(layer_bg, limits=(-3 * std, 3 * std))
            t_med = np.median(layer_bg[np.abs(layer_bg - med) < 3 * t_std])

            medians.append(t_med)

        #Fit low-order polynomial to estimate background levels through cube
        medians = np.array(medians)
        bg_model_0 = models.Polynomial1D(degree=2)

        bg_model_1 = fitter(bg_model_0, wav[zmask], medians[zmask])
        for i, wav_i in enumerate(wav):
            cube[i][~mask_2d] -= bg_model_1(wav_i)
            model_cube[i][~mask_2d] = bg_model_1(wav_i)

        if var is not None:
            var_out = scale_variance(cube, var)

    #Subtract using simple layer-by-layer median value
    elif method == "median":

        if mask_reg is not None:
            msk2d = reg2mask(inputfits, mask_reg)[0].data
        else:
            msk2d = np.zeros_like(cube[0], dtype=bool)

        for z_ind in range(cube.shape[0]):
            layer = cube[z_ind].copy()
            layer_clipped = sigmaclip(layer[msk2d == 0], low=2, high=2).clipped
            model_cube[z_ind] = np.median(layer_clipped)

        #model_cube = medfilt(model_cube, kernel_size=(3, 1, 1))
        cube -= model_cube

        if var is not None:
            var_out = scale_variance(cube, var)

    if var is None:
        return cube, model_cube
    return cube, model_cube, var_out

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


    if ktype == 'box':
        kernel = convolution.Box1DKernel(scale)
    elif ktype == 'gaussian':
        sigma = fwhm2sigma(scale)
        kernel = convolution.Gaussian1DKernel(sigma)
    else:
        err = "No kernel type '%s' for wavelength smoothing" % (ktype)
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


    if ktype == 'box':
        kernel = convolution.Box2DKernel(scale)
    elif ktype == 'gaussian':
        sigma = fwhm2sigma(scale)
        kernel = convolution.Gaussian2DKernel(sigma)
    else:
        err = "No kernel type '%s' for spatial smoothing" % (ktype)
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

    if axes is None:
        axes = range(len(data.shape))

    axes = np.array(axes)
    naxes = len(axes)
    ndims = len(data.shape)

    if naxes > ndims or np.any(axes >= ndims):
        raise ValueError("Requested axis greater than dimensions of data.")

    if ndims < 1 or ndims > 3:
        raise ValueError("smooth_nd only works for 1-3 dimensional data.")

    #Smooth subset of axes for 1D or 3D data
    if ndims in [1, 3]:

        if ktype == 'box':
            kernel = convolution.Box1DKernel(scale)
        elif ktype == 'gaussian':
            sigma = fwhm2sigma(scale)
            kernel = convolution.Gaussian1DKernel(sigma)
        else:
            err = "No kernel type '%s' for %iD smoothing" % (ktype, naxes)
            raise ValueError(err)

        kernel = np.power(np.array(kernel), 2) if var else np.array(kernel)

        for axis_i in axes:
            data_copy = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode='same'),
                axis=axis_i,
                arr=data_copy.copy()
            )

        return data_copy

    #Otherwise - data must be 2D
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

    if np.issubdtype(type(obj_id), np.integer):
        bin_cube = obj_mask == obj_id
    elif isinstance(obj_id, list) and np.all(np.array([type(x) for x in obj_id]) == int):
        bin_cube = np.zeros_like(obj_mask, dtype=bool)
        for oid in obj_id:
            bin_cube[obj_mask == oid] = 1
    else:
        raise TypeError("obj_id must be an integer or list of integers.")
    return bin_cube

def segment(fits_in, var, snrmin=3, includes=None, excludes=None, nmin=10, pad=0,
            fill_holes=False, snr_int=None):
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
            edge artifacts.
        fill_holes (bool): Set to TRUE to auto-fill holes in 3D objects.
        snr_int (float): Integrated SNR threshold, use instead of nmin to base
            selection on the total SNR instead of size.

    Returns:
        numpy.ndarray: An object mask with labelled regions

    """
    hdu = utils.extract_hdu(fits_in)
    data, header = hdu.data.copy(), hdu.header.copy()

    #Create wavelength masked based on input
    wav = coordinates.get_wav_axis(header)

    #Start off with empty includes mask and add included regions
    include_mask = np.zeros_like(wav, dtype=bool)
    if isinstance(includes, list):
        for (wav0, wav1) in includes:
            include_mask[(wav > wav0) & (wav < wav1)] = 1

    #If no includes given, use whole array by default
    if np.sum(include_mask) == 0:
        include_mask[:] = 1

    exclude_mask = np.zeros_like(wav, dtype=bool)
    if excludes is not None:
        for (wav0, wav1) in excludes:
            exclude_mask[(wav > wav0) & (wav < wav1)] = 1

    use_mask = include_mask & ~exclude_mask

    #Apply XY padding
    data = data.T
    data[pad:-pad, pad:-pad] = 0
    data = data.T

    snr = data / np.sqrt(var)
    snr[~use_mask] = snrmin - 1
    det = (snr >= snrmin)

    #Repair objects if requested
    if fill_holes:
        det = morphology.binary_closing(det)

    reg = measure.label(det)
    reg_props = measure.regionprops_table(
        reg,
        intensity_image=data,
        properties=['area', 'label', 'mean_intensity']
    )
    large_regs = reg_props['area'] > nmin
    if snr_int is None:
        detected_regs = large_regs

    else:
        var_props = measure.regionprops_table(
            reg,
            intensity_image=var,
            properties=['area', 'label', 'mean_intensity']
        )
        int_totals = reg_props['area'] * reg_props['mean_intensity']
        var_totals = var_props['area'] * var_props['mean_intensity']

        #Scale for covariance
        if "COV_ALPH" in header:
            alpha = header["COV_ALPH"]
            norm = header["COV_NORM"]
            thresh = header["COV_THRE"]

            large = reg_props['area'] > thresh
            beta = norm * (1 + alpha * np.log(thresh))

            var_totals[large] *= (beta**2)
            var_totals[~large] *= (norm * (1 + alpha * np.log(reg_props['area'][~large])))**2

            snr_totals = int_totals / np.sqrt(var_totals)
            detected_regs = (snr_totals >= snr_int) & (large_regs)

    #Convert selected regions into new mask
    obj_mask = np.zeros_like(data, dtype=int)
    for reg_label in reg_props['label'][detected_regs]:
        obj_mask[reg == reg_label] = 1

    #Re-label objects so numbers start at 1 and are consecutive
    obj_mask = measure.label(obj_mask)

    header["BUNIT"] = "OBJ_ID"
    obj_out = utils.match_hdu_type(fits_in, obj_mask, header)

    return obj_out
