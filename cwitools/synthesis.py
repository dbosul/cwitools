"""Tools for generating scientific products from the extracted signal."""

#Standard Imports
import warnings

#Third-party Imports
from astropy.cosmology import WMAP9
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import sigmaclip

import numpy as np
import reproject

#Local Imports
from cwitools import reduction, coordinates, measurement, utils, extraction, modeling

def whitelight(fits_in, wmask=None, var_cube=None, mask_sky=False, skywidth=None, wavgood=True):
    """Get white-light image from cube.

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Returned objects will be of same type as input.

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        wmask (list): List of wavelength tuples to exclude when making
            white-light image. Use to exclude nebular emission or sky lines.
        var (Numpy.ndarray): Variance cube corresponding to input cube
        mask_sky (bool): Set to TRUE to mask some known bright sky lines.
        skywidth (float): Width of sky lines (Angstrom) for the purpose of
            masking. Estimated using header info if not provided.
        wavgood (bool): Set to TRUE to limit to WAVGOOD region.

    Returns:
        HDU / HDUList*: White-light image + header
        HDU / HDUList*: Esimated variance on WL image.
        *Return type matches type of fits_in argument.

    """

    #Extract data + meta-data
    hdu = utils.extract_hdu(fits_in)
    data, header = hdu.data.copy(), hdu.header.copy()

    #Filter data for bad values
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

    #Get new header object for 2D output
    header2d = coordinates.get_header2d(header)
    header2d_var = header2d.copy()

    #Get wavelength axis for masking
    wav_axis = coordinates.get_wav_axis(header)

    if wmask is None:
        wmask = []

    #Create wavelength masked based on input
    if wavgood:
        wmask.append([0, header["WAVGOOD0"]])
        wmask.append([header["WAVGOOD1"], wav_axis[-1]])

    #Apply mask
    zmask = np.zeros_like(wav_axis, dtype=bool)
    for (wav0, wav1) in wmask:
        zmask[(wav_axis > wav0) & (wav_axis < wav1)] = 1

    #Add sky mask if requested
    if mask_sky:
        skymask = utils.get_skymask(header, linewidth=skywidth)
        zmask = zmask | skymask #OR combine

    #Sum over WL wavelengths
    wl_img = np.sum(data[~zmask], axis=0)

    #Get variance estimate, whether variance given or not
    if var_cube is not None:
        var_cube = np.nan_to_num(var_cube, nan=0, posinf=0, neginf=0)
        wl_var = np.sum(var_cube[~zmask], axis=0)
    else:
        wl_var = np.var(data[~zmask], axis=0)

    #Unit conversions
    coeff, bunit_sb = reduction.units.bunit_to_sb(header)
    wl_img *= coeff
    wl_var *= coeff**2
    header2d['BUNIT'] = bunit_sb
    header2d_var['BUNIT'] = bunit_sb

    #Get return type (HDU or HDUList)
    wl_hdu = utils.match_hdu_type(fits_in, wl_img, header2d)
    wl_var_hdu = utils.match_hdu_type(fits_in, wl_var, header2d_var)

    return wl_hdu, wl_var_hdu



def pseudo_nb(fits_in, wav_center, wav_width, pos=None, fit_rad=1, sub_rad=6, var_cube=None):
    """Create a pseudo-Narrow-Band (pNB) image from a data cube.

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Returned objects will be of same type as input.

    Args:
        fits_in (astropy HDU or HDUList): Input HDU/HDUList with 3D data.
        wav_center (float): The central wavelength of the pNB, in Angstrom.
        wav_width (float): The bandwidth of the pNB, in Angstrom.
        pos (float tuple): Provide the x,y location the source to subtract.
            Leave empty to skip white-light subtraction.
        fit_rad (float): Radius (px) to use for scaling the PSF.
        sub_rad (float): Radius (px) to use when subtracting PSF.
        var_cube (NumPy.ndarray): Variance cube associated with input cube.
            Provide to obtain variance estimates on pNB (and WL) images.

    Returns:
        HDU / HDUList*: pseudo-Narrowband image.
        HDU / HDUList*: The variance on the pNB image.
        HDU / HDUList*: White-light / broad-band image.
        HDU / HDUList*: The variance on the white-light image
        *Return type matches type of fits_in argument.

    """

    #Extract data and header from relevant HDU
    hdu = utils.extract_hdu(fits_in)
    int_cube, header3d = hdu.data.copy(), hdu.header.copy()

    #Get 2D header for output
    header2d = coordinates.get_header2d(header3d)
    header2d_var = header2d.copy()

    #Filter out bad values
    int_cube = np.nan_to_num(int_cube, nan=0, posinf=0, neginf=0)

    #Get parameters for NB image and WL image
    pnb_w0 = wav_center - wav_width / 2
    pnb_w1 = wav_center + wav_width / 2

    #Get indices of NB image
    pnb_i0, pnb_i1 = coordinates.get_indices(pnb_w0, pnb_w1, header3d)

    #Handle out of bounds errors or warnings
    if pnb_i1 <= 0 or pnb_i0 >= int_cube.shape[0] - 1:
        raise ValueError("Requested pNB bandpass outside cube range.")

    if pnb_i0 < 0:
        warnings.warn("Requested pNB bandpass is clipped by cube range.")
        pnb_i0 = 0

    if pnb_i1 > int_cube.shape[0]-1:
        warnings.warn("Requested pNB bandpass is clipped by cube range.")
        pnb_i1 = -1

    #Create the narrowband image
    nb_img = np.sum(int_cube[pnb_i0:pnb_i1], axis=0)

    #Get WL data and variance
    wl_hdu, wl_var_hdu = whitelight(
        fits_in,
        wmask=[[pnb_w0, pnb_w1]],
        var_cube=var_cube,
        mask_sky=True
    )
    wl_img = wl_hdu[0].data.copy()
    wl_var = wl_var_hdu[0].data.copy()

    #Estimate or sum the variance
    if var_cube is not None:
        var_cube[np.isnan(var_cube)] = 0
        nb_var = np.sum(var_cube[pnb_i0:pnb_i1], axis=0)
    else:
        nb_var = np.var(int_cube[pnb_i0:pnb_i1], axis=0)

    #nb_img -= np.median(nb_img)
    #wl_img -= np.median(wl_img)

    #Unit conversions
    coeff, bunit_sb = reduction.units.bunit_to_sb(header3d)
    nb_img *= coeff
    nb_var *= coeff**2
    header2d['BUNIT'] = bunit_sb
    header2d_var['BUNIT'] = reduction.units.multiply_bunit(bunit_sb, bunit_sb)

    #Subtract source if a position is provided
    if pos is not None:

        #Get masks for scaling + subtracting
        rr_qso = coordinates.get_rgrid(wl_hdu, pos, unit='arcsec')
        fit_mask = (rr_qso <= fit_rad) & (nb_img > 0) & (wl_img > 0)
        sub_mask = rr_qso <= sub_rad

        #Find scaling factor
        scale_factors = sigmaclip(nb_img[fit_mask] / wl_img[fit_mask]).clipped

        scale = np.nanmedian(scale_factors)

        #Scale WL image subtract
        wl_img *= scale
        nb_img[sub_mask] -= wl_img[sub_mask]

        #Propagate error
        wl_var *= (scale**2)
        nb_var[sub_mask] += wl_var[sub_mask]


    #Add info to header
    header2d["NB_CENTR"] = wav_center
    header2d["NB_WIDTH"] = wav_width

    #Convert all output to HDUs
    nb_out = utils.match_hdu_type(fits_in, nb_img, header2d)
    nb_var_out = utils.match_hdu_type(fits_in, nb_var, header2d)
    wl_out = utils.match_hdu_type(fits_in, wl_img, header2d)
    wl_var_out = utils.match_hdu_type(fits_in, wl_var, header2d)

    return nb_out, nb_var_out, wl_out, wl_var_out

def radial_profile(fits_in, pos, pos_type='image', r_min=None, r_max=None, n_bins=10,
                   scale='lin', r_unit='px', redshift=None, var=None, mask=None, cosmo=WMAP9):
    """Measures a radial profile from a surface brightness (SB) map.

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Args:
        fits_in (HDU or HDUList): Input HDU/HDUList containing SB map.
        pos (float tuple): The center of the profile in image coordinates.
        r_min  (float): The minimum radius, in units determined by runit.
        r_max (float): The maximum radius, in units determined by runit.
        nbins (int): The number of radial bins between r_min  and r_max to use.
        scale (str): The scale for the radial bins.
            'lin' makes bins equal size in linear R
            'log' makes bins equal size in log(R)
        mask (NumPy.ndarray): A 2D binary mask of regions to exclude.
        var (NumPy.ndarray): A 2D map of variance, used for error propagation.
        runit (str): The unit of r_min  and r_max. Can be 'pkpc' or 'px'
            'pkpc' Proper kiloparsec, redshift must also be provided.
            'ckpc' Comoving kiloparsec, redshift must be provided.
            'arcsec' Arcseconds.
            'px' pixels (i.e. distance in image coordinates).
        redshift (float): Redshift of the source, required for calculating physical scales.
        pos_type (str): The type of coordinate given for the 'pos' argument.
            'radec' - (RA, DEC) tuple in decimal degrees
            'image' - (x, y) tuple in image coordinates (default)
    Returns:
        astropy.io.fits.TableHDU: Table containing columns 'radius', 'sb_avg',
            and 'sb_err' (i.e. the radial sb profile)

    """
    #Extract input data
    hdu = utils.extract_hdu(fits_in)
    sb_map, header2d = hdu.data.copy(), hdu.header.copy()

    #Check mask and set to empty if none given
    mask = np.zeros_like(sb_map) if mask is None else mask

    #Get grid of radial distance from pos
    r_grid = coordinates.get_rgrid(
        fits_in, pos,
        unit=r_unit,
        redshift=redshift,
        pos_type=pos_type,
        cosmo=cosmo
        )

    #Get min and max
    r_min = np.min(r_grid) if r_min is None else r_min
    r_max = np.max(r_grid) if r_max is None else r_max

    #Get r array
    if scale == 'lin':
        r_edges = np.linspace(r_min, r_max, n_bins)

    elif scale == 'log':
        r_edges_log = np.linspace(np.log10(r_min), np.log10(r_max), n_bins)
        r_edges = np.power(10, r_edges_log)

    else:
        raise ValueError("'scale' argument can only be 'lin' or 'log'")

    #Create array for radial profile and error
    rprof = np.zeros_like(r_edges[:-1])
    rprof[:] = 0
    rprof_err = np.copy(rprof)
    rcenters = np.copy(rprof)

    #Loop over edges and calculate radial profile
    for i in range(r_edges[:-1].size):

        #Get binary mask of useable spaxels in this radial bin
        rmask = (r_grid >= r_edges[i]) & (r_grid < r_edges[i+1]) & (mask == 0)

        #Skip empty bins
        nmask = np.count_nonzero(rmask)
        if nmask == 0:
            continue

        sb_avg = np.sum(sb_map[rmask]) / nmask

        #Calculate variance, from given variance or sb map
        if var is None:
            sb_var = np.var(sb_map[rmask])
        else:
            sb_var = np.sum(var[rmask]) / nmask**2

        sb_err = np.sqrt(sb_var)
        if "COV_ALPH" in header2d:
            alpha = header2d["COV_ALPH"]
            norm = header2d["COV_NORM"]
            thresh = header2d["COV_THRE"]
            if nmask > thresh:
                sb_err *= norm * (1 + alpha * np.log(thresh))
            else:
                sb_err *= norm * (1 + alpha * np.log(nmask))

        rprof[i] = sb_avg
        rprof_err[i] = sb_err
        rcenters[i] = (r_edges[i] + r_edges[i+1]) / 2.0

    sb_bunit = reduction.units.get_bunit(header2d)

    col1 = fits.Column(
        name='radius',
        format='D',
        array=rcenters,
        unit=r_unit
    )
    col2 = fits.Column(
        name='sb_avg',
        format='D',
        array=rprof,
        unit=sb_bunit
    )
    col3 = fits.Column(
        name='sb_err',
        format='D',
        array=rprof_err,
        unit=sb_bunit
    )
    table_hdu = fits.TableHDU.from_columns([col1, col2, col3])
    return table_hdu

def obj_sb(fits_in, obj_cube, obj_id, var_cube=None, fill_bg=0, redshift=None):
    """Get surface brightness map from segmented 3D objects.

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Returned objects will be of same type as input.

    Args:
        fits_in (astropy HDU or HDUList): Input HDU/HDUList with 3D data.
        obj_cube (NumPy.ndarray): Data cube containing labelled 3D regions.
        obj_id (list or int): ID or list of IDs of objects to include.
        var_cube (NumPy.ndarray): Data cube containing 3D variance estimate.
        redshift (float): Redshift of the emission - provide to apply (1+z)^4 factor for
            cosmological surface brightness dimming correction.
    Returns:
        HDU / HDUList*: Surface brightness map and header.
        HDU / HDUList*: (if var_cube given) Variance on surface brightness map, with header.
        *Return type matches fits_in.
    """
    #Extract data and header
    hdu = utils.extract_hdu(fits_in)
    int_cube, header3d = hdu.data.copy(), hdu.header.copy()

    #Get 2D header and update units
    header2d = coordinates.get_header2d(header3d)
    header2d_var = header2d.copy()

    #Unit conversions
    coeff, bunit_sb = reduction.units.bunit_to_sb(header3d)
    header2d['BUNIT'] = bunit_sb
    header2d_var['BUNIT'] = reduction.units.multiply_bunit(bunit_sb, bunit_sb)

    bin_msk = extraction.obj2binary(obj_cube, obj_id)

    #Mask non-object data and sum SB map
    int_cube[~bin_msk] = 0
    fluxmap = np.sum(int_cube, axis=0)
    sb_map = fluxmap * coeff

    if fill_bg:
        mskmap = np.max(bin_msk, axis=0)
        mskspc = np.sum(bin_msk.astype(int), axis=(1, 2))
        zcenter = np.sum(np.arange(bin_msk.shape[0]) * mskspc) / np.sum(mskspc)
        zcenter = int(round(zcenter))
        bg_map = hdu.data[zcenter].copy()
        bg_map *= coeff
        sb_map[mskmap == 0] = bg_map[mskmap == 0]

    #Apply dimming correction if redshift is provided.
    if redshift is not None:
        sb_map *= (1 + redshift)**4

    #Get output of same FITS/HDU type    as input
    sb_out = utils.match_hdu_type(fits_in, sb_map, header2d)

    #Calculate and return with variance map if varcube provided
    if var_cube is not None:
        var = var_cube.copy()
        var[~bin_msk] = 0
        varmap = np.sum(var, axis=0) * (coeff**2)
        sb_var_out = utils.match_hdu_type(fits_in, varmap, header2d_var)
        return sb_out, sb_var_out

    return sb_out

def obj_spec(fits_in, obj_cube, obj_id, var_cube=None, extend_z=False, rescale_cov=True):
    """Get 1D spectrum of a 3D object or multiple objects.

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Args:
        fits_in (astropy HDU or HDUList): Input HDU/HDUList with 3D data.
        obj_cube (NumPy.ndarray): Data cube containing labelled 3D regions.
        obj_id (list or int): ID or list of IDs of objects to include.
        var_cube (NumPy.ndarray): Data cube containing 3D variance estimate.
        extend_z (bool): Set to TRUE to include full spectrum range from each object spaxel.
        rescale_cov (bool): Rescale the variance cube based on the covariance
            information in the FITS header. This only works when relevant
            keywords are presented in the input FITS file.

    Returns:
        astropy.io.fits.TableHDU: Table with columns 'wav' (wavelength), 'flux',
            and - if var_cube was provided - 'flux_err'.
    """
    #Extract relevant data and header
    hdu = utils.extract_hdu(fits_in)
    int_cube, header3d = hdu.data.copy(), hdu.header.copy()

    bin_msk = extraction.obj2binary(obj_cube, obj_id)

    coeff, bunit_flam = reduction.units.bunit_to_flam(header3d)

    #Extend mask along full z-axis if desired
    if extend_z:
        msk2d = np.max(bin_msk, axis=0)
        bin_msk = np.zeros_like(obj_cube)
        bin_msk[:, msk2d] = 1

    #Mask data and sum over spatial axes
    bin_msk = bin_msk.astype(bool)
    int_cube[~bin_msk] = 0
    spec1d = np.sum(int_cube, axis=(1, 2)) * coeff

    #Get wavelength array
    wav_axis = coordinates.get_wav_axis(header3d)

    col1 = fits.Column(
        name='WAVE',
        format='D',
        array=wav_axis,
        unit=header3d["CUNIT3"]
    )

    col2 = fits.Column(
        name='FLUX',
        format='E',
        array=spec1d,
        unit=bunit_flam
    )

    #Propagate variance and add error column if provided
    if var_cube is not None:
        var = var_cube.copy()
        var[~bin_msk] = 0
        spec1d_var = np.sum(var, axis=(1, 2))
        spec1d_err = np.sqrt(spec1d_var) * coeff

        if rescale_cov and ('COV_A' in header3d):
            n_summed = np.sum(bin_msk, axis=(1, 2))
            cov_terms = ["ALPH", "NORM", "THRE"]
            cov_params = [header3d[k] for k in cov_terms]
            spec1d_err = spec1d_err * modeling.covar_curve(cov_params, n_summed)

        col3 = fits.Column(
            name='FLUX_ERR',
            format='E',
            array=spec1d_err,
            unit=bunit_flam
        )
        table_hdu = fits.TableHDU.from_columns([col1, col2, col3])
    else:
        table_hdu = fits.TableHDU.from_columns([col1, col2])

    return table_hdu

def obj_moments(fits_in, obj_cube, obj_id, var_cube=None, unit='kms'):
    """Creates 2D maps of 1st and 2nd z-moments for 3D objects.

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Returned objects will be of same type as input.

    Args:
        fits_in (astropy HDU or HDUList): Input HDU/HDUList with 3D data.
        obj_cube (NumPy.ndarray): Data cube containing labelled 3D regions.
        obj_id (list or int): ID or list of IDs of objects to include.
        var_cube (NumPy.ndarray): Data cube containing 3D variance estimate.
        unit (str): Desired output unit.
            'kms' - kilometers per second
            'wav' - wavelength units (same as input z-axis)

    Returns:
        HDU / HDUList*: First moment (velocity) map, with header
        HDU / HDUList*: Error on first moment map, with header
        HDU / HDUList*: Second moment (dispersion) map, with header
        HDU / HDUList*: Error on second moment map, with header
        *Return type matches fits_in.
    """
    #Extract relevant data and header
    hdu = utils.extract_hdu(fits_in)
    int_cube, header3d = hdu.data.copy(), hdu.header.copy()

    #Validate unit selection
    if unit not in ['kms', 'wav']:
        raise ValueError("'unit' argument can only be 'wav' or 'kms'")

    #Get 2D header for output
    header2d = coordinates.get_header2d(header3d)

    #Get wavelength axis
    wav_axis = coordinates.get_wav_axis(header3d)

    bin_msk = extraction.obj2binary(obj_cube, obj_id)

    #Create 2D map of object spaxels
    msk2d = np.max(bin_msk, axis=0)

    #Create blank arrays for moment maps
    mu1_map = np.zeros_like(msk2d, dtype=float)
    mu2_map = np.zeros_like(msk2d, dtype=float)

    #Initialize as NaNs
    mu1_map[:] = np.NaN
    mu2_map[:] = np.NaN

    #Also create arrays for moment map error
    mu1_err_map = np.copy(mu1_map)
    mu2_err_map = np.copy(mu2_map)

    #Loop over spaxels and calculate moments
    for y_i in range(int_cube.shape[1]):
        for x_j in range(int_cube.shape[2]):

            msk_ij = bin_msk[:, y_i, x_j]

            #Skip empty spaxels
            if np.count_nonzero(msk_ij) == 0:
                continue

            #Extract wavelength domain and spectrum for this spaxel
            wav_ij = wav_axis[msk_ij]
            spc_ij = int_cube[msk_ij, y_i, x_j]
            var_ij = None if var_cube is None else var_cube[msk_ij, y_i, x_j].copy()

            #Calculate first moment
            mu1, mu1_err = measurement.first_moment(
                wav_ij,
                spc_ij,
                method='basic',
                y_var=var_ij,
                get_err=True
            )

            mu1_map[y_i, x_j] = mu1
            mu1_err_map[y_i, x_j] = mu1_err

            #Calculate second moment
            mu2, mu2_err = measurement.second_moment(
                wav_ij,
                spc_ij,
                mu1=mu1,
                y_var=var_ij,
                get_err=True
            )

            mu2_map[y_i, x_j] = mu2
            mu2_err_map[y_i, x_j] = mu2_err

    #If velocity units requested
    if unit.lower() == 'kms':

        #Get flux-weighted average wavelength
        spec1d = obj_spec(fits_in, obj_cube, obj_id).data['flux']

        zmsk1d = np.max(bin_msk, axis=(1, 2))

        mu1_ref = measurement.first_moment(wav_axis[zmsk1d], spec1d[zmsk1d], method='basic')

        #Convert maps to velocity, in km/s
        cfactor = 3e5 / mu1_ref #speed of light
        mu1_map = cfactor * (mu1_map - mu1_ref)
        mu1_err_map *= cfactor
        mu2_map *= cfactor
        mu2_err_map *= cfactor

        header2d['BUNIT'] = 'km/s'

    else:

        header2d['BUNIT'] = header3d['CUNIT3']

    #Add each to its own HDU or HDUList structure
    mu1_out = utils.match_hdu_type(fits_in, mu1_map, header2d)
    mu1_err_out = utils.match_hdu_type(fits_in, mu1_err_map, header2d)
    mu2_out = utils.match_hdu_type(fits_in, mu2_map, header2d)
    mu2_err_out = utils.match_hdu_type(fits_in, mu2_err_map, header2d)

    #Return all
    return mu1_out, mu1_err_out, mu2_out, mu2_err_out

def obj_moments_zfit(int_fits, obj_cube, obj_id, peak_wav, redshift=0, vel_max=2000,
                     disp_bounds=(50, 500), ratio_bounds=(0.5, 2.0), unit="kms", var=None):
    """Calculate a 2D map of first/second moments using singlet or doublet line fitting.

    Args:
        int_fits (HDUList-like): HDU, HDUList or path to data cube.
        obj_cube (numpy.ndarray): Object label cube
        obj_id (int or list): Object ID or list of objects IDs to include.
        peak_wav (float): Wavelength or 2-tuple of wavelengths representing the peak(s) of singlet
            or doublet line profile. You can provide rest-frame value and redshift, or just provide
            the observed wavelengths and leave redshift = 0.
            For doublet fits, peak separation is fixed, dispersions of both components are tied, and
            the blue-to-red peak ratio is constrained by the ratio_bounds argument.
        redshift (float): The redshift of the source, used to convert the wavelength
            axis to rest-frame units.
        vel_max (float): The maximum velocity offset, in km/s, for the fit center relative to the
            provided peak wavelength(s).
        disp_bounds (float tuple): The lower and upper dispersion bounds for the fit. Dispersions of
            the two components in a doublet fit are tied together.
        ratio_bounds (float tuple): For doublet fits only - the min/max ratio between the blue peak
            and the red peak.
        unit (str): Output unit for moment maps, can be 'kms' for kilometers/second or 'wav' for
            input wavelength axis units
        var (numpy.ndarray): The variance cube associated with the input data

    Returns:
        HDUList/HDU: HDUlist or HDU containing the first moment map.
        HDUList/HDU: HDulist or HDU containing the second moment map.
    """

    hdu = utils.extract_hdu(int_fits)

    #Validate peak_wav argument
    if isinstance(peak_wav, float):
        mode = 1
        peak_wav = [peak_wav]
    elif isinstance(peak_wav, (list, tuple)):
        if len(peak_wav) == 1:
            mode = 1
        elif len(peak_wav) == 2:
            mode = 2
        else:
            raise ValueError("peak_wav can only be one or two values.")
    else:
        raise TypeError("peak_wav must be a float or a list/tuple of floats with length 1-2")

    #Convert to observer-frame units

    for i, p_w in enumerate(peak_wav):
        peak_wav[i] *= (1 + redshift)

    #Extract data cube
    int_cube = hdu.data.copy()

    #Get wavelength axis
    wav_axis = coordinates.get_wav_axis(hdu.header)

    #Get binary mask of object voxels
    bin_mask = extraction.obj2binary(obj_cube, obj_id)
    bin_mask2d = np.max(bin_mask, axis=0)

    #Make maps / arrays for parameter models
    #Note - for convenience, we consider the single peak to be the 'blue' peak from now on
    #i.e. b_amp, b_cen, b_wmin, b_wmax - all apply to the single component of the singlet model
    b_amp = np.zeros_like(int_cube[0])
    b_cen = np.zeros_like(b_amp)
    b_std = np.zeros_like(b_amp)
    ratio = np.zeros_like(b_amp) #only used for doublet

    model_3d = np.zeros_like(int_cube) #Also save model residuals

    #Get lower/upper bounds on the blue (or single) peak
    b_wmin = peak_wav[0] * (1 - vel_max / 3e5)
    b_wmax = peak_wav[0] * (1 + vel_max / 3e5)

    #Get lower/upper bounds on dispersion in wavelength units, using blue peak
    #There will be a slight discrepancy between blue/red peak dispersions, given slightly different
    #wavelengths, but they should be negligible (i.e. < 1%)
    disp_min_wav = peak_wav[0] * disp_bounds[0] / 3e5
    disp_max_wav = peak_wav[0] * disp_bounds[1] / 3e5

    #Figure out model bounds
    if mode == 2:

        #Get boolean mask of wavelength range
        usewav = (wav_axis >= b_wmin) & (wav_axis <= (peak_wav[1] * (1 + vel_max / 3e5)))

        #Establish bounds on doublet model
        model_bounds = [
            (0, int_cube[usewav].max() * 2),
            (b_wmin, b_wmax),
            (disp_min_wav, disp_max_wav),
            ratio_bounds
        ]

    else:

        usewav = (wav_axis >= b_wmin) & (wav_axis <= b_wmax)

        #Establish bounds on doublet model
        model_bounds = [
            (0, int_cube[usewav].max() * 2),
            (b_wmin, b_wmax),
            (disp_min_wav, disp_max_wav)
        ]

    #Get subset of wavelength axis
    wavgood = wav_axis[usewav]

    #Set non-object voxels to zero
    int_cube[~usewav] = 0

    #Get indices under mask and loop over them
    xindices, yindices = np.where(bin_mask2d)
    for y_i, x_i in zip(xindices, yindices):

        spec_ij = int_cube[usewav, y_i, x_i]
        spec_var = None if var is None else var[usewav, y_i, x_i]

        #Fit model params using D.E.
        if mode == 1:
            fit_result = modeling.fit_model1d(
                modeling.gauss1d,
                model_bounds,
                wavgood,
                spec_ij,
                y_var=spec_var
            )
            line_model = modeling.gauss1d(fit_result.x, wavgood)
            line_model_full =  modeling.gauss1d(fit_result.x, wav_axis)
        else:
            fit_result = modeling.fit_model1d(
                modeling.doublet,
                model_bounds,
                wavgood,
                spec_ij,
                peak_wav,
                y_var=spec_var
            )
            #Get model
            line_model = modeling.doublet(fit_result.x, wavgood, peak_wav)
            line_model_full = modeling.doublet(fit_result.x, wav_axis, peak_wav)

        #Skip spaxel if model did not fit successfully
        if not fit_result.success:
            continue

        #Test if fit out-performs basic polynomial using AIC
        poly_coeff = np.polyfit(wavgood, spec_ij, 2)
        poly_model = np.poly1d(poly_coeff)(wavgood)
        poly_aic = modeling.aic(poly_model, spec_ij, 3)
        line_aic = modeling.aic(line_model, spec_ij, 4)
        _, line_p = modeling.bic_weights([poly_aic, line_aic])

        #Reject if model is not confident
        if line_p < 0.95:
            continue

        b_amp[y_i, x_i] = fit_result.x[0]
        b_cen[y_i, x_i] = fit_result.x[1]
        b_std[y_i, x_i] = fit_result.x[2]
        if mode == 2:
            ratio[y_i, x_i] = fit_result.x[3]

        model_3d[:, y_i, x_i] = line_model_full

    #Store as HDULists/HDUs and return
    hdr2d = coordinates.get_header2d(hdu.header)
    nan_map = b_cen == 0

    #Derive moment maps
    if unit == 'kms':
        mu1 = 3e5 * (b_cen - peak_wav[0]) / peak_wav[0]
        mu1[nan_map] = np.nan
        mu2 = (3e5 / peak_wav[0]) * b_std
        mu2[nan_map] = np.nan
        hdr2d["BUNIT"] = "km/s"
    elif unit == 'wav':
        mu1 = b_cen
        mu1[nan_map] = np.nan
        mu2 = b_std
        mu2[nan_map] = np.nan
        hdr2d["BUNIT"] = hdu.header["CUNIT3"]
    else:
        raise ValueError("'unit' argument must be 'kms' or 'wav'")

    mu1_fits_out = utils.match_hdu_type(int_fits, mu1, hdr2d)
    mu2_fits_out = utils.match_hdu_type(int_fits, mu2, hdr2d)
    model3d_fits = utils.match_hdu_type(int_fits, model_3d, hdu.header)

    return mu1_fits_out, mu2_fits_out, model3d_fits

def cylindrical(fits_in, center, seg_mask=None, ellipticity=1., pos_ang=0., n_r=None, npa=None,
                r_range=None, pa_range=None, d_r=None, dpa=None, c_radec=False, compress=True,
                redshift=None, cosmo=WMAP9):
    """Resample a cube in cartesian coordinate to cylindrical coordinate (with ellipticity).
        This function can be used to project 3D cubes to the 2D spectra in lambda-r space.

    Args:
        fits_in (astropy HDU or HDUList): Input HDU/HDUList with 3D data.
        center (float tuple): Center of the cylindrical projection, in pixel coordinate or
            [RA, DEC] if c_radec is True.
        seg_mask (float arrray): Segment mask in 2D (typically generated by SExtractor) that
            masks additional continuum sources that do not corresponds to the central object.
        ellipticity (float): Axis-ratio.
        pos_ang (float): Position angle of the *MAJOR* axis that is east of the north direction.
        n_r (int): Number of pixels of the post-projection cubes in the radial direction.
            Default: The closest integer that makes the size of individual pixels to
            be 0.3 arcsec in the major axis.
        npa (int): Number of pixels of the post-projection cubes in the angular direction.
            Default: The closest integer that makes the size of individual pixels to
            be 1 degree.
        r_range (float tuple): radial bins in major axies inside which the projection is applied.
            Default: 0 to the minimum radius to include all signals in the input cube.
        pa_range (float tuple): position angles bins inside which the projections is applied.
            Default: [0,360]
        d_r (float): radial size per pixel.
        dpa (float): PA size per pixel.
        c_radec (bool): If set, the center is specified using [RA, DEC] instead of pixels.
        compress (bool): Remove axis with only 1 pixel. This risks of losing WCS information on
            the corresponding axis, but is convenient when using DS9.
        cosmo (FlatLambdaCDM): The cosmology to use, as one of Astropy's
            cosmologies (astropy.cosmology.FlatLambdaCDM). Default is WMAP9.
        redshift (float): If set, an additional WCS is added to represent rest-wavelength and pkpc.

    Returns:
        HDU / HDUList*: Post-projection image/cube.
        HDU / HDUList*: Coverage image.
        *Return type matches type of fits_in argument.
    """

    # 0 - the original HDU
    hdu0 = utils.extract_hdu(fits_in)
    hdu0.data = np.nan_to_num(hdu0.data, nan=0, posinf=0, neginf=0)
    wcs0 = WCS(hdu0.header)
    s_z = hdu0.data.shape

    # masking
    mask_3d = np.zeros(hdu0.shape, dtype=bool)
    if not seg_mask is None:
        for i in range(mask_3d.shape[0]):
            mask_3d[i, :, :] = np.bitwise_or(mask_3d[i, :, :], seg_mask)

    hdu0_mask = hdu0.copy()
    hdu0_mask.data[mask_3d] = 0

    # Skewing
    hdu1 = hdu0_mask.copy()
    hdr1 = hdu1.header

    if ellipticity != 1:

        cd0 = np.array([[hdr1['CD1_1'], hdr1['CD1_2']], [hdr1['CD2_1'], hdr1['CD2_2']]])
        rot = np.radians(pos_ang)

        cd_rot = np.matmul([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], cd0)
        cd_shr = cd_rot

        if ellipticity < 1:
            cd_shr[0, 0] = cd_rot[0, 0] / ellipticity
            cd_shr[0, 1] = cd_rot[0, 1] / ellipticity
        elif ellipticity > 1:
            cd_shr[0, 0] = cd_rot[0, 0] * ellipticity
            cd_shr[0, 1] = cd_rot[0, 1] * ellipticity


        cd1 = np.matmul([[np.cos(-rot), -np.sin(-rot)], [np.sin(-rot), np.cos(-rot)]], cd_shr)
        hdr1['CD1_1'] = cd1[0, 0]
        hdr1['CD1_2'] = cd1[0, 1]
        hdr1['CD2_1'] = cd1[1, 0]
        hdr1['CD2_2'] = cd1[1, 1]

    # Shift hdu to the south pole
    wcs1 = WCS(hdr1)
    hdu2 = hdu1.copy()
    hdr2 = hdu2.header

    if c_radec:

        tmp = wcs1.wcs_world2pix(center[0], center[1] + 0.3 / 3600., 0, 0)
        ref_pix = [float(tmp[0]), float(tmp[1])]

        center_ad = center
        tmp = wcs0.wcs_world2pix(center[0], center[1], 0, 0)
        center_pix = [float(tmp[0]), float(tmp[1])]

    else:
        center_pix = center
        tmp = wcs0.all_pix2world(center[0], center[1], 0, 0)
        center_ad = [float(tmp[0]), float(tmp[1])]
        tmp = wcs1.all_world2pix(tmp[0], tmp[1] + 0.3 / 3600., 0, 0)
        ref_pix = [float(tmp[0]), float(tmp[1])]


    hdr2['CRPIX1'] = ref_pix[0] + 1
    hdr2['CRPIX2'] = ref_pix[1] + 1
    hdr2['CRVAL1'] = 0.
    hdr2['CRVAL2'] = -90 + 0.3 / 3600.
    wcs2 = WCS(hdr2)

    if pa_range is None:
        pa_range = [0, 360]

    # Calculate the x dimension of the final cube
    if npa is None and dpa is None:
        d_x0 = 1.
        n_x0 = int(np.round((pa_range[1] - pa_range[0]) / d_x0))
        pa_range[1] = pa_range[0] + d_x0 * n_x0
    else:
        if npa is not None:
            nx0 = int(npa)
            dx0 = (pa_range[1] - pa_range[0]) / nx0
        if dpa is not None:
            nx0 = int(np.round(pa_range[1] - pa_range[0] / dpa))
            dx0 = dpa
            pa_range[1] = pa_range[0] + dx0 * nx0

    # Split too large pixels
    if d_x0 > 1:
        n_x = int(dx0) * n_x0
        d_x = (pa_range[1] - pa_range[0]) / n_x
    else:
        n_x = n_x0
        d_x = d_x0

    # Splitting into multiple cubes. Also need padding to avoid edge effect.
    nx3 = np.round(n_x / 3)
    nx3 = np.array([nx3, nx3, n_x - 2 * nx3])
    xr3 = np.zeros((2, 3))
    xr3[0, 0] = pa_range[1]
    xr3[1, 0] = pa_range[1] - nx3[0] * d_x
    xr3[0, 1] = xr3[1, 0]
    xr3[1, 1] = pa_range[1] - (nx3[0] + nx3[1]) * d_x
    xr3[0, 2] = xr3[1, 1]
    xr3[1, 2] = pa_range[1] - np.sum(nx3) * d_x
    nx3 = nx3 + 10
    xr3[0, :] = xr3[0, :] + 5 * d_x
    xr3[1, :] = xr3[1, :] - 5 * d_x


    # Calculate the y dimension
    if r_range is None:
        p_corner_x = np.array([0, 0, s_z[2]-1, s_z[2]-1])
        p_corner_y = np.array([0, s_z[1]-1, s_z[1]-1, 0])
        p_corner_z = np.array([0, 0, 0, 0])
        tmp = wcs2.wcs_pix2world(p_corner_x, p_corner_y, p_corner_z, 0)
        #a_corner = tmp[0] - unused variable?
        d_corner = tmp[1]
        r_corner = np.abs(d_corner+90) * 3600.
        r_range = [0, np.max(r_corner)]

    if n_r is None and d_r is None:
        d_y = 0.3
        n_y = int(np.round((r_range[1] - r_range[0]) / d_y))
        r_range[1] = r_range[0] + d_y * n_y
    else:
        if n_r is not None:
            n_y = int(n_r)
            d_y = (r_range[1]-r_range[0])/n_y
        if d_r is not None:
            n_y = int(np.round((r_range[1] - r_range[0]) / d_r))
            d_y = d_r
            r_range[1] = r_range[0] + d_y * n_y

    # Set up headers
    hdr3_1 = hdu2.header.copy()
    hdr3_1['NAXIS1'] = int(nx3[0])
    hdr3_1['NAXIS2'] = n_y
    hdr3_1['CTYPE1'] = 'RA---CAR'
    hdr3_1['CTYPE2'] = 'DEC--CAR'
    hdr3_1['CUNIT1'] = 'deg'
    hdr3_1['CUNIT2'] = 'deg'
    hdr3_1['CRVAL1'] = xr3[0, 0]
    hdr3_1['CRVAL2'] = 0.
    hdr3_1['CRPIX1'] = 0.5
    hdr3_1['CRPIX2'] = (90. - r_range[0]) / (d_y / 3600.) + 0.5
    hdr3_1['CD1_1'] = -d_x
    hdr3_1['CD2_1'] = 0.
    hdr3_1['CD1_2'] = 0.
    hdr3_1['CD2_2'] = d_y / 3600.
    hdr3_1['LONPOLE'] = 180.
    hdr3_1['LATPOLE'] = 0.

    hdr3_2 = hdr3_1.copy()
    hdr3_2['NAXIS1'] = int(nx3[1])
    hdr3_2['CRVAL1'] = xr3[0, 1]

    hdr3_3 = hdr3_1.copy()
    hdr3_3['NAXIS1'] = int(nx3[2])
    hdr3_3['CRVAL1'] = xr3[0, 2]

    # reproject

    # This is to avoid user-panicking when runing long programs...
    utils.output("\tProjecting...\n")
    utils.output("\t\t#1 of 3\n")
    cube3_1, area3_1 = reproject.reproject_interp(hdu2, hdr3_1)
    utils.output("\t\t#2 of 3\n")
    cube3_2, area3_2 = reproject.reproject_interp(hdu2, hdr3_2)
    utils.output("\t\t#3 of 3\n")
    cube3_3, area3_3 = reproject.reproject_interp(hdu2, hdr3_3)

    # Merge the separate cubes

    #Short-hand refs to condense the code below
    c31s = cube3_1.shape
    c32s = cube3_2.shape
    c33s = cube3_3.shape
    data4 = np.zeros((c31s[0], c31s[1], c31s[2] + c32s[2] + c33s[2] - 30))

    #Shortening the following assignments using intermediate 'index' variables
    ind0, ind1 = 0, c31s[2] - 10
    data4[:, :, ind0:ind1] = cube3_1[:, :, 5 : c31s[2] - 5]

    ind0, ind1 = c31s[2] - 10, c31s[2] + c32s[2] - 20
    data4[:, :, ind0:ind1] = cube3_2[:, :, 5 : c32s[2] - 5]

    ind0, ind1 = c31s[2] + c32s[2] - 20, c31s[2] + c32s[2] + c33s[2] - 30
    data4[:, :, ind0:ind1] = cube3_3[:, :, 5:c33s[2]-5]

    data4[data4 == 0] = np.nan

    #More short-hand variable names to condense code
    a31s = area3_1.shape
    a32s = area3_2.shape
    a33s = area3_3.shape
    area4 = np.zeros((a31s[0], a31s[1], a31s[2] + a32s[2] + a33s[2] - 30))

    #Re-using index variables
    ind0, ind1 = 0, a31s[2] - 10
    area4[:, :, ind0:ind1] = area3_1[:, :, 5:a31s[2]-5]

    ind0, ind1 = a31s[2] - 10, a31s[2] + a32s[2]-20
    area4[:, :, ind0:ind1] = area3_2[:, :, 5:a32s[2]-5]

    ind0, ind1 = a31s[2] + a32s[2] - 20, a31s[2] + a32s[2] + a33s[2] - 30
    area4[:, :, ind0:ind1] = area3_3[:, :, 5:a33s[2] - 5]

    area4 = np.nan_to_num(area4)
    area4[~np.isfinite(data4)] = 0

    # Setup WCS
    hdr5 = hdr3_1.copy()
    tmp_dict = {
        'NAXIS1'  : n_x0,
        'CTYPE1'  : 'PA',
        'CTYPE2'  : 'Radius',
        'CNAME1'  : 'PA',
        'CNAME2'  : 'Radius',
        'CRVAL1'  : pa_range[1],
        'CRVAL2'  : r_range[0],
        'CRPIX2'  : 0.5,
        'CUNIT2'  : 'arcsec',
        'CD1_1'   : -d_x0,
        'CD2_2'   : d_y,
        'C2C_ORA' : (center_ad[0], 'RA of origin'),
        'C2C_ODEC': (center_ad[1], 'DEC of origin'),
        'C2C_OX'  : (center_pix[0] + 1, 'X of origin'),
        'C2C_OY'  : (center_pix[1] + 1, 'Y of origin'),
        'C2C_E'   : (ellipticity, 'Axis raio'),
        'C2C_EPA' : (pos_ang, 'PA of the major axis')
    }
    for key, val in tmp_dict.items():
        hdr5[key] = val

    if redshift is not None:
        a_dis = (cosmo.arcsec_per_kpc_proper(redshift)).value
        tmp_dict = {
            'CTYPE1A': 'PA',
            'CTYPE2A': 'Radius',
            'CTYPE3A': hdr5['CTYPE3'],
            'CNAME1A': 'PA',
            'CNAME2A': 'Radius',
            'CNAME3A': hdr5['CNAME3'],
            'CRVAL1A': pa_range[1],
            'CRVAL2A': r_range[0] / a_dis,
            'CRVAL3A': hdr5['CRVAL3'] / (1 + redshift),
            'CRPIX1A': hdr5['CRPIX1'],
            'CRPIX2A': 0.5,
            'CRPIX3A': hdr5['CRPIX3'],
            'CUNIT1A': hdr5['CUNIT1'],
            'CUNIT2A': 'kpc',
            'CUNIT3A': hdr5['CUNIT3'],
            'CD1_1A' : -d_x0,
            'CD2_2A' : d_y / a_dis,
            'CD3_3A' : hdr5['CD3_3'] / (1 + redshift),
        }
        for key, val in tmp_dict.items():
            hdr5[key] = val

    ahdr5 = hdr5.copy()
    data5 = np.zeros((data4.shape[0], data4.shape[1], n_x0))
    area5 = np.zeros((data4.shape[0], data4.shape[1], n_x0))

    # averaging redundant pixels
    ratio = int(d_x0 / d_x)
    if ratio != 1:
        for i in range(nx0):
            tmp = data4[:, :, i * ratio : (i + 1) * ratio]
            data5[:, :, i] = np.nanmean(tmp, axis=2)
            tmp = area4[:, :, i * ratio : (i + 1) * ratio]
            area5[:, :, i] = np.sum(tmp, axis=2)

    # Compress
    if compress:
        if n_x0 == 1:

            tmp = hdr5.copy()
            hdr5['NAXIS'] = 2
            hdr5['NAXIS1'] = tmp['NAXIS3']
            hdr5['NAXIS2'] = tmp['NAXIS2']
            del hdr5['NAXIS3']
            hdr5['CTYPE1'] = tmp['CTYPE3']
            hdr5['CTYPE2'] = tmp['CTYPE2']
            del hdr5['CTYPE3']
            hdr5['CUNIT1'] = tmp['CUNIT3']
            hdr5['CUNIT2'] = tmp['CUNIT2']
            del hdr5['CUNIT3']
            hdr5['CNAME1'] = tmp['CNAME3']
            hdr5['CNAME2'] = tmp['CNAME2']
            del hdr5['CNAME3']
            hdr5['CRVAL1'] = tmp['CRVAL3']
            hdr5['CRVAL2'] = tmp['CRVAL2']
            del hdr5['CRVAL3']
            hdr5['CRPIX1'] = tmp['CRPIX3']
            hdr5['CRPIX2'] = tmp['CRPIX2']
            del hdr5['CRPIX3']
            hdr5['CD1_1'] = tmp['CD3_3']
            hdr5['CD2_2'] = tmp['CD2_2']
            del hdr5['CD3_3']

            if redshift is not None:

                hdr5['CTYPE1A'] = tmp['CTYPE3A']
                hdr5['CTYPE2A'] = tmp['CTYPE2A']
                del hdr5['CTYPE3A']
                hdr5['CUNIT1A'] = tmp['CUNIT3A']
                hdr5['CUNIT2A'] = tmp['CUNIT2A']
                del hdr5['CUNIT3A']
                hdr5['CNAME1A'] = tmp['CNAME3A']
                hdr5['CNAME2A'] = tmp['CNAME2A']
                del hdr5['CNAME3A']
                hdr5['CRVAL1A'] = tmp['CRVAL3A']
                hdr5['CRVAL2A'] = tmp['CRVAL2A']
                del hdr5['CRVAL3A']
                hdr5['CRPIX1A'] = tmp['CRPIX3A']
                hdr5['CRPIX2A'] = tmp['CRPIX2A']
                del hdr5['CRPIX3A']
                hdr5['CD1_1A'] = tmp['CD3_3A']
                hdr5['CD2_2A'] = tmp['CD2_2A']
                del hdr5['CD3_3A']

            data5 = np.transpose(np.squeeze(data5, axis=2))
            ahdr5 = hdr5.copy()
            area5 = np.transpose(np.squeeze(area5, axis=2))

        elif n_y == 1:

            tmp = hdr5.copy()
            hdr5['NAXIS'] = 2
            hdr5['NAXIS1'] = tmp['NAXIS3']
            hdr5['NAXIS2'] = tmp['NAXIS1']
            del hdr5['NAXIS3']
            hdr5['CTYPE1'] = tmp['CTYPE3']
            hdr5['CTYPE2'] = tmp['CTYPE1']
            del hdr5['CTYPE3']
            hdr5['CUNIT1'] = tmp['CUNIT3']
            hdr5['CUNIT2'] = tmp['CUNIT1']
            del hdr5['CUNIT3']
            hdr5['CNAME1'] = tmp['CNAME3']
            hdr5['CNAME2'] = tmp['CNAME1']
            del hdr5['CNAME3']
            hdr5['CRVAL1'] = tmp['CRVAL3']
            hdr5['CRVAL2'] = tmp['CRVAL1']
            del hdr5['CRVAL3']
            hdr5['CRPIX1'] = tmp['CRPIX3']
            hdr5['CRPIX2'] = tmp['CRPIX1']
            del hdr5['CRPIX3']
            hdr5['CD1_1'] = tmp['CD3_3']
            hdr5['CD2_2'] = tmp['CD1_1']
            del hdr5['CD3_3']

            if redshift is not None:

                hdr5['CTYPE1A'] = tmp['CTYPE3A']
                hdr5['CTYPE2A'] = tmp['CTYPE1A']
                del hdr5['CTYPE3A']
                hdr5['CUNIT1A'] = tmp['CUNIT3A']
                hdr5['CUNIT2A'] = tmp['CUNIT1A']
                del hdr5['CUNIT3A']
                hdr5['CNAME1A'] = tmp['CNAME3A']
                hdr5['CNAME2A'] = tmp['CNAME1A']
                del hdr5['CNAME3A']
                hdr5['CRVAL1A'] = tmp['CRVAL3A']
                hdr5['CRVAL2A'] = tmp['CRVAL1A']
                del hdr5['CRVAL3A']
                hdr5['CRPIX1A'] = tmp['CRPIX3A']
                hdr5['CRPIX2A'] = tmp['CRPIX1A']
                del hdr5['CRPIX3A']
                hdr5['CD1_1A'] = tmp['CD3_3A']
                hdr5['CD2_2A'] = tmp['CD1_1A']
                del hdr5['CD3_3A']

            data5 = np.transpose(np.squeeze(data5, axis=1))
            ahdr5 = hdr5.copy()
            area5 = np.transpose(np.squeeze(area5, axis=2))

    hdu5 = utils.match_hdu_type(fits_in, data5, hdr5)
    ahdu5 = utils.match_hdu_type(fits_in, area5, ahdr5)

    return (hdu5, ahdu5)

def sum_spec_r(fits_in, ra, dec, redshift, radius, var_cube=None, wmask=None, rescale_cov=False):
    """Get summed spectrum of a region specified by a center and radius.

    Args:
        fits_in (HDUList or HDU): The input FITS with data cube
        ra (float): The right-ascension of the central position, in degress.
        dec (float): The declination of the central position, in degrees.
        redshift (float): The redshift of the source
        radius (float): The radius, in proper kpc, within which to sum.
        wmask (list): List of wavelength tuples to mask
    Returns:
        astropy.io.fits.TableHDU: Table with columns 'wav' (wavelength), 'flux',
            and - if var_cube was provided - 'flux_err'.
    """
    hdu = utils.extract_hdu(fits_in)
    data, header3d = hdu.data.copy(), hdu.header.copy()

    coeff, bunit_flam = reduction.units.bunit_to_flam(header3d)

    #Get radius grid
    r_grid = coordinates.get_rgrid(
        fits_in,
        (ra, dec),
        unit='pkpc',
        redshift=redshift,
        pos_type='radec'
    )

    #Get binary mask of spaxels to sum
    rmask = r_grid <= radius

    #Set spaxels outside the mask to zero
    cube_t = data.T
    cube_t[~rmask.T] = 0
    cube = cube_t.T


    #Sum spectrum over remaining spaxels
    spec = np.sum(cube, axis=(1, 2)) * coeff
    wav_axis = coordinates.get_wav_axis(header3d)

    #Mask some wavelengths if requested
    if wmask is not None:
        zmask = np.zeros_like(wav_axis, dtype=bool)
        for (w_0, w_1) in wmask:
            zmask[(wav_axis > w_0) & (wav_axis < w_1)] = 1
        spec[zmask] = np.median(spec)

    col1 = fits.Column(
        name='WAVE',
        format='D',
        array=wav_axis,
        unit=header3d["CUNIT3"]
    )
    col2 = fits.Column(
        name='FLUX',
        format='E',
        array=spec,
        unit=bunit_flam
    )

    #Propagate variance and add error column if provided
    if var_cube is not None:

        var_cube_t = var_cube.copy().T
        var_cube_t[~rmask.T] = 0

        spec1d_var = np.sum(var_cube_t.T, axis=(1, 2))
        spec1d_err = np.sqrt(spec1d_var) * coeff

        if rescale_cov and ('COV_A' in header3d):
            n_summed = np.sum(rmask)
            cov_terms = ["ALPH", "NORM", "THRE"]
            cov_params = [header3d[k] for k in cov_terms]
            spec1d_err = spec1d_err * modeling.covar_curve(cov_params, n_summed)

        col3 = fits.Column(
            name='FLUX_ERR',
            format='E',
            array=spec1d_err,
            unit=bunit_flam
        )
        table_hdu = fits.TableHDU.from_columns([col1, col2, col3])
    else:
        table_hdu = fits.TableHDU.from_columns([col1, col2])

    return table_hdu
