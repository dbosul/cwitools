"""Tools for generating scientific products from the extracted signal."""
from astropy import units as u
from astropy import convolution
from astropy.cosmology import WMAP9
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from cwitools import coordinates, measurement, utils
from cwitools.modeling import fwhm2sigma
from scipy.stats import sigmaclip
from skimage import measure

import numpy as np
import os
import pyregion
import warnings

def whitelight(fits_in,  wmask=[], var_cube=None, mask_sky=False, wavgood=True):
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
        wavgood (bool): Set to TRUE to limit to WAVGOOD region.

    Returns:
        HDU / HDUList*: White-light image + header
        HDU / HDUList*: Esimated variance on WL image.
        *Return type matches type of fits_in argument.

    """

    #Extract data + meta-data
    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data, hdu.header

    #Filter data for bad values
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

    #Get new header object for 2D output
    header2d = coordinates.get_header2d(header)
    header2d_var = header2d.copy()

    #Get wavelength axis for masking
    wav_axis = coordinates.get_wav_axis(header)

    #Create wavelength masked based on input
    if wavgood:
        wmask.append([0, header["WAVGOOD0"]])
        wmask.append([header["WAVGOOD1"], wav_axis[-1]])

    #Apply mask
    zmask = np.zeros_like(wav_axis, dtype=bool)
    for (w0, w1) in wmask:
        zmask[(wav_axis > w0) & (wav_axis < w1)] = 1

    #Add sky mask if requested
    if mask_sky:
        skymask = utils.get_skymask(header)
        zmask = zmask | skymask #OR combine

    #Sum over WL wavelengths
    wl_img = np.sum(data[~zmask], axis=0)

    #Get variance estimate, whether variance given or not
    if var_cube is not None:
        var = np.nan_to_num(var_cube, nan=0, posinf=0, neginf=0)
        wl_var = np.sum(var_cube[~zmask], axis=0)
    else:
        wl_var = np.var(data[~zmask], axis=0)

    #Unit conversions
    if 'BUNIT' in header.keys():
        bunit = utils.get_bunit(header)
        if not 'electrons' in bunit:
            bunit2d = utils.multiply_bunit(bunit, 'angstrom')
            bunit2d_var = utils.multiply_bunit(bunit2d, bunit2d)

            flam2f = coordinates.get_pxsize_angstrom(header)
            wl_img *= flam2f
            wl_var *= flam2f**2

        #Update header
        header2d['BUNIT'] = bunit2d
        header2d_var['BUNIT'] = bunit2d_var

    #Get return type (HDU or HDUList)
    wl_hdu = utils.matchHDUType(fits_in, wl_img, header2d)
    wl_var_hdu = utils.matchHDUType(fits_in, wl_var, header2d_var)

    return wl_hdu, wl_var_hdu



def pseudo_nb(fits_in, wav_center, wav_width, pos=None, fit_rad=2,
sub_rad=None, var_cube=None):
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
    hdu = utils.extractHDU(fits_in)
    int_cube, header3d = hdu.data, hdu.header

    #Get 2D header for output
    header2d = coordinates.get_header2d(header3d)
    header2d_var = header2d.copy()

    #Filter out bad values
    int_cube = np.nan_to_num(int_cube, nan=0, posinf=0, neginf=0)

    #Get wavelength axis
    wav_axis = coordinates.get_wav_axis(header3d)

    #Get parameters for NB image and WL image
    pnb_wA = wav_center - wav_width / 2
    pnb_wB = wav_center + wav_width / 2

    #Get indices of NB image
    A, B = coordinates.get_indices(pnb_wA, pnb_wB, header3d)

    #Handle out of bounds errors or warnings
    if B <= 0 or A >= int_cube.shape[0] - 1:
        raise ValueError("Requested pNB bandpass outside cube range.")

    if A < 0:
        warnings.warn("Requested pNB bandpass is clipped by cube range.")
        A = 0

    if B > int_cube.shape[0]-1:
        warnings.warn("Requested pNB bandpass is clipped by cube range.")
        B = -1

    #Create the narrowband image
    nb_img = np.sum(int_cube[A:B], axis=0)

    #Estimate or sum the variance
    if var_cube is not None:
        var_cube[np.isnan(var_cube)] = 0
        nb_var = np.sum(var_cube[A:B], axis=0)
    else:
        nb_var = np.var(var_cube[A:B], axis=0)

    #Unit conversions
    if 'BUNIT' in header.keys():
        bunit = utils.get_bunit(header)
        if not 'electrons' in bunit:
            bunit2d = utils.multiply_bunit(bunit, 'angstrom')
            bunit2d_var = utils.multiply_bunit(bunit2d, bunit2d)

            flam2f = coordinates.get_pxsize_angstrom(header)
            wl_img *= flam2f
            wl_var *= flam2f**2

        #Update header
        header2d['BUNIT'] = bunit2d
        header2d_var['BUNIT'] = bunit2d_var

    #Get WL data and variance
    wl_hdu, wl_var_hdu = whitelight(fits_in,
        wmask=[[pnb_wA, pnb_wB]],
        var_cube=var_cube
    )
    wl_img = wl_hdu.data

    #Subtract source if a position is provided
    if pos is not None:

        #Get masks for scaling + subtracting
        rr_qso = coordinates.get_rgrid(wl_hdu, pos[0], pos[1])
        fitMask = rr_qso <= fit_rad
        subMask = rr_qso <= sub_rad

        #Find scaling factor
        scale_factors = sigmaclip(nb_img[fitMask] / wl_img[fitMask]).clipped
        scale = np.median(scale_factors)

        #Scale WL image subtract
        wl_img *= scale
        nb_img[subMask] -= wl_img[subMask]

        #Propagate error
        wl_var *= (scale**2)
        nb_var[subMask] += wl_var[subMask]

    #Add info to header
    header2d["NB_CENTR"] = wav_center
    header2d["NB_WIDTH"] = wav_width

    #Convert all output to HDUs
    nb_out = utils.matchHDUType(fits_in, nb_img, header2d)
    nb_var_out = utils.matchHDUType(fits_in, nb_var, header2d)
    wl_out = utils.matchHDUType(fits_in, wl_img, header2d)
    wl_var_out = utils.matchHDUType(fits_in, wl_var, header2d)

    return nb_out, nb_var_out, wl_out, wl_var_out

def radial_profile(fits_in, pos, rmin=-1, rmax=-1, nbins=10, scale='lin',
mask=None, var_map=None, runit='px', redshift=None):
    """Measures a radial profile from a surface brightness (SB) map.

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Args:
        fits_in (HDU or HDUList): Input HDU/HDUList containing SB map.
        pos (float tuple): The center of the profile in image coordinates.
        rmin (float): The minimum radius, in units determined by runit.
        rmax (float): The maximum radius, in units determined by runit.
        nbins (int): The number of radial bins between rmin and rmax to use.
        scale (str): The scale for the radial bins.
            'lin' makes bins equal size in linear space.
            'log' makes bins equal size in log space.
        mask (NumPy.ndarray): A 2D binary mask of regions to exclude.
        var (NumPy.ndarray): A 2D map of variance, used for error propagation.
        runit (str): The unit of rmin and rmax. Can be 'pkpc' or 'px'
            'pkpc' Proper kiloparsec, redshift must also be provided.
            'px' pixels (i.e. distance in image coordinates)

    Returns:
        astropy.io.fits.TableHDU: Table containing columns 'radius', 'sb_avg',
            and 'sb_err' (i.e. the radial sb profile)

    """
    #Extract input data
    hdu = utils.extractHDU(fits_in)
    sb_map, header2d = hdu.data, hdu.header

    #Check mask and set to empty if none given
    mask = np.zeros_like(sb_map) if mask is none else mask

    if runit == 'pkpc':
        rr = coordinates.get_rgrid(fits_in, pos[0], pos[1], unit='arcsec')
        if redshift is None:
            raise ValueError("Redshift must be provided if runit='pkpc'")
        else:
            pkpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift) / 60.0
        rr *= pkpc_per_arcsec

    #Get min and max
    rmin = np.min(rr) if rmin == -1 else rmin
    rmax = np.max(rr) if rmax == -1 else rmax

    #Get r array
    if scale == 'lin':
        r_edges = np.linspace(rmin, rmax, nbins)

    elif scale == 'log':
        r_edges_log = np.linspace(np.log10(rmin), np.log10(rmax), nbins)
        r_edges = np.power(10, r_edges_log)

    else:
        raise ValueError("'scale' argument can only be 'lin' or 'log'")

    #Create array for radial profile and error
    rprof = np.zeros_like(r_edges[:-1])
    rprof[:] = np.NaN
    rprof_err = np.copy(rprof)
    rcenters = np.copy(rprof)

    #Loop over edges and calculate radial profile
    for i in range(r_edges[:-1].size):

        #Get binary mask of useable spaxels in this radial bin
        rmask = (rr >= r_edges[i]) & (rr < r_edges[i+1]) & ~mask

        #Skip empty bins
        nmask = np.count_nonzero(rmask)
        if nmask == 0: continue

        sb_avg = np.sum(sb_map[rmask]) / nmask

        #Calculate variance, from given variance or sb map
        if var_map is None:
            sb_var = np.var(sb_map[rmask])
        else:
            sb_var = np.sum(var_map[rmask]) / nmask**2

        sb_err = np.sqrt(sb_var)

        rprof[i] = sb_avg
        rprof_err[i] = sb_err
        rcenters[i] = (r_edges[i] + r_edges[i+1]) / 2.0

    col1 = fits.Column(
        name='radius',
        format='D',
        array=rcenters,
        unit=runit
    )
    col2 = fits.Column(
        name='sb_avg',
        format='D',
        array=rprof,
        unit=utils.get_bunit(header2d)
    )
    col3 = fits.Column(
        name='sb_err',
        format='D',
        array=rprof_err,
        unit=utils.get_bunit(header2d)
    )
    table_hdu = fits.TableHDU.from_columns([col1, col2, col3])
    return table_hdu

def obj_sb(fits_in, obj_cube, obj_id, var_cube=None):
    """Get surface brightness map from segmented 3D objects.

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Returned objects will be of same type as input.

    Args:
        fits_in (astropy HDU or HDUList): Input HDU/HDUList with 3D data.
        obj_cube (NumPy.ndarray): Data cube containing labelled 3D regions.
        obj_id (list or int): ID or list of IDs of objects to include.
        var_cube (NumPy.ndarray): Data cube containing 3D variance estimate.

    Returns:
        HDU / HDUList*: Surface brightness map and header.
        HDU / HDUList*: Variance on surface brightness map, with header.
        *Return type matches fits_in.
    """
    #Extract data and header
    hdu = utils.extractHDU(fits_in)
    int_cube, header3d = hdu.data, hdu.header

    #Get conversion to SB
    flam2sb = coordinates.get_flam2sb(header3d)

    bin_msk = extraction.obj2binary(obj_cube, obj_id)

    #Mask non-object data and sum SB map
    int_cube[~bin_msk] = 0
    fluxmap = np.sum(int_cube, axis=0)
    sbmap = fluxmap * flam2sb

    #Get 2D header and update units
    header2d = coordinates.get_header2d(header3d)
    header2d['BUNIT'] = header3d['BUNIT'].replace('FLAM', 'SB')

    #Get output of same FITS/HDU type as input
    sb_out = utils.matchHDUType(fits_in, sbmap, header2d)

    #Calculate and return with variance map if varcube provided
    if type(var_cube) == type(obj_cube):
        var_cube[~bin_msk] = 0
        varmap = np.sum(var_cube, axis=0) * (flam2sb**2)
        sb_var_out = utils.matchHDUType(fits_in, varmap, header2d)
        return sb_out, sb_var_out

    else:
        return sb_out

def obj_spec(fits_in, obj_cube, obj_id, var_cube=None, limit_z=True):
    """Get 1D spectrum of segmented 3D objects.

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Args:
        fits_in (astropy HDU or HDUList): Input HDU/HDUList with 3D data.
        obj_cube (NumPy.ndarray): Data cube containing labelled 3D regions.
        obj_id (list or int): ID or list of IDs of objects to include.
        var_cube (NumPy.ndarray): Data cube containing 3D variance estimate.
        limit_z (bool): Set to False to use full spectrum in each object spaxel.

    Returns:
        astropy.io.fits.TableHDU: Table with columns 'wav' (wavelength), 'flux',
            and - if var_cube was provided - 'flux_err'.
    """
    #Extract relevant data and header
    hdu = utils.extractHDU(fits_in)
    int_cube, header3d = hdu.data, hdu.header

    bin_msk = extraction.obj2binary(obj_cube, obj_id)

    #Extend mask along full z-axis if desired
    if not limit_z:
        msk2d = np.max(bin_msk, axis=0)
        bin_msk = np.zeros_like(obj_cube).T
        bin_msk[msk2d] = 1
        bin_msk = bin_msk.T

    #Mask data and sum over spatial axes
    int_cube[bin_msk] = 0
    spec1d = np.sum(int_cube, axis=(1, 2))

    #Get wavelength array
    wav_axis = coordinates.get_wav_axis(header3d)

    #Get 1D header and create HDU-like object matching input type
    header1d = coordinates.get_header1d(header3d)
    spec1d_out = utils.matchHDUType(fits_in, spec1d, header1d)

    col1 = fits.Column(
        name='wav',
        format='D',
        array=wav_axis,
        unit=header3d["CUNIT3"]
    )
    col2 = fits.Column(
        name='flux',
        format='E',
        array=spec1d,
        unit=header3d['BUNIT']
    )

    #Propagate variance and add error column if provided
    if var_cube is not None:
        var_cube[bin_msk] = 0
        spec1d_var = np.sum(var_cube, axis=(1, 2))
        spec1d_err = np.sqrt(spec1d_var)
        col3 = fits.Column(
            name='flux_err',
            format='E',
            array=spec1d_err,
            unit=header3d['BUNIT']
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
    hdu = utils.extractHDU(fits_in)
    int_cube, header3d = hdu.data, hdu.header

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
    m1_map = np.zeros_like(msk2d, dtype=float)
    m2_map = np.zeros_like(msk2d, dtype=float)

    #Initialize as NaNs
    m1_map[:] = np.NaN
    m2_map[:] = np.NaN

    #Also create arrays for moment map error
    m1_err_map = np.copy(m1_map)
    m2_err_map = np.copy(m2_map)

    #Loop over spaxels and calculate moments
    for yi in range(int_cube.shape[1]):
        for xj in range(int_cube.shape[2]):

            msk_ij = bin_msk[:, yi, xj]

            #Skip empty spaxels
            if np.count_nonzero(msk_ij) == 0: continue

            #Extract wavelength domain and spectrum for this spaxel
            wav_ij = wav_axis[msk_ij]
            spc_ij = int_cube[msk_ij, yi, xj]
            var_ij = [] if var_cube == [] else var_cube[msk_ij, yi, xj]

            #Calculate first moment
            m1, m1_err = measurement.first_moment(wav_ij, spc_ij,
                method = 'basic',
                y_var = var_ij,
                get_err = True
            )

            m1_map[yi, xj] = m1
            m1_err_map[yi, xj] = m1_err

            #Calculate second moment
            m2, m2_err = measurement.second_moment(wav_ij, spc_ij,
                m1 = m1,
                y_var = var_ij,
                get_err = True
            )

            m2_map[yi, xj] = m2
            m2_err_map[yi, xj] = m2_err

    #If velocity units requested
    if unit.lower() == 'kms':

        #Get flux-weighted average wavelength
        spec1d = obj_spec(fits_in, obj_cube, obj_id)
        m1_ref = measurement.first_moment(wav_axis, spec1d, method='basic')

        #Convert maps to velocity, in km/s
        cfactor = 3e5 / m1_ref #speed of light
        m1_map = cfactor * (m1_map - m1_ref)
        m1_err_map *= cfactor
        m2_map *= cfactor
        m2_err_map *= cfactor

        header2d['BUNIT'] = 'km/s'

    else:

        header2d['BUNIT'] = header3d['CUNIT3']

    #Add each to its own HDU or HDUList structure
    m1_out = utils.matchHDUType(fits_in, m1_map, header2d)
    m1_err_out = utils.matchHDUType(fits_in, m1_err_map, header2d)
    m2_out = utils.matchHDUType(fits_in, m2_map, header2d)
    m2_err_out = utils.matchHDUType(fits_in, m2_err_map, header2d)

    #Return all
    return m1_out, m1_err_out, m2_out, m2_err_out
