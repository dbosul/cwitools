"""Tools for generating scientific products from the extracted signal."""
from astropy import units as u
from astropy import convolution
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from cwitools import coordinates
from cwitools.modeling import fwhm2sigma
from scipy.stats import sigmaclip
from skimage import measure

import numpy as np
import os
import pyregion
import warnings

def get_cutout(fits_in, ra, dec, box_size, z=0, fill=0):
    """Extract a box (in pkpc) around a central position from a 2D or 3D FITS.

    Args:
        fits_in (astropy.io.fits.HDUList): The input FITS file.
        ra (float): Right-ascension of box center, in decimal degrees.
        dec (float): Declination of box center, in decimal degrees.
        z (float): Cosmological redshift of the source.
        box_size (float): The size of the box, in proper kiloparsec.
        fill (string): The fill value for box regions outside the image bounds.
            Default: 0.

    Returns:
        astropy.io.fits.HDUList: The FITS with cutout data and header.

    Examples:

        To extract a 2D region of size 250x250 pkpc^2 around the QSO from a
        pseudo-Narrowband image:

        >>> from cwitools import imaging
        >>> from cwitools import parameters
        >>> from astropy.io import fits
        >>> qso_nb_fits = fits.open("QSO_123.fits")
        >>> target_params = parameters.load_params("QSO_123.param")
        >>> qso_cutout = imaging.get_cutout(qso_nb_fits, target_params, 250)

        This method assumes a 1:1 aspect ratio for the spatial axes of the
        input.
    """
    header = fits_in[0].header

    #Get 2D WCS information from cube regardless of 2D or 3D input
    if header["NAXIS"] == 2:
        wcs2d = WCS(fits_in[0].header)

    elif header["NAXIS"] == 3:
        wcs2d = WCS(coordinates.get_header2d(fits_in[0].header))

    else:
        raise ValueError("2D or 3D input only for get_cutout.")

    #Use 2D WCS to calculate central pixels and plate-scale
    pos = tuple(float(x) for x in wcs2d.all_world2pix(ra, dec, 0))
    pkpc_per_px = coordinates.get_pkpc_per_px(wcs2d, z)
    box_size_px = box_size / pkpc_per_px

    #Create modified fits and update spatial axes WCS
    fits_out = fits_in.copy()
    if header["NAXIS"] == 2:
        cutout = Cutout2D(fits[0].data, pos, box_size_px, wcs,
            mode='partial',
            fill_value=fill
        )
        fits_out[0].data = cutout.data

    #Create new cube if input data is 3D
    else:
        new_cube = []
        for i in range(len(fits_in[0].data)):
            layer_cutout = Cutout2D(fits_in[0].data[i], pos, box_size_px, wcs2d,
                mode='partial',
                fill_value=fill
            )
            new_cube.append(layer_cutout.data)
        fits_out[0].data = np.array(new_cube)

    #Update WCS of fits
    fits_out[0].header["CRVAL1"] = ra
    fits_out[0].header["CRVAL2"] = dec
    fits_out[0].header["CRPIX1"] = pos[0]
    fits_out[0].header["CRPIX2"] = pos[1]
    fits_out[0].header["NAXIS1"] = fits_out[0].data.shape[2]
    fits_out[0].header["NAXIS2"] = fits_out[0].data.shape[1]

    #Return
    return fits_out

def get_wl(fits_in,  wmasks=[], var=[]):
    """Get white-light image from cube.

    Args:
        fits_in (astropy FITS object): Input data cube/FITS.
        wmasks (list): List of wavelength tuples to exclude when making
            white-light image. Use to exclude nebular emission or sky lines.
        var (Numpy.ndarray): Variance cube corresponding to input cube

    Returns:
        numpy.ndarray: White-light image
        numpy.ndarray: (if var given) Variance on the white-light image

    """
    #Extract data + meta-data
    cube, hdr = fits_in[0].data, fits_in[0].header
    cube = np.nan_to_num(cube, nan=0, posinf=0, neginf=0)

    pxscales = proj_plane_pixel_scales(WCS(hdr))
    xscale = (pxscales[0] * u.deg).to(u.arcsec).value
    yscale = (pxscales[1] * u.deg).to(u.arcsec).value
    wscale = (pxscales[2] * u.meter).to(u.angstrom).value
    px_size_arcsec2 = xscale * yscale

    #Get conversion from flam to surf brightness
    flam2sb = wscale / px_size_arcsec2

    #Create wavelength masked based on input
    wav_axis = coordinates.get_wav_axis(hdr)
    zmask = np.ones_like(wav_axis, dtype=bool)
    for (w0, w1) in wmasks:
        zmask[(wav_axis > w0) & (wav_axis < w1)] = 0

    wl_img = np.sum(cube[zmask], axis=0)
    wl_img *= flam2sb

    if var != []:
        var = np.nan_to_num(var, nan=0, posinf=0, neginf=0)
        wl_var = np.sum(var[zmask], axis=0)
        wl_var *= flam2sb**2
        return wl_img, wl_var

    else:
        return wl_img

def get_pnb(fits, wav_center, wav_width, wl_sub=True, pos=None, cwing=50,
fit_rad=2, sub_rad=None, smooth=None, smoothtype='box', var=[], medsub=True,
mask_psf=False, fg_mask=[]):
    """Create a pseudo-Narrow-Band (pNB) image from a data cube.

    Args:
        fits (astropy.io.fits.HDUList): The input FITS file.
        wav_center (float): The central wavelength of the narrow-band in Angstrom.
        wav_width (float): The width of the narrow-band image in Angstrom.
        wl_sub (bool): Set to TRUE to scale and subtract a white-light image.
        pos (float tuple): The location the continuum source in x,y to subtract.
            Leave empty to skip white-light subtraction.
        fit_rad (float): The radius around the source to use for scaling the PSF.
        sub_rad (float): The radius around the soruce to use when subtracting.
        smooth (float): Size of smoothing kernel to use prior to subtraction.
        smoothtype (string): Type of smoothing kernel to use:
            'box': 2D Boxcar kernel (size=width)
            'gaussian': 2D Gaussian Kernel (size=full-width at half-max)
        var (NumPy.ndarray): Variance cube associated with input cube, required
            to output associated variance images.
        medsub (bool): Set to TRUE to median subtract WL image and NB (happens
            before scaling of WL image, if performing subtraction.)
        mask_psf (bool): Set to TRUE to mask the spaxels used for fitting.
        fg_mask (numpy.ndarray): Binary mask of continuum sources to mask
            and exclude during median subtraction.

    Returns:
        numpy.ndarray: The pseudo-NB image (WL-subtracted if requested.)
        numpy.ndarray: The white-light (WL) image.
        numpy.ndarray: (If var given as input) The variance on the pNB image.
        numpy.ndarray: (If var given as input) The variance on the WL image

    Examples:

        Note that this algorithm assumes input a cube with units of
        erg/s/cm2/Angstrom, and header wavelength units of Angstrom.

        To obtain a simple pseudo-Narrowband (and white-light image) with no
        subtraction or variance estimation:

        >>> from cwitools import imaging
        >>> from astropy.io import fits
        >>> myfits = fits.open("cube.fits")
        >>> pNB, WL = imaging.get_nb(myfits, 4500, 25)

        If there is a QSO in the image at (x, y) = (40, 50) - then we can
        obtain the continuum subtracted version with:

        >>> pNB_sub, WL = imaging.get_nb(myfits, 4500, 25, pos=(40, 50))

        Finally, if we want variance estimates on the output, we must provide
        a variance cube:

        >>> myvar = fits.getdata("varcube.fits")
        >>> r = imaging.get_nb(myfits, 4500, 25, pos=(40, 50), var=myvar)
        >>> NB, WL, NB_var, WL_var = r //Unpack the output in this order

    """

    #Flag whether variance is being used or not
    usevar = not(var == [])

    #Extract data and header
    cube, header = fits[0].data, fits[0].header
    cube = np.nan_to_num(cube, nan=0, posinf=0, neginf=0)

    #Prep: get some useful structures numbers
    hdr2D = coordinates.get_header2d(header)
    wcs2D = WCS(hdr2D)

    #Get conversion from summed image to SB units
    wavbin = header["CD3_3"] #Assuming units of Angstrom here
    pxScls = (proj_plane_pixel_scales(wcs2D) * u.deg).to(u.arcsec).value
    pxArea = pxScls[0] * pxScls[1]
    im2sb  = wavbin / pxArea

    #Get wavelength axis
    wav_axis = coordinates.get_wav_axis(header)

    #Get parameters for NB image and WL image
    NB_wavLo = wav_center - wav_width / 2
    NB_wavHi = wav_center + wav_width / 2
    WL_wavLo = NB_wavLo - cwing
    WL_wavHi = NB_wavHi + cwing

    #Create NB image - call these A and B
    A, B = coordinates.get_indices(NB_wavLo, NB_wavHi, header)

    #If requested NB is not in range of cube, return empty data
    if B <= 0 or A >= cube.shape[0] - 1:
        warnings.warn("Requested pNB bandpass outside cube range.")
        return [np.zeros_like(cube[0])]*5

    #If it is clipped on left, adjust size
    if A<0:
        warnings.warn("Requested pNB bandpass is clipped by cube range.")
        A = 0

    #Clipped on right, adjust size
    if B>cube.shape[0]-1:
        warnings.warn("Requested pNB bandpass is clipped by cube range.")
        B = -1

    #Create narrowband image and convert to SB units
    NB = np.sum(cube[A:B], axis=0) * im2sb

    #Get indices of WL image - call these C and D (C < A < B < D)
    C, D = coordinates.get_indices(WL_wavLo, WL_wavHi,header)

    if C < 0:
        warnings.warn("White-light image bandpass is clipped by cube range.")
        D = 0

    #Clipped on right, adjust size
    if D > cube.shape[0] - 1:
        warnings.warn("White-light image bandpass is clipped by cube range.")
        D = -1

    #Get WL image and convert to SB units
    WL = (np.sum(cube[C:A], axis=0) + np.sum(cube[B:D], axis=0)) * im2sb

    #If we are propagating error, make variance images
    if usevar:
        #First, replace any NaNs with inf
        var[np.isnan(var)] = 0

        #NB variance image
        NB_var = np.sum(var[A:B], axis=0)
        NB_var *= im2sb**2

        #White-light variance image
        WL_var = np.sum(var[C:A], axis=0) + np.sum(var[B:D], axis=0)
        WL_var *= im2sb**2

    #If user does not provide a mask, use full image
    if fg_mask == []:
        fg_mask = np.zeros_like(NB, dtype=bool)
    else:
        fg_mask == (fg_mask > 0)

    #Median subtract both
    if medsub:
        NB_sigclip = sigmaclip(NB[~fg_mask].copy())
        NB -= np.median(NB_sigclip.clipped)

        WL_sigclip = sigmaclip(WL[~fg_mask].copy())
        WL -= np.median(WL_sigclip.clipped)

    #If smoothing requested
    if smooth!=None:

        NB = smooth_nd(NB, smooth, ktype=smoothtype)
        WL = smooth_nd(WL, smooth, ktype=smoothtype)

        if usevar:

            NB_var =  smooth_nd(NB_var, smooth, ktype=smoothtype, var=True)
            WL_var =  smooth_nd(WL_var, smooth, ktype=smoothtype, var=True)


    #Subtract source if source position provided
    if wl_sub:

        #Use source if provided
        if pos != None:
            yy, xx = np.indices(WL.shape)
            rr_qso = np.sqrt((xx - pos[0])**2 + (yy - pos[1])**2)
            fMask = rr_qso <= fit_rad
            sMask = rr_qso <= sub_rad

        #Use whole image otherwise
        else:
            fMask = np.ones_like(WL, dtype=bool)
            sMask = np.ones_like(WL, dtype=bool)
            mask_psf = False

        scale_factors = NB[fMask] / WL[fMask]
        scale_factors_clipped = sigmaclip(scale_factors)
        S = np.median(scale_factors_clipped.clipped)

        #Scale WL image and variance
        WL *= S

        sMask[WL <= 0] = 0 #Do not subtract negative values
        sMask[fg_mask] = 0 #Do not subtract over the foreground objects

        NB[sMask] -= WL[sMask]

        if mask_psf:
            NB[fMask == 1] = 0

        if usevar:
            WL_var *= (S**2)
            NB_var[sMask] += WL_var[sMask]

    if usevar:
        return NB, WL, NB_var, WL_var

    else:
        return NB, WL
