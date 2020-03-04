"""Tools for masking, smoothing, and extracting regions."""
from astropy import units as u
from astropy import convolution
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from cwitools import coordinates
from cwitools.modeling import fwhm2sigma
from scipy.stats import sigmaclip

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



def get_mask(image, header, reg, fit=True, fit_box=10, width=3, units='sigma',
get_model=False):
    """Get fitted mask of sources based on a DS9 region file.

    Args:
        image (NumPy.ndarray): The input image data.
        header (Astropy Header): The header associated with the image
        reg (string): The path to the DS9 region file
        fit_box (int): The box size to extract/use for fitting each source.
        get_model (bool): Set to TRUE to return the both the mask and model
        width (float): The width of each mask, in standard deviations.
        units (str): Units of the width argument. Options are
            'px' (pixels), 'arcsec' (arcseconds), or 'sigmas' (i.e. width=3
            would mean each mask is set to 3*std_dev of the best-fit Gaussian)


    Returns:
        numpy.ndarray: A mask with source regions labelled sequentially.
        numpy.ndarray: (if get_model = TRUE) A model of the source flux.

    Examples:

        To get a mask representing the sources in a narrowband image ("NB.fits")
        based on a DS9 region file ("mysources.reg"):

        >>> from cwitools import imaging
        >>> from astropy.io import fits
        >>> nb_image, hdr = fits.open("NB.fits", header=True)
        >>> reg = "mysources.reg"
        >>> source_mask = imaging.get_mask(nb_image, hdr, reg)

    """

    if get_model and not(fit):
        raise ValueError("get_model can only be used when fit=TRUE")

    if units == 'sigma' and not(fit):
        raise ValueError("units can only be 'sigma' when fit=TRUE")

    wcs = WCS(header)
    px_scales = proj_plane_pixel_scales(wcs)
    xscale = (px_scales[0] * u.deg).to(u.arcsec).value
    yscale = (px_scales[1] * u.deg).to(u.arcsec).value

    if units == 'px':
        mask_size_x = width
        mask_size_y = width

    elif units == 'arcsec':
        mask_size_x = width / xscale
        mask_size_y = width / yscale

    elif units != 'sigma':
        raise ValueError("units must be 'px', 'arcsec', or 'sigma'")

    yy, xx = np.indices(image.shape)

    if os.path.isfile(reg):
        reg_wcs = pyregion.open(reg) #World coordinates
        reg_img = reg_wcs.as_imagecoord(header=header) #Image coordinates

    else:
        raise FileNotFoundError("%s does not exist." % reg)

    psf_fitter = fitting.LevMarLSQFitter() #Fitter for PSF modeling

    src_mask = np.zeros_like(image) #Mask of continuum (foreground) sources

    if get_model:
        src_model = np.zeros_like(image) #Model of continuum sources

    #Run through sources
    for i, src in enumerate(reg_img):

        #Extract position and correct to 0-indexed
        x, y, rad = src.coord_list
        x -= 1
        y -= 1
        theta = 0

        if fit:
            #Get meshgrid of distance from source
            rr = np.sqrt( (xx-x)**2 + (yy-y)**2 )

            #Cast to integers for indexing
            y = int(round(y))
            x = int(round(x))

            #If outside FOV, ignore spurce
            if np.min(rr) > 1: continue

            #Extract box around source
            fit_box_half = fit_box / 2
            boxLeft = int(round(max(0, y - fit_box_half)))
            boxRight = int(round(min(image.shape[0] - 1, y + fit_box_half + 1)))
            boxBottom = int(round(max(0, x - fit_box_half)))
            boxTop = int(round(min(image.shape[1] - 1, x + fit_box_half + 1)))

            box = image[boxLeft:boxRight, boxBottom:boxTop]

            #Get meshgrid within box, centered on 0
            box_xx, box_yy = np.indices(box.shape, dtype=float)
            box_xx -= (boxTop - boxBottom) / 2.0
            box_yy -= (boxRight - boxLeft) / 2.0

            #Set bounds on model
            fit_bounds = {
                'amplitude':(0, 5 * np.max(box)),
                'x_mean':(-fit_box_half, fit_box_half),
                'y_mean':(-fit_box_half, fit_box_half),
                'x_stddev':(1.5, 4),
                'y_stddev':(1.5, 4)
            }

            #Make initial guess of PSF
            model_guess = models.Gaussian2D(
                amplitude = box.max(),
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

            x, y = model_fit.x_mean, model_fit.y_mean

            theta = model_fit.theta

            if units == 'sigma':

                mask_size_x = width * model_fit.x_stddev
                mask_size_y = width * model_fit.y_stddev

            model_i = model_fit(xx, yy)

            if get_model:
                src_model += model_fit(xx, yy)

        #Create elliptical source mask from fitted PSF
        ellipse_model = models.Ellipse2D(
            x_0 = x,
            y_0 = y,
            theta = theta,
            a = mask_size_x,
            b = mask_size_y
        )
        ellipse_mask = ellipse_model(xx, yy) > 0

        src_mask[ellipse_mask] = i + 1

    if get_model:
        return src_mask, src_model
    else:
        return src_mask


#Return a pseudo-Narrowband image (either SB units or SNR)
def get_pseudo_nb(fits, wav_center, wav_width, wl_sub=True, pos=None, cwing=50,
fit_rad=2, sub_rad=None, smooth=None, smoothtype='box', var=[],
medsub=True, mask_psf=False, fg_mask=[]):
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
        >>> pNB, WL = imaging.get_pseudo_nb(myfits, 4500, 25)

        If there is a QSO in the image at (x, y) = (40, 50) - then we can
        obtain the continuum subtracted version with:

        >>> pNB_sub, WL = imaging.get_pseudo_nb(myfits, 4500, 25, pos=(40, 50))

        Finally, if we want variance estimates on the output, we must provide
        a variance cube:

        >>> myvar = fits.getdata("varcube.fits")
        >>> r = imaging.get_pseudo_nb(myfits, 4500, 25, pos=(40, 50), var=myvar)
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

def slice_fix(image, mask=None, axis=0, scval=3):
    """Perform slice-by-slice median correction in an image.

    Args:
        image (NumPy.ndarray): The input image data.
        mask (NumPy.ndarray): A corresponding mask, used to exclude pixels from
            the median calculation ((>=1)=ignore, 0=use)
        axis (int): The axis along which to subtract.
        scval (float): The sigma-clipping threshold applied before median calculation.

    Returns:
        numpy.ndarray: The slice-by-slice median subtracted image.

    """
    if mask == None: mask = np.zeros_like(image, dtype=bool)
    else: mask = mask > 0

    if axis == 0:
        for yi in range(image.shape[0]):
            sliceclip = sigmaclip(image[yi, ~mask[yi]], low=scval, high=scval)
            image[yi, :] -= np.median(sliceclip.clipped )

    elif axis == 1:
        for xi in range(image.shape[1]):
            sliceclip = sigmaclip(image[~mask[:, xi], xi], low=scval, high=scval)
            image[:, xi] -= np.median(sliceclip.clipped )

    return image

#Function to smooth along wavelength axis
def smooth_nd(data, scale, axes=None, ktype='gaussian', var=False):
    """Smooth along all/any axes of a data cube with a box or gaussian kernel.

    Args:
        cube (numpy.ndarray): The input datacube.
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

    if naxes < 1 or naxes > 3:
        raise ValueError("smooth_nd only works for 1-3 dimensional data.")

    elif naxes == 1 or naxes == 3:

        #Set kernel type
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
            data_copy = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'),
                                           axis=a,
                                           arr=data_copy.copy()
            )

        return data_copy

    else: #i.e. naxis == 2

        #Set kernel type
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
