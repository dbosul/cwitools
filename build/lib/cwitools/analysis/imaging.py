from astropy import units as u
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from cwitools import coordinates
from cwitools.smoothing import smooth_nd
from scipy.stats import sigmaclip

import numpy as np
import pyregion
import warnings

def get_cutout(fits, params, box_size, fill=0):
    """Extract a square region (in pkpc) around a source from a 2D FITS image.

    Args:
        fits (astropy.io.fits.HDUList): The input FITS file.
        params (float): CWITools parameters dictionary.
        box_size (float): The size of the box in pkpc.
        fill (string): The fill value for box regions outside the image bounds.
            Default: 0.

    Returns:
        numpy.ndarray: A 2D square region of side `box_size' centered on `pos`.

    Examples:

        To extract a 2D region of size 250x250 pkpc^2 around the QSO from a
        pseudo-Narrowband image:

        >>> from cwitools import imaging
        >>> from cwitools import parameters
        >>> from astropy.io import fits
        >>> qso_nb_fits = fits.open("QSO_123.fits")
        >>> target_params = parameters.load_params("QSO_123.param")
        >>> qso_cutout = imaging.get_cutout(qso_nb_fits, target_params, 250)

    """
    wcs = WCS(coordinates.get_header2d(fits[0].header))

    pos = wcs.all_world2pix(params["RA"],params["DEC"],0)

    pkpc_per_px = coordinates.get_pkpc_per_px(wcs, params["Z"])
    box_size_px = box_size/pkpc_per_px

    cutout = Cutout2D(fits[0].data, pos, box_size_px, wcs,
    mode='partial', fill_value=fill)

    return cutout.data

def get_source_mask(image, header, reg, src_box=3, model=False, mask_width=3):
    """Get fitted mask of sources based on a DS9 region file.

    Args:
        image (NumPy.ndarray): The input image data.
        header (Astropy Header): The header associated with the image
        reg (string): The path to the DS9 region file
        src_box (int): The box size to extract/use for fitting each source.
        model (bool): Set to TRUE to return the both the mask and model
        mask_width (float): The width of each mask, in standard deviations.

    Returns:
        numpy.ndarray: A mask with source regions labelled sequentially.
        numpy.ndarray: A model of the source flux.

    Examples:

        To get a mask representing the sources in a narrowband image ("NB.fits")
        based on a DS9 region file ("mysources.reg"):

        >>> from cwitools import imaging
        >>> from astropy.io import fits
        >>> nb_image, hdr = fits.open("NB.fits", header=True)
        >>> reg = "mysources.reg"
        >>> source_mask = imaging.get_source_mask(nb_image, hdr, reg)

    """

    yy, xx = np.indices(image.shape)

    reg_wcs = pyregion.open(reg) #World coordinates
    reg_img = reg_wcs.as_imagecoord(header=header) #Image coordinates

    psf_fitter = fitting.LevMarLSQFitter() #Fitter for PSF modeling

    src_mask = np.zeros_like(image) #Mask of continuum (foreground) sources

    if model: src_model = np.zeros_like(image) #Model of continuum sources

    #Run through sources
    for i, src in enumerate(reg_img):

        #Extract position and correct to 0-indexed
        x, y, rad = src.coord_list
        x -= 1
        y -= 1

        #Get meshgrid of distance from source
        rr = np.sqrt( (xx-x)**2 + (yy-y)**2 )

        #Cast to integers for indexing
        y = int(round(y))
        x = int(round(x))

        #If outside FOV, ignore spurce
        if np.min(rr) > 1: continue

        #Extract box around source
        boxLeft = max(0, y - src_box)
        boxRight = min(image.shape[0] - 1, y + src_box + 1)
        boxBottom = max(0, x - src_box)
        boxTop = min(image.shape[1] - 1, x + src_box + 1)

        box = WL[boxLeft:boxRight, boxBottom:boxTop]

        #Get meshgrid within box, centered on 0
        box_xx, box_yy = np.indices(box.shape, dtype=float)
        box_xx -= (boxT - boxB) / 2.0
        box_yy -= (boxR - boxL) / 2.0

        #Set bounds on model
        fit_bounds = {
            'amplitude':(0, 5 * np.max(box)),
            'x_mean':(-src_box, src_box),
            'y_mean':(-src_box, src_box),
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

        #Create elliptical source mask from fitted PSF
        mask_ellipse = models.Ellipse2D(
            x_0 = model_fit.x_mean,
            y_0 = model_fit.y_mean,
            theta = model_fit.theta,
            a = mask_width * model_fit.x_stddev,
            b = mask_width * model_fit.y_stddev
        )

        mask_i = mask_ellipse(xx, yy)
        model_i = model_fit(xx, yy)

        src_model += model_fit(xx, yy)
        src_mask[mask_i] = i

    if model:
        return src_mask, src_model
    else:
        return src_mask


#Return a pseudo-Narrowband image (either SB units or SNR)
def get_pseudo_nb(fits, wav_center, wav_width, pos=None, cwing=50, fit_r=2,\
sub_r=None, smooth=None, smoothtype='box', mask=None, var=[], medsub=True):
    """Create a pseudo-Narrow-Band (pNB) image from a data cube.

    Args:
        fits (astropy.io.fits.HDUList): The input FITS file.
        wav_center (float): The central wavelength of the narrow-band in Angstrom.
        wav_width (float): The width of the narrow-band image in Angstrom.
        pos (float tuple): The location the continuum source in x,y to subtract.
            Leave empty to skip white-light subtraction.
        fit_r (float): The radius around the source to use for scaling the PSF.
        sub_r (float): The radius around the soruce to use when subtracting.
        smooth (float): Size of smoothing kernel to use prior to subtraction.
        smoothtype (string): Type of smoothing kernel to use:
            'box': 2D Boxcar kernel (size=width)
            'gaussian': 2D Gaussian Kernel (size=full-width at half-max)
        var (NumPy.ndarray): Variance cube associated with input cube, required
            to output associated variance images.
        medsub (bool): Set to TRUE to median subtract WL image and NB (happens
            before scaling of WL image, if performing subtraction.)

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
    usevar = False if var == [] else True

    #Extract data and header
    cube, header = fits[0].data, fits[0].header

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

        #White-light variance image
        WL_var = np.sum(var[a:A], axis=0) + np.sum(var[B:b], axis=0)
        WL_var *= im2sb**2

        #NB variance image
        NB_var = np.sum(var[A:B], axis=0)
        NB_var *= im2sb**2

    #If user does not provide a mask, use full image
    if mask == None: mask = np.zeros_like()

    #Median subtract both
    if medsub:

        NB_sigclip = sigmaclip(NB[mask == 0].copy())
        NB -= np.median(NB_sigclip.clipped)

        WL_sigclip = sigmaclip(WL[mask == 0].copy())
        WL -= np.median(WL_sigclip.clipped)


    #If smoothing requested
    if smooth!=None:

        NB = smooth_nd(NB, smooth, ktype=smoothtype)
        WL = smooth_nd(WL, smooth, ktype=smoothtype)

        if usevar:

            NB_var =  smooth_nd(NB_var, smooth, ktype=smoothtype, var=True)
            WL_var =  smooth_nd(WL_var, smooth, ktype=smoothtype, var=True)

    #Calculate scaling factor

    #Subtract source if source position provided
    if pos != None:

        yy, xx = np.indices(WL.shape)
        rr_qso = np.sqrt((xx - pos[0])**2 + (yy - pos[1])**2)
        fMask = rr_qso <= fit_r
        sMask = rr_qso <= sub_r

        scale_factors = NB[fMask] / WL[fMask]
        scale_factors_clipped = sigmaclip(scale_factors)
        S = np.median(scale_factors_clipped.clipped)

        #Scale WL image and variance
        WL *= S
        WL_var *= (S**2)

        sMask[WL <= 0] = 0 #Do not subtract negative values
        sMask[mask != 0] = 0 #Do not subtract over the provided mask

        NB[sMask] -= WL[sMask]
        NB[fMask == 1] = 0

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
            sliceclip = sigmaclip(NB[yi, ~mask[yi]], low=scval, high=scval)
            image[yi, :] -= np.median(sliceclip.clipped )

    elif axis == 1:
        for xi in range(image.shape[1]):
            sliceclip = sigmaclip(NB[~mask[:, xi], xi], low=scval, high=scval)
            image[:, xi] -= np.median(sliceclip.clipped )

    return image
