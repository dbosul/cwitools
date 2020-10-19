"""Reduction tools directly related to world cooridnate system corrections."""

#Standard Imports

#Third-party Imports
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.signal import correlate
import matplotlib.pyplot as plt
import numpy as np

#Local Imports
from cwitools import coordinates, modeling, utils, synthesis

def rotate(wcs, theta, keep_center=True):
    """Rotate WCS coordinates to new orientation given by theta.

    Analog to ``astropy.wcs.WCS.rotateCD``, which is deprecated since
    version 1.3 (see https://github.com/astropy/astropy/issues/5175).

    Args:
        wcs (astropy.wcs.WCS): The input WCS to be rotated
        theta (float): The rotation angle, in degrees.
        keep_center: Use the center of the image as the reference.
            Otherwise, use the CR keywords.

    Returns:
        astropy.wcs.WCS: The rotated WCS
        
    Example: 
        
        To rotate the an image WCS with respect to *CRVAL*:
        
        >>> wcs_new = rotate(wcs_old, theta)
        
        To rotate with respect to the *center of the FoV*:
        
        >>> wcs_new = rotate(wcs_old, theta, keep_center=True)

    """
    theta = np.deg2rad(theta)
    sinq = np.sin(theta)
    cosq = np.cos(theta)
    mrot = np.array([[cosq, -sinq], [sinq, cosq]])

    # reset center
    if keep_center:
        naxis = np.flip(wcs.array_shape)
        crpix = (np.array(naxis)+1)/2.
        crval = [float(i) for i in wcs.all_pix2world(crpix[0], crpix[1], 1)]

        wcs.wcs.crpix = crpix
        wcs.wcs.crval = crval

    if wcs.wcs.has_cd():    # CD matrix
        newcd = np.dot(mrot, wcs.wcs.cd)
        wcs.wcs.cd = newcd
        wcs.wcs.set()
        return wcs

    if wcs.wcs.has_pc():      # PC matrix + CDELT
        newpc = np.dot(mrot, wcs.wcs.get_pc())
        wcs.wcs.pc = newpc
        wcs.wcs.set()
        return wcs

    raise TypeError("Unsupported wcs type (need CD or PC matrix)")

def xcor_crpix3(fits_list, x_margin=2, y_margin=2):
    """Get relative offsets in wavelength axis by cross-correlating sky spectra.

    Args:
        fits_list (Astropy.io.fits.HDUList list): List of sky cube FITS objects.
        x_margin (int): Margin to use along FITS axis 1 when summing spatially to
            create spectra. e.g. xmargin = 2 - exclude the edge 2 pixels left
            and right from contributing to the spectrum.
        y_margin (int): Margin to use along fits axis 2 when creating spectrum.

    Returns:
        crpix3_corr (list): List of corrected CRPIX3 values.

    """
    #Extract wavelength axes and normalized sky spectra from each fits
    n_fits = len(fits_list)
    wavs, spcs, crval3s, crpix3s = [], [], [], []
    wav0 = -1
    wav1 = 1e6
    for i, sky_fits in enumerate(fits_list):

        sky_data, sky_hdr = sky_fits[0].data, sky_fits[0].header
        sky_data = np.nan_to_num(sky_data, nan=0, posinf=0, neginf=0)

        wav = coordinates.get_wav_axis(sky_hdr)

        sky = np.sum(sky_data[:, y_margin:-y_margin, x_margin:-x_margin], axis=(1, 2))
        sky /= np.max(sky)

        if wav[0] > wav0:
            wav0 = wav[0]
        if wav[-1] < wav1:
            wav1 = wav[-1]

        spcs.append(sky)
        wavs.append(wav)
        crval3s.append(sky_hdr["CRVAL3"])
        crpix3s.append(sky_hdr["CRPIX3"])

    #Create common wavelength axis to interpolate sky spectra onto
    dw_min = np.min([x[1] - x[0] for x in wavs])
    n_wav = int((wav1 - wav0) / dw_min) + 1
    wav_common = np.linspace(wav0, wav1, n_wav)

    #Interpolate (linearly) spectra onto common wavelength axis
    spc_interps = [interp1d(wavs[i], spcs[i])(wav_common) for i in range(n_fits)]

    #Cross-correlate interpolated spectra to look for shifts between them
    corrs = []
    for i, spc_int in enumerate(spc_interps):
        corr_ij = correlate(spc_interps[0], spc_int, mode='full')
        corrs.append(np.nanargmax(corr_ij))

    #Subtract first self-correlation (reference point)
    corrs = corrs[0] -  np.array(corrs)

    #Create new
    crpix3s_corr = [crpix3s[i] + c for i, c in enumerate(corrs)]

    #Return corrections to CRPIX3 values
    return crpix3s_corr

def xcor_2d(hdu0_in, hdu1_in, crval=None, crpix=None, maxstep=None, box=None,
            upscale=1, conv_filter=2., bg_subtraction=False,
            bg_level=None, reset_center=False, method='interp-bicubic',
            output_flag=False, plot=0):
    """Perform 2D cross correlation to image HDUs and returns the relative shifts.

    This function is the base of xcor_crpix12() for frame alignment.

    Args:
        hdu0_in (astropy HDU / HDUList): HDU/HDUList with 2D data for reference.
        hdu1_in (astropy HDU / HDUList): HDU/HDUList with 2D data to be shifted.
        crval (float tuple): RA and DEC of the reference object in HDU0.
        crpix (float tuple): X and Y pixel postions of the reference object in HDU1.
        maxstep (int tupe): Maximum pixel search range in X and Y directions.
            Default is 1/4 of the image size.
        box (int tuple): Specify a certain region in [X0, Y0, X1, Y1] of HDU0 to
            be cross-correlated. Default is the whole image.
        upscale (int): Factor for increased sampling.
        conv_filter (float): Size of the convolution filter when searching for
            the local maximum in the xcor map.
        bg_subtraction (bool): Set to True to apply background subtraction.
            If "bg_level" is not specified, median is used.
        bg_level (float tuple): Background value of the two images.
            Pixels below this level will be ignored.
        reset_center (bool): Quick switch to ignore the WCS information in HDU1 and force its
            center to be the same as HDU0. This overrides the "crval" and "crpix" keywords.
        method (str): Sampling method for sub-pixel interpolations.
            Supported values:
                "interp-nearest-neighbor"
                "interp-bilinear"
                "interp-bicubic" (Default)
                "exact"
        output_flag (bool): If set return [xshift, yshift, flag] even if the
            program failed to locate a local maximum (flag = 0). Otherwise,
            return [xshift, yshift] only if a  local maximum if found.
        plot (int): Make plots?
            0 - No plot.
            1 - Only the xcor map.
            2 - All diagnostic plots.

    Return:
        x_final (float): Amount of shift in X that need to be added to CRPIX1.
        y_final (float): Amount of shift in Y that need to be added to CRPIX2.
        flag (bool) (Only return if output_flag == True.):
            0 - Failed to locate local maximum.  x_final and y_final unreliable.
            1 - Success.

    """

    plot = int(plot)

    # Properties
    hdu0 = utils.extract_hdu(hdu0_in).copy()
    hdu1 = utils.extract_hdu(hdu1_in).copy()
    hdu0.data = np.nan_to_num(hdu0.data, nan=0, posinf=0, neginf=0)
    hdu1.data = np.nan_to_num(hdu1.data, nan=0, posinf=0, neginf=0)
    sz0 = hdu0.shape
    sz1 = hdu1.shape
    wcs0_old = WCS(hdu0.header)
    wcs1_old = WCS(hdu1.header)

    # Defaults
    if maxstep is None:
        maxstep = [sz1[1]/4., sz1[0]/4.]
    maxstep = [int(np.round(i)) for i in maxstep]

    if box is None:
        box = [0, 0, sz0[1], sz0[0]]

    # Preset CRs
    if (crval is not None) != (crpix is not None):
        raise ValueError("'CRVAL' and 'CRPIX' need to be set simultaneously.")

    if crval is not None:
        #hdu1.header['CRPIX1'] = preshift[0]
        #hdu1.header['CRPIX2'] = preshift[1]
        hdu1.header['CRVAl1'] = crval[0]
        hdu1.header['CRVAL2'] = crval[1]
        hdu1.header['CRPIX1'] = crpix[0]
        hdu1.header['CRPIX2'] = crpix[1]

    if reset_center:
        # Quick switch to center each HDU's data
        c00, c01 = sz0[0] / 2 + 0.5, sz0[1] / 2 + 0.5
        c10, c11 = sz1[0] / 2 + 0.5, sz1[1] / 2 + 0.5

        ad_center0 = wcs0_old.all_pix2world(c01, c00, 0)
        ad_center0 = [float(i) for i in ad_center0]

        xy_center0to1 = wcs1_old.all_world2pix(*ad_center0, 0)
        xy_center0to1 = [float(i) for i in xy_center0to1]

        hdu1.header['CRPIX1'] += c11 - xy_center0to1[0]
        hdu1.header['CRPIX2'] += c10 - xy_center0to1[1]


    # Record old values for returning
    old_crpix1 = [hdu1.header['CRPIX1'], hdu1.header['CRPIX2']]
    old_crval1 = [hdu1.header['CRVAL1'], hdu1.header['CRVAL2']]

    wcs0 = WCS(hdu0.header)
    wcs1 = WCS(hdu1.header)

    hdu0 = coordinates.scale_hdu(hdu0, upscale, reproject_mode=method)
    hdu1 = coordinates.scale_hdu(hdu1, upscale, reproject_mode=method)

    # project 1 to 0
    hdu1_scaled = coordinates.reproject_hdu(hdu1, hdu0.header, method=method)
    img1 = hdu1_scaled.data

    img0 = np.nan_to_num(hdu0.data, nan=0, posinf=0, neginf=0)
    img1 = np.nan_to_num(img1, nan=0, posinf=0, neginf=0)

    sz0_sc = int(sz0[0] * upscale), int(sz0[1] * upscale) #Scaled size
    img1_expand = np.zeros((3 * sz0_sc[0], 3 * sz0_sc[1]))
    img1_expand[sz0_sc[0] : 2 * sz0_sc[0], sz0_sc[1] : 2 * sz0_sc[1]] = img1

    # +/- maxstep pix
    xcor_size = ((np.array(maxstep) - 1) * upscale + 1) + int(np.ceil(conv_filter))
    xcor_size = xcor_size.astype(int)
    x_arr = np.linspace(-xcor_size[0], xcor_size[0], 2 * xcor_size[0] + 1, dtype=int)
    y_arr = np.linspace(-xcor_size[1], xcor_size[1], 2 * xcor_size[1] + 1, dtype=int)
    dy_grid, dx_grid = np.meshgrid(y_arr, x_arr)

    xcor = np.zeros(dx_grid.shape)
    box_sc = [b * upscale for b in box]
    box_sc = np.array(box_sc).astype(int)

    for i in range(xcor.shape[0]):
        for j in range(xcor.shape[1]):

            cut0 = img0[box_sc[1]:box_sc[3], box_sc[0]:box_sc[2]]
            cut1 = img1_expand[
                box_sc[1] - dy_grid[i, j] + sz0_sc[0]:box_sc[3] - dy_grid[i, j] + sz0_sc[0],
                box_sc[0] - dx_grid[i, j] + sz0_sc[1]:box_sc[2] - dx_grid[i, j] + sz0_sc[1]
                ]

            if bg_subtraction:
                if bg_level is None:
                    back_val0 = np.median(cut0[cut0 != 0])
                    back_val1 = np.median(cut1[cut1 != 0])
                else:
                    back_val0 = float(bg_level[0])
                    back_val1 = float(bg_level[1])

                cut0 = cut0 - back_val0
                cut1 = cut1 - back_val1
            else:
                if not bg_level is None:
                    cut0[cut0 < bg_level[0]] = 0
                    cut1[cut1 < bg_level[1]] = 0

            cut0[cut0 < 0] = 0
            cut1[cut1 < 0] = 0
            mult = cut0 * cut1

            if np.sum(mult != 0) > 0:
                xcor[i, j] = np.sum(mult) / np.sum(mult != 0)

    # local maxima
    max_conv = ndimage.filters.maximum_filter(xcor, 2 * conv_filter + 1)
    maxima = (xcor == max_conv)
    labeled, _ = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    xindex, yindex = [], []

    for d_x, d_y in slices:
        x_center = (d_x.start + d_x.stop - 1) / 2
        xindex.append(x_center)
        y_center = (d_y.start + d_y.stop-1) / 2
        yindex.append(y_center)

    xindex = np.array(xindex).astype(int)
    yindex = np.array(yindex).astype(int)

    # remove boundary effect
    index = ((xindex >= conv_filter) & (xindex < 2 * xcor_size[0] - conv_filter) &
             (yindex >= conv_filter) & (yindex < 2 * xcor_size[1] - conv_filter))

    xindex = xindex[index]
    yindex = yindex[index]

    # First plots before possible failure.
    if plot != 0:
        if plot == 1:
            fig, axes = plt.subplots(figsize=(6, 6))
        elif plot == 2:
            fig, axes = plt.subplots(3, 2, figsize=(8, 12))
        else:
            raise ValueError('Allowed values for "plot": 0, 1, 2.')

        # xcor map
        if plot == 2:
            axis = axes[0, 0]
        elif plot == 1:
            axis = axes
        xplot = (np.append(x_arr, x_arr[1] - x_arr[0] + x_arr[-1]) - 0.5) / upscale
        yplot = (np.append(y_arr, y_arr[1] - y_arr[0] + y_arr[-1]) - 0.5) / upscale
        colormesh = axis.pcolormesh(xplot, yplot, xcor.T)
        axis.plot([xplot.min(), xplot.max()], [0, 0], 'w--')
        axis.plot([0, 0], [yplot.min(), yplot.max()], 'w--')
        axis.set_xlabel('dx')
        axis.set_ylabel('dy')
        axis.set_title('XCOR_MAP')
        fig.colorbar(colormesh, ax=axis)

        if plot == 2:
            fig.delaxes(axes[0, 1])

            # adu0
            cut0_plot = img0[box_sc[1]:box_sc[3], box_sc[0]:box_sc[2]]
            axis = axes[1, 0]
            imshow = axis.imshow(cut0_plot, origin='bottom')
            axis.set_xlabel('x')
            axis.set_ylabel('y')
            axis.set_title('Ref img')
            fig.colorbar(imshow, ax=axis)

            # adu1
            cut1_plot = img1_expand[box_sc[1] + sz0_sc[0] : box_sc[3] + sz0_sc[0],
                                    box_sc[0] + sz0_sc[1] : box_sc[2] + sz0_sc[1]]
            axis = axes[1, 1]
            imshow = axis.imshow(cut1_plot, origin='bottom')
            axis.set_xlabel('x')
            axis.set_ylabel('y')
            axis.set_title('Original img')
            fig.colorbar(imshow, ax=axis)


            # sub1
            axis = axes[2, 0]
            imshow = axis.imshow(cut1_plot - cut0_plot, origin='bottom')
            axis.set_xlabel('x')
            axis.set_ylabel('y')
            axis.set_title('Original sub')
            fig.colorbar(imshow, ax=axis)

    # closest local maxima
    if len(xindex) == 0:
        # Error handling
        if output_flag:
            return 0., 0., 0., 0., False

        # perhaps we can use the global maximum here, but it is also garbage...
        raise ValueError('Unable to find local maximum in the XCOR map.')

    max_val = np.max(max_conv[xindex, yindex])
    med_val = np.median(xcor)
    index = np.where(max_conv[xindex, yindex] > 0.3 * (max_val - med_val) + med_val)
    xindex = xindex[index]
    yindex = yindex[index]

    if len(xindex) == 0:
        if output_flag:
            return 0., 0., 0., 0., False

        # perhaps we can use the global maximum here, but it is also garbage...
        raise ValueError('Unable to find local maximum in the XCOR map.')

    index = (x_arr[xindex]**2 + y_arr[yindex]**2).argmin()
    xshift = x_arr[xindex[index]] / upscale
    yshift = y_arr[yindex[index]] / upscale

    hdu1 = coordinates.scale_hdu(hdu1, 1 / upscale, header_only=True)
    hdu0 = coordinates.scale_hdu(hdu0, 1 / upscale, header_only=True)
    wcs0 = WCS(hdu0.header)
    wcs1 = WCS(hdu1.header)

    tmp = wcs0.all_pix2world(hdu0.header['CRPIX1'] + xshift, hdu0.header['CRPIX2'] + yshift, 1)
    ashift = float(tmp[0]) - hdu0.header['CRVAL1']
    dshift = float(tmp[1]) - hdu0.header['CRVAL2']

    tmp = wcs1.all_world2pix(old_crval1[0] - ashift, old_crval1[1] - dshift, 1)
    x_final = tmp[0] - old_crpix1[0]
    y_final = tmp[1] - old_crpix1[1]

    crval1_final = old_crval1[0]
    crval2_final = old_crval1[1]
    crpix1_final = old_crpix1[0] + x_final
    crpix2_final = old_crpix1[1] + y_final

    if plot != 0:

        if plot == 1:
            axis = axes
        if plot == 2:
            axis = axes[0, 0]

        axis.plot(xshift, yshift, '+', color='r', markersize=20)

        if plot == 2:
            # sub2
            cut1_best = img1_expand[
                int((box[1] + sz0[0] - yshift) * upscale):int((box[3] + sz0[0] - yshift) * upscale),
                int((box[0] + sz0[1] - xshift) * upscale):int((box[2] + sz0[1] - xshift) * upscale)
                ]
            axis = axes[2, 1]
            imshow = axis.imshow(cut1_best - cut0_plot, origin='bottom')
            axis.set_xlabel('x')
            axis.set_ylabel('y')
            axis.set_title('Best sub')
            fig.colorbar(imshow, ax=axis)


        fig.tight_layout()
        plt.show()

    if output_flag:
        return crpix1_final, crpix2_final, crval1_final, crval2_final, True

    return crpix1_final, crpix2_final, crval1_final, crval2_final

def xcor_crpix12(fits_in, fits_ref, wmask=None, maxstep=None, ra=None, dec=None, box_size=None,
                 crpix=None, pixscale=None, orientation=None, dimension=None, upscale=10.,
                 conv_filter=2., bg_subtraction=False, bg_level=None, reset_center=False,
                 method='interp-bicubic', plot=1):
    """Use cross-correlation to measure the values of CRPIX1/2 and CRVAL1/2.

    This function is a wrapper of xcor_2d() to optimize the reduction process.

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data to be shifted.
        fits_ref (astropy HDU / HDUList): Input HDU/HDUList with 3D data as reference.
        wmask (list): List of wavelength ranges (float tuples), in angstrom, to exclude when making
            the white-light images.
        maxstep (int tupe): Maximum pixel search range in X and Y directions.
            Default is 1/4 of the image size.
        ra (float): Center RA of the fitting box if "box_size" is set. Reference RA of
            CRVAL if "crpix" is set.
        dec (float): Center DEC of the fitting box if "box_size" is set. Reference DEC of
            CRVAL if "crpix" is set.
        box_size (int tuple): Box size in arcsec to be cross-correlated. If set None,
            The whole image is measured.
        crpix (float): Reference pixels in CRPIX. This can be used to reset the initial
            pointing if it is too far off.
        pixscale (float tuple): Size of pixels in X and Y in arcsec of the reference grid.
            Default is the smallest size between X and Y of "fits_ref".
        orienation (float): Position angle of Y axis.
            Default: The same as "fits_ref".
        dimension (float tuple): Size of the reference grid.
            Default: Just enough to contain the whole "fits_ref".
        upscale (int): Factor for increased sampling during the 2nd iteration. This determines
            the output precision.
        conv_filter (float): Size of the convolution filter when searching for the local
            maximum in the xcor map.
        bg_subtraction (bool): Apply background subtraction to the image?
            If "bg_level" is not specified, it uses median as the background.
        bg_level (float tuple): Background value of the two images. Pixels below
            these will be ignored.
        reset_center (bool): Ignore the WCS information in HDU1 and force its center to be
            the same as HDU0.
        method (str): Sampling method for sub-pixel interpolations. Supported values:
            "interp-nearest", "interp-bilinear", "inter-bicubic" (Default), "exact".
        plot (int): Make plots?
            0 - No plot.
            1 - Only the xcor map.
            2 - All diagnostic plots.

    Return:
        crpix1 (float): True value of CRPIX1.
        crpix2 (float): True value of CRPIX2.
        crval1 (float): True value of CRVAL1
        crval2 (float): True value of CRVAL2
        
    Examples:
    
        Suppose there are two exposures with overlapping FoV: hdu0 and hdu1, where we want to shift 
        hdu1 to match hdu0. If the initial headers are not too far off:
        
        >>> crpix1, crpix2, crval1, crval2 = xcor_crpix12(hdu1, hdu0)
        
        However, this may fail when the two headers are off by over a FoV. In this case, you 
        could reset the center,
        
        >>> crpix1, crpix2, crval1, crval2 = xcor_crpix12(hdu1, hdu0, reset_center=True)
        
        or roughly specify the RA and DEC of a known object,
        
        >>> crpix1, crpix2, crval1, crval2 = xcor_crpix12(hdu1, hdu0, ra=ra, dec=dec, crpix=(x,y))

    """

    hdu = utils.extract_hdu(fits_in)
    hdu_ref = utils.extract_hdu(fits_ref)

    # whitelight images
    hdu_img, _ = synthesis.whitelight(hdu, wmask=wmask, mask_sky=True)
    hdu_img_ref, _ = synthesis.whitelight(hdu_ref, wmask=wmask, mask_sky=True)

    ### CHANGED - Use Astropy to get pixel sizes
    wcs = WCS(hdu_img.header)
    pixel_scales = proj_plane_pixel_scales(wcs)
    ps_x = (pixel_scales[0] * u.deg).to(u.arcsec).value
    ps_y = (pixel_scales[1] * u.deg).to(u.arcsec).value

    ### CHANGED - simplified a few lines to one here
    if pixscale is None:
        pixscale_x = pixscale_y = np.min([ps_x, ps_y])
    else:
        pixscale_x, pixscale_y = pixscale

    # Post projection image size
    if dimension is None:
        d_x = int(np.round(ps_x * hdu_ref.shape[2] / pixscale_x))
        d_y = int(np.round(ps_y * hdu_ref.shape[1] / pixscale_y))
        dimension = [d_x, d_y]

    # Construct WCS for the reference HDU in uniform grid
    hdrtmp = hdu_img_ref.header.copy()
    wcstmp = WCS(hdrtmp).copy()
    center = wcstmp.wcs_pix2world(
        (wcstmp.pixel_shape[0] - 1) / 2.,
        (wcstmp.pixel_shape[1] - 1) / 2.,
        0,
        ra_dec_order=True
        )

    hdr0 = hdrtmp.copy()
    hdr0['NAXIS1'] = dimension[0]
    hdr0['NAXIS2'] = dimension[1]
    hdr0['CRPIX1'] = (dimension[0] + 1) / 2.
    hdr0['CRPIX2'] = (dimension[1] + 1) / 2.
    hdr0['CRVAL1'] = float(center[0])
    hdr0['CRVAL2'] = float(center[1])
    old_cd11 = hdr0['CD1_1']
    old_cd21 = hdr0['CD2_1']
    hdr0['CD1_1'] = -pixscale_x / 3600
    hdr0['CD2_2'] = pixscale_y / 3600
    hdr0['CD1_2'] = 0.
    hdr0['CD2_1'] = 0.
    wcs0 = WCS(hdr0)

    # orientation
    if orientation is None:
        orientation = np.degrees(np.arctan(old_cd21 / (-old_cd11)))

    ### CHANGED - USE reduction.rotate() here to simplify code
    wcs_rot = rotate(wcs0, orientation)
    hdr0["CD1_1"] = wcs_rot.wcs.cd[0, 0]
    hdr0["CD1_2"] = wcs_rot.wcs.cd[0, 1]
    hdr0["CD2_1"] = wcs_rot.wcs.cd[1, 0]
    hdr0["CD2_2"] = wcs_rot.wcs.cd[1, 1]
    wcs0 = WCS(hdr0)

    ### CHANGED - Use new reproject_hdu wrapper
    hdu_img_ref0 = coordinates.reproject_hdu(hdu_img_ref, hdr0)

    # Reformat the box parameter
    if box_size is not None:

        if ra is None or dec is None:
            raise ValueError("ra' and 'dec' must be provided along with 'box'")

        box_x, box_y = wcs0.all_world2pix(ra, dec, 0)
        box = [box_x - int(box_size / pixscale_x / 2),
               box_y - int(box_size / pixscale_y /2),
               box_x + int(box_size / pixscale_x /2),
               box_y + int(box_size / pixscale_y /2)]
    else:
        box = None

    # CRs
    if crpix is not None:
        if ra is None or dec is None:
            raise ValueError("'ra' and 'dec' must be provided if 'crpix' is set")
        crval = [ra, dec]
    else:
        crval = None

    # First iteration
    crpix1_tmp, crpix2_tmp, crval1_tmp, crval2_tmp, flag = xcor_2d(
        hdu_img_ref0,
        hdu_img,
        crval=crval,
        crpix=crpix,
        maxstep=maxstep,
        box=box,
        upscale=1,
        conv_filter=conv_filter,
        bg_subtraction=bg_subtraction,
        bg_level=bg_level,
        reset_center=reset_center,
        method=method,
        output_flag=True,
        plot=plot
        )
    if not flag:
        if not reset_center:
            utils.output('\tFirst attempt failed. Trying to recenter\n')
            crpix1_tmp, crpix2_tmp, crval1_tmp, crval2_tmp = xcor_2d(
                hdu_img_ref0,
                hdu_img,
                crval=crval,
                crpix=crpix,
                maxstep=maxstep,
                box=box,
                upscale=1,
                conv_filter=conv_filter,
                bg_subtraction=bg_subtraction,
                bg_level=bg_level,
                reset_center=True,
                method=method,
                output_flag=True,
                plot=plot
                )
        else:
            raise ValueError('Unable to find local maximum in the XCOR map.')

    utils.output('\tFirst iteration:\n')
    utils.output("\t\tCRPIX = %.2f, %.2f; CRVAL1 = %.4f, %.4f\n" %
                 (crpix1_tmp, crpix2_tmp, crval1_tmp, crval2_tmp))

    crpix1, crpix2, crval1, crval2 = xcor_2d(
        hdu_img_ref0,
        hdu_img,
        crval=[crval1_tmp, crval2_tmp],
        crpix=[crpix1_tmp, crpix2_tmp],
        maxstep=[2, 2],
        box=box,
        upscale=upscale,
        conv_filter=conv_filter,
        bg_subtraction=bg_subtraction,
        bg_level=bg_level,
        method=method,
        plot=plot
        )

    utils.output('\tSecond iteration:\n')
    utils.output("\t\tCRPIX = %.2f, %.2f; CRVAL1 = %.4f, %.4f\n" % (crpix1, crpix2, crval1, crval2))

    return crpix1, crpix2, crval1, crval2

def fit_crpix12(fits_in, crval1, crval2, box_size=10, plot=False, std_max=4, crpix12_guess=None):
    """Fit the PSF of a known source to get crpix1/2 and crval1/2.

    Args:
        fits_in (Astropy.io.fits.HDUList): The input data cube as a fits object
        crval1 (float): The RA/CRVAL1 of the known source
        crval2 (float): The DEC/CRVAL2 of the known source
        crpix12_guess (int tuple): The estimated x,y location of the source.
            If none provided, the existing WCS will be used to estimate x,y.
        box_size (float): The size of the box (in arcsec) to use for measuring.

    Returns:
        cpix1 (float): The axis 1 centroid of the source
        cpix2 (float): The axis 2 centroid of the source

    """

    # Convention here is that cube dimensions are (w, y, x)
    # For KCWI - x is the across-slice axis, for PCWI it is y

    #Load input
    hdu = utils.extract_hdu(fits_in)
    cube = hdu.data.copy()
    header3d = hdu.header.copy()

    #Create 2D WCS and get pixel sizes in arcseconds
    header2d = coordinates.get_header2d(header3d)
    wcs2d = WCS(header2d)
    pixel_scales = proj_plane_pixel_scales(wcs2d)
    y_scale = (pixel_scales[1] * u.deg).to(u.arcsec).value
    x_scale = (pixel_scales[0] * u.deg).to(u.arcsec).value

    #Get initial estimate of source position
    if crpix12_guess is None:
        crpix1, crpix2 = wcs2d.all_world2pix(crval1, crval2, 0)
    else:
        crpix1, crpix2 = crpix12_guess

    if np.isnan(crpix1) or np.isnan(crpix2):
        raise ValueError("Problem with input WCS - getting NaN values for initial x,y estimates.\
        \nCheck the input WCS and verify the header values are roughly accurate.")
    #Limit cube to good wavelength range and clean cube
    wavgood0, wavgood1 = header3d["WAVGOOD0"], header3d["WAVGOOD1"]
    wav_axis = coordinates.get_wav_axis(header3d)
    use_wav = (wav_axis > wavgood0) & (wav_axis < wavgood1)
    cube[~use_wav] = 0
    cube = np.nan_to_num(cube, nan=0, posinf=0, neginf=0)

    #Create WL image
    wl_img = np.sum(cube, axis=0)
    wl_img -= np.median(wl_img)

    #Extract box and measure centroid
    box_size_x = box_size / x_scale
    box_size_y = box_size / y_scale

    #Get bounds of box - limited by image bounds.
    x_lo = max(0, int(crpix1 - box_size_x / 2))
    x_hi = min(cube.shape[2] - 1, int(crpix1 + box_size_x / 2 + 1))

    y_lo = max(0, int(crpix2 - box_size_y / 2))
    y_hi = min(cube.shape[1] - 1, int(crpix2 + box_size_y / 2 + 1))

    #Create data structures for fitting
    x_domain = np.arange(x_lo, x_hi)
    y_domain = np.arange(y_lo, y_hi)

    x_prof = np.sum(wl_img[y_lo:y_hi, x_lo:x_hi], axis=0)
    y_prof = np.sum(wl_img[y_lo:y_hi, x_lo:x_hi], axis=1)

    x_prof /= np.max(x_prof)
    y_prof /= np.max(y_prof)

    #Determine bounds for gaussian profile fit
    x_bounds = [
        (0, 10),
        (x_lo, x_hi),
        (0, std_max / x_scale)
        ]
    y_bounds = [
        (0, 10),
        (y_lo, y_hi),
        (0, std_max / y_scale)
        ]

    #Run differential evolution fit on each profile
    x_fit = modeling.fit_model1d(modeling.gauss1d, x_bounds, x_domain, x_prof)
    y_fit = modeling.fit_model1d(modeling.gauss1d, y_bounds, y_domain, y_prof)

    x_center, y_center = x_fit.x[1], y_fit.x[1]

    #Fit Gaussian to each profile
    if plot:

        x_dom_smooth = np.linspace(x_domain[0], x_domain[-1], 10 * len(x_domain))
        y_dom_smooth = np.linspace(y_domain[0], y_domain[-1], 10 * len(y_domain))
        x_prof_model = modeling.gauss1d(x_fit.x, x_dom_smooth)
        y_prof_model = modeling.gauss1d(y_fit.x, y_dom_smooth)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].set_title("Full Image", fontsize=24)
        axes[0, 0].pcolor(wl_img, vmin=0, vmax=wl_img.max())
        axes[0, 0].plot([x_lo, x_lo], [y_lo, y_hi], 'w-')
        axes[0, 0].plot([x_lo, x_hi], [y_hi, y_hi], 'w-')
        axes[0, 0].plot([x_hi, x_hi], [y_hi, y_lo], 'w-')
        axes[0, 0].plot([x_hi, x_lo], [y_lo, y_lo], 'w-')
        axes[0, 0].plot(crpix1, crpix2, 'wx', markersize=15, markeredgewidth=4.0)
        axes[0, 0].plot(x_center + 0.5, y_center + 0.5, 'rx', markersize=15, markeredgewidth=4.0)
        axes[0, 0].set_aspect(y_scale/x_scale)

        axes[0, 1].set_title("%.1f x %.1f Arcsec Box" % (box_size, box_size), fontsize=24)
        axes[0, 1].pcolor(wl_img[y_lo:y_hi, x_lo:x_hi], vmin=0, vmax=wl_img.max())
        axes[0, 1].plot(crpix1 + 0.5 - x_lo, crpix2 + 0.5 - y_lo, 'wx', markersize=15,
                        markeredgewidth=4.0)
        axes[0, 1].plot(x_center + 0.5 - x_lo, y_center + 0.5 - y_lo, 'rx', markersize=15,
                        markeredgewidth=4.0)
        axes[0, 1].set_aspect(y_scale/x_scale)

        axes[1, 0].set_title("X Profile Fit", fontsize=24)
        axes[1, 0].plot(x_domain, x_prof, 'k.-', linewidth=2, label="Data")
        axes[1, 0].plot(x_dom_smooth, x_prof_model, 'r--', linewidth=2, label="Model")
        axes[1, 0].plot([x_center]*2, [0, 1], 'r--')
        axes[1, 0].legend(fontsize=18)

        axes[1, 1].set_title("Y Profile Fit", fontsize=24)
        axes[1, 1].plot(y_domain, y_prof, 'k.-', linewidth=2, label="Data")
        axes[1, 1].plot(y_dom_smooth, y_prof_model, 'r--', linewidth=2, label="Model")
        axes[1, 1].plot([y_center]*2, [0, 1], 'r--')
        axes[1, 1].legend(fontsize=18)

        for ax_i in fig.axes:
            ax_i.set_xticks([])
            ax_i.set_yticks([])
        fig.tight_layout()
        fig.show()
        plt.waitforbuttonpress()
        plt.close()

    #Return
    return x_center + 1, y_center + 1


def fit_crpix3(fits_in, crval3, crpix3_init=None, window=20, plot=False):
    """Fit the location of a known sky emission line and return it

    Args:
        fits_in (Astropy.io.fits.HDUList): The input data cube as a fits object
        crval3 (float): The wavelength of the sky emission line to fit with a Gaussian
        crpix3_init (float): (Optional) The initial estimate of the feature's location, if the input
            WCS is too innacurate for an initial estimate. (By default, it is assumed that the input
            WCS is accurate within +/- w/2, where 'w' is the 'window' argument.)
        window (float): The full size of the wavelength window (in Angstrom) to use for fitting.
            Default is 20A.

    Returns:
        cpix3 (float): The fitted location of the feature at 'CRVAL3' Angstrom.

    """

    #Load input
    hdu = utils.extract_hdu(fits_in)
    cube = hdu.data.copy()
    hd3d = hdu.header.copy()

    #Get sky spectrum and wavelength axis (in Angstrom as well as pixel units)
    sky_spec = np.median(cube, axis=(1, 2))

    pix_axis = np.arange(cube.shape[0])
    wav_axis = coordinates.get_wav_axis(hd3d)
    window_px = window / hd3d["CD3_3"]

    #Get initial estimate of line position, if not provided
    if crpix3_init is None:
        crpix3_init = np.nanargmin(np.abs(wav_axis - crval3))
    else:
        crpix3_init = int(round(crpix3_init))

    #Get fitting window
    fit_mask = np.abs(pix_axis - crpix3_init) <= window_px / 2
    px_low, px_high = crpix3_init - window_px / 2, crpix3_init + window_px / 2
    #Fit a gaussian line to it
    gauss_bounds = [
        (0, 5),
        (px_low, px_high), #
        (0.5 / hd3d["CD3_3"], 10 / hd3d["CD3_3"])
        ]

    wing_left = (pix_axis < px_low) & (pix_axis > px_low - window_px)
    wing_right = (pix_axis > px_high) & (pix_axis < px_high + window_px)

    #Subtract median of wing regions and normalize
    sky_spec -= np.median(sky_spec[wing_left | wing_right])
    sky_spec /= sky_spec.max()

    gauss_fit = modeling.fit_model1d(
        modeling.gauss1d,
        gauss_bounds,
        pix_axis[fit_mask],
        sky_spec[fit_mask]
        )

    if plot:

        pix_axis_smooth = np.linspace(pix_axis[0], pix_axis[-1], 10 * len(pix_axis))
        gauss_model = modeling.gauss1d(gauss_fit.x, pix_axis_smooth)
        fit_label = "Model" if gauss_fit.success else "Model (Failed)"
        fig, axis = plt.subplots(1, 1, figsize=(12, 6))

        axis.set_title("Fitting sky-line at %.2fA" % crval3, fontsize=24)
        axis.step(pix_axis, sky_spec, 'k-', label="Sky Spectrum")
        axis.plot(pix_axis_smooth, gauss_model, 'r-', label=fit_label)
        axis.plot([crpix3_init - window_px / 2] * 2, [0, 1], 'r--')
        axis.plot([crpix3_init + window_px / 2] * 2, [0, 1], 'r--')
        axis.set_ylim([0, 1])
        axis.set_xlabel(r"z-index [px]", fontsize=18)
        axis.set_ylabel(r"Flux Norm", fontsize=18)
        axis.legend(fontsize=18)
        fig.tight_layout()
        fig.show()
        utils.output("Fit success = " + str(gauss_fit.success) + "\n")
        utils.output("CRPIX3 = %.1f\n" % gauss_fit.x[1])
        utils.output("Hit Enter to continue > ")
        input("")
        plt.close()

    if not gauss_fit.success:
        return -1
    return gauss_fit.x[1]
