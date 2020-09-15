"""Tools for extended data reduction."""

#Standard Imports
import warnings

#Third-party Imports
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.signal import correlate
from scipy.stats import sigmaclip, mode
from skimage import measure, morphology
from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon as shapely_polygon
from tqdm import tqdm

import astropy.coordinates
import astropy.stats
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

#Local Imports
from cwitools import coordinates, modeling, utils, synthesis, extraction

def slice_corr(fits_in, mask_reg=None):
    """Perform slice-by-slice median correction for scattered light.

    Args:
        fits_in (HDU or HDUList): The input data cube

    Returns:
        HDU or HDUList (same type as input): The corrected data

    """
    hdu = utils.extractHDU(fits_in)
    data = hdu.data.copy()

    instrument = utils.get_instrument(hdu.header)
    if instrument == "PCWI":
        slice_axis = 1
    elif instrument == "KCWI":
        slice_axis = 2
    else:
        raise ValueError("Unrecognized instrument")

    slice_axis = np.nanargmin(data.shape)
    nslices = data.shape[slice_axis]

    if mask_reg is not None:
        msk2d = extraction.reg2mask(fits_in, mask_reg)[0].data > 0
    else:
        msk2d = np.zeros_like(data[0], dtype=bool)

    #Run through slices
    for slice_i in tqdm(range(nslices)):

        if slice_axis == 1:
            slice_2d = data[:, slice_i, :]
            msk1d = msk2d[slice_i, :]
        elif slice_axis == 2:
            slice_2d = data[:, :, slice_i]
            msk1d = msk2d[:, slice_i]
        else:
            raise RuntimeError("Shortest axis should be slice axis.")

        #Shrink mask if needed to obtain measurement
        while np.count_nonzero(msk1d == 0) < 5:
            msk1d = morphology.binary_erosion(msk1d)


        #Run through wavelength layers
        for layer_j in range(slice_2d.shape[0]):

            xprof = slice_2d[layer_j].copy()[msk1d == 0]

            _, lower, upper = sigmaclip(xprof, low=2, high=2)
            usex = (xprof >= lower) & (xprof <= upper)

            slice_2d[layer_j] -= np.median(xprof[usex])

    return fits_in


def estimate_variance(inputfits, window=50, nmin=30, snrmin=2.5, wmasks=None):
    """Estimates the 3D variance cube of an input cube.

    Args:
        inputfits (astropy.io.fits.HDUList): FITS object to estimate variance of.
        snrmin (float): SNR threshold to use when removing 'objects' during
            variance rescaling.
        nmin (int): Size of regions above 'snrmin' to detect and exclude from
            background SNR distribution (used for rescaling variance.)
        window (int): Wavelength window (Angstrom) to use for local 2D variance estimation.
        wmasks (list): List of wavelength tuples to exclude when estimating variance.
        sclip (float): Sigmaclip threshold to apply when comparing layer-by-layer noise.

    Returns:
        NumPy ndarray: Estimated variance cube

    """
    hdu = utils.extractHDU(inputfits)

    varcube = np.zeros_like(hdu.data)
    z_indices = np.arange(hdu.data.shape)
    wav_axis = coordinates.get_wav_axis(hdu.header)

    #Create wavelength masked based on input
    zmask = np.ones_like(wav_axis, dtype=bool)
    if wmasks is not None:
        for pair in wmasks:
            zmask[(wav_axis > pair[0]) & (wav_axis < pair[1])] = 0
    nzmax = np.count_nonzero(zmask)

    #Loop over wavelength first to minimize repetition of wl-mask calculation
    for z_j in enumerate(z_indices):

        #Get initial width of white-light bandpass in px
        width_px = window / hdu.header["CD3_3"]

        #Create initial white-light mask, centered on j with above width
        vmask = zmask & (np.abs(z_indices - z_j) <= width_px / 2)

        #Grow until minimum number of valid wavelength layers included
        while np.count_nonzero(vmask) < min(nzmax, window / hdu.header["CD3_3"]):
            width_px += 2
            vmask = zmask & (np.abs(z_indices - z_j) <= width_px / 2)

        varcube[z_j] = np.var(hdu.data[vmask], axis=0)

    #Adjust first estimate by rescaling, if set to do so
    varcube_scaled, _ = scale_variance(hdu.data, varcube, n_min=nmin, snr_min=snrmin)

    return varcube_scaled

def scale_variance(data, var, snr_min=3, n_min=50, plot=True, snr_range=(-5, 5), snr_bins=100):
    """Automatically scale an initial 3D variance estimate using background pixels.

    Args:
        data (numpy.ndarray): The 3D data cube we are estimating variance for
        var (numpy.ndarray): The initial 3D variance estimate
        snr_min (float): Signal-to-noise ratio (SNR) threshold to use for iterative scaling method.
            Contiguous regions of size n_min above a SNR of snr_min will be rejected as systematics
            or emission regions, and the scaling will be based only on remaining background regions.
        n_min (int): Minimum size of a contiguous region with SNR > snr_min to count as a systematic
            and be excluded from the variance scaling.
        plot (bool): Set to TRUE to show diagnostic plots.
        snr_range (float tuple): The range of SNR values to use when finding scaling factor. Default
            is -5 to +5.
        snr_bins (int): The number of SNR bins across snr_range to use for generating histograms.
            Scaling factors are determined by best-fit Gaussian models to SNR histograms, assuming
            background (i.e. shot-noise) limited observations. Default: 100


    Returns:
        numpy.ndarray: The rescaled variance estimate
        float: The final rescaling factor, f, such that var_out = f * var_in
    """

    data = np.nan_to_num(data.copy(), nan=0)
    var = np.nan_to_num(var.copy(), nan=np.inf)

    scale_factor = 1
    std_fit = 99
    n_iter = 0

    utils.output("\t%10s %15s %15s %15s\n" % ("iter", "scale_f", "std-dev", "1/std-dev"))
    while abs(std_fit - 1) >= 0.001:

        n_iter += 1
        snr = data / np.sqrt(var)
        #Adjust SNR dist. using latest scale factor
        snr_scaled = snr * scale_factor

        #Segment into regions
        vox_msk = np.abs(snr_scaled) > snr_min
        vox_lab = measure.label(vox_msk)

        #Measure sizes of regions above (in absolute terms) snr min
        reg_props = measure.regionprops_table(vox_lab, properties=['area', 'label'])
        large_regions = reg_props['area'] > n_min

        # Create object mask to exclude these regions
        obj_mask = np.zeros_like(data, dtype=bool)
        for label in reg_props['label'][large_regions]:
            obj_mask[vox_lab == label] = 1

        #Get SNR distribution of non-masked regions
        counts, edges = np.histogram(
            snr_scaled[~obj_mask],
            range=snr_range,
            bins=snr_bins
       )

        #Fit Gaussian model
        centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(edges.size - 1)])
        noisefitter = fitting.LevMarLSQFitter()
        noisemodel0 = models.Gaussian1D(amplitude=counts.max(), mean=0, stddev=1)
        noisemodel1 = noisefitter(noisemodel0, centers, counts)
        std_fit1 = noisemodel1.stddev.value
        fit_cens = np.abs(centers) > 0.5 * std_fit1
        noisemodel2 = noisefitter(noisemodel0, centers[fit_cens], counts[fit_cens])
        std_fit = noisemodel2.stddev.value

        if plot:
            matplotlib.use('TkAgg')
            gs_grid = gridspec.GridSpec(
                1, 1,
                top=0.95,
                bottom=0.12,
                left=0.16,
                right=0.99
           )
            fig = plt.figure(figsize=(14, 14))
            snr_ax = fig.add_subplot(gs_grid[0, 0])
            snr_ax.hist(
                snr_scaled.flatten(),
                facecolor='k',
                range=snr_range,
                bins=snr_bins,
                alpha=0.4,
                label="All Data"
           )
            snr_ax.hist(
                snr_scaled[~obj_mask].flatten(),
                facecolor='g',
                range=snr_range,
                bins=snr_bins,
                alpha=0.5,
                label="Systematics Removed"
           )
            snr_ax.plot(
                centers,
                noisemodel0(centers),
                'k--',
                alpha=0.5,
                linewidth=2.0,
                label="Standard Normal"
           )
            snr_ax.plot(
                centers,
                noisemodel2(centers),
                'k',
                linewidth=2.0,
                label="Best-fit Gaussian"
           )
            snr_ax.set_yscale('log')
            snr_ax.set_ylabel(r"$\mathrm{N_{vox}}$", fontsize=16)
            snr_ax.set_xlabel(r"$\mathrm{SNR}$", fontsize=16)
            snr_ax.tick_params(labelsize=14)
            snr_ax.legend(fontsize=12)
            fig.tight_layout()
            fig.show()

            input("")
            plt.close()

        new_scale_factor = 1 / std_fit
        utils.output(
            "\t%10i %15.5f %15.5f %15.5f\n"
            % (n_iter, scale_factor, std_fit, 1 / std_fit)
       )
        scale_factor *= new_scale_factor

    var_rescale_factor = (1 / scale_factor**2)

    return var * var_rescale_factor, var_rescale_factor

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
    for i, sky_fits in enumerate(fits_list):

        sky_data, sky_hdr = sky_fits[0].data, sky_fits[0].header
        sky_data = np.nan_to_num(sky_data, nan=0, posinf=0, neginf=0)

        wav = coordinates.get_wav_axis(sky_hdr)

        sky = np.sum(sky_data[:, y_margin:-y_margin, x_margin:-x_margin], axis=(1, 2))
        sky /= np.max(sky)

        spcs.append(sky)
        wavs.append(wav)
        crval3s.append(sky_hdr["CRVAL3"])
        crpix3s.append(sky_hdr["CRPIX3"])

    #Create common wavelength axis to interpolate sky spectra onto
    wav0, wav1 = np.min(wavs), np.max(wavs)
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

    This function is the base of xcor_cr12() for frame alignment.

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
    hdu0 = utils.extractHDU(hdu0_in).copy()
    hdu1 = utils.extractHDU(hdu1_in).copy()
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

def xcor_cr12(fits_in, fits_ref, wmask=None, maxstep=None, ra=None, dec=None, box_size=None,
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
        Dimension (float tuple): Size of the reference grid.
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

    """

    hdu = utils.extractHDU(fits_in)
    hdu_ref = utils.extractHDU(fits_ref)

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

def fit_crpix12(fits_in, crval1, crval2, box_size=10, plot=False, std_max=4):
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
    cube = fits_in[0].data.copy()
    header3d = fits_in[0].header

    #Create 2D WCS and get pixel sizes in arcseconds
    header2d = coordinates.get_header2d(header3d)
    wcs2d = WCS(header2d)
    pixel_scales = proj_plane_pixel_scales(wcs2d)
    y_scale = (pixel_scales[1] * u.deg).to(u.arcsec).value
    x_scale = (pixel_scales[0] * u.deg).to(u.arcsec).value

    #Get initial estimate of source position
    crpix1, crpix2 = wcs2d.all_world2pix(crval1, crval2, 0)

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
        (y_lo, x_hi),
        (0, std_max / y_scale)
   ]

    #Run differential evolution fit on each profile
    x_fit = modeling.fit_model1d(modeling.gauss1d, x_bounds, x_domain, x_prof)
    y_fit = modeling.fit_model1d(modeling.gauss1d, y_bounds, y_domain, y_prof)

    x_center, y_center = x_fit.x[1], y_fit.x[1]

    #Fit Gaussian to each profile
    if plot:

        x_prof_model = modeling.gauss1d(x_fit.x, x_domain)
        y_prof_model = modeling.gauss1d(y_fit.x, y_domain)

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
        axes[1, 0].plot(x_domain, x_prof_model, 'r--', linewidth=2, label="Model")
        axes[1, 0].plot([x_center]*2, [0, 1], 'r--')
        axes[1, 0].legend(fontsize=18)

        axes[1, 1].set_title("Y Profile Fit", fontsize=24)
        axes[1, 1].plot(y_domain, y_prof, 'k.-', linewidth=2, label="Data")
        axes[1, 1].plot(y_domain, y_prof_model, 'r--', linewidth=2, label="Model")
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

def rebin(inputfits, bin_xy=1, bin_z=1, vardata=False):
    """Re-bin a data cube along the spatial (x,y) and wavelength (z) axes.

    Args:
        inputfits (astropy FITS object): Input FITS to be rebinned.
        bin_xy (int): Integer binning factor for x,y axes. (Def: 1)
        bin_z (int): Integer binning factor for z axis. (Def: 1)
        vardata (bool): Set to TRUE if rebinning variance data. (Def: True)
        fileExt (str): File extension for output (Def: .binned.fits)

    Returns:
        astropy.io.fits.HDUList: The re-binned cube with updated WCS/Header.

    """


    #Extract useful structures
    data = inputfits[0].data.copy()
    head = inputfits[0].header.copy()

    #Get dimensions & Wav array
    n_z, n_y, n_x = data.shape

    #Get new sizes
    n_z_new = int(n_z // bin_z)
    n_y_new = int(n_y // bin_xy)
    n_x_new = int(n_x // bin_xy)

    #Perform wavelenght-binning first, if bin provided
    if bin_z > 1:

        #Create new data cube shape
        data_zbinned = np.zeros((n_z_new, n_y, n_x))

        #Run through all input wavelength layers and add to new cube
        for z_i in range(n_z_new * bin_z):
            data_zbinned[int(z_i // bin_z)] += data[z_i]

        #Normalize so that units remain as "erg/s/cm2/A"
        if vardata:
            data_zbinned /= bin_z**2
        else:
            data_zbinned /= bin_z

        #Update central reference and pixel scales
        head["CD3_3"] *= bin_z
        head["CRPIX3"] /= bin_z

    else:

        data_zbinned = data

    #Perform spatial binning next
    if bin_xy > 1:

        #Get new shape
        data_xybinned = np.zeros((n_z_new, n_y_new, n_x_new))

        #Run through spatial pixels and add
        for y_i in range(n_y_new * bin_xy):
            for x_i in range(n_x_new * bin_xy):
                xindex = int(x_i // bin_xy)
                yindex = int(y_i // bin_xy)
                data_xybinned[:, yindex, xindex] += data_zbinned[:, y_i, x_i]

        #
        # No normalization needed for binning spatial pixels.
        # Units remain as 'per pixel' but pixel size changes.
        #

        #Update reference pixel
        head["CRPIX1"] /= float(bin_xy)
        head["CRPIX2"] /= float(bin_xy)

        #Update pixel scales
        for key in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]: head[key] *= bin_xy

    else: data_xybinned = data_zbinned

    binned_fits = fits.HDUList([fits.PrimaryHDU(data_xybinned)])
    binned_fits[0].header = head

    return binned_fits

def get_crop_params(fits_in, zero_only=False, pad=0, nsig=3, plot=False):
    """Get optimized crop parameters for crop().

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Returned objects will be of same type as input.

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        zero_only (bool): Set to only crop zero-valued pixels.
        pad (int / int list): Additonal crop Margin on the x/y axes, given as
            either an integer or a tuple of ints specifying the value for each axis.
        nsig (float): Number of sigmas in sigma-clipping.
        plot (bool): Set to True to show diagnostic plots.

    Returns:
        wcrop (int tuple): Padding wavelengths in the z direction.
        ycrop (int tuple): Padding indices in the y direction.
        xcrop (int tuple): Padding indices in the x direction.



    """

    if isinstance(pad, int):
        pad = (pad, pad)

    hdu = utils.extractHDU(fits_in)
    data = hdu.data.copy()
    header = hdu.header.copy().copy()

    # instrument
    inst = utils.get_instrument(header)

    hdu_2d, _ = synthesis.whitelight(hdu, mask_sky=True)

    #Extract data so that axes are [wav, in-slice, across-slice] = [w, y, x]
    if inst == 'KCWI':
        wl_img = hdu_2d.data
    elif inst == 'PCWI':
        wl_img = hdu_2d.data.T
        pad[0], pad[1] = pad[1], pad[0]
    else:
        raise ValueError('Instrument not recognized.')

    npix, nslices = wl_img.shape #Number of in-slice pixels, Number of slice pixels
    # zpad
    wav_axis = coordinates.get_wav_axis(header)

    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    xprof = np.max(data, axis=(0, 1))
    yprof = np.max(data, axis=(0, 2))
    zprof = np.max(data, axis=(1, 2))

    w_0, w_1 = header["WAVGOOD0"], header["WAVGOOD1"]
    z_0, z_1 = coordinates.get_indices(w_0, w_1, header)
    wcrop = w_0, w_1 = [wav_axis[z_0], wav_axis[z_1]]

    # xpad
    xbad = xprof <= 0
    ybad = yprof <= 0

    x_0 = int(np.round(xbad.tolist().index(False) + pad[0]))
    x_1 = int(np.round(len(xbad) - xbad[::-1].tolist().index(False) - 1 - pad[0]))

    xcrop = [x_0, x_1]

    if zero_only:

        y_0 = ybad.tolist().index(False) + pad[1]
        y_1 = len(ybad) - ybad[::-1].tolist().index(False) - 1 - pad[1]

    else:

        bot_pads = np.repeat(np.nan, nslices)
        top_pads = np.repeat(np.nan, nslices)

        for i in range(nslices):

            stripe = wl_img[:, i]
            stripe_clean = stripe[stripe != 0]

            if len(stripe_clean) == 0:
                continue

            stripe_clean_masked, low, high = sigma_clip(
                stripe_clean,
                sigma=nsig,
                return_bounds=True
           )

            med = np.median(stripe_clean_masked.data[~stripe_clean_masked.mask])
            thresh = ((med - low) + (high - med)) / 2
            stripe_abs = np.abs(stripe - med)

            # Run from left to right, counting consecutive values below thresh
            top_pads[i] = 0
            bot_pads[i] = 0
            for j in range(1, npix):
                if stripe_abs[j] > stripe_abs[j-1] and stripe_abs[j] > thresh:
                    top_pads[i] = j

            # Run from right to left, doing the same.
            bot_pads[i] = npix - 1
            for j in range(npix - 2, 1, -1):
                if stripe_abs[j] > stripe_abs[j+1] and stripe_abs[j] > thresh:
                    bot_pads[i] = j

        #Take median of calculated in-slice crops
        y_0 = np.nanmedian(bot_pads) + pad[1]
        y_1 = np.nanmedian(top_pads) - pad[1]

    #Round up to nearest index
    y_0 = int(np.round(y_0))
    y_1 = int(np.round(y_1))
    ycrop = [y_0, y_1]

    if inst == 'PCWI':
        x_0, x_1, y_0, y_1 = y_0, y_1, x_0, x_1
        xcrop, ycrop = ycrop, xcrop

    utils.output("\tAutoCrop Parameters:\n")
    utils.output("\t\tx-crop: %02i:%02i\n" % (x_0, x_1))
    utils.output("\t\ty-crop: %02i:%02i\n" % (y_0, y_1))
    utils.output("\t\tz-crop: %i:%i (%i:%i A)\n" % (z_0, z_1, w_0, w_1))

    if plot:

        x_0, x_1 = xcrop
        y_0, y_1 = ycrop

        xprof_clean = np.max(data[z_0:z_1, y_0:y_1, :], axis=(0, 1))
        yprof_clean = np.max(data[z_0:z_1, :, x_0:x_1], axis=(0, 2))
        zprof_clean = np.max(data[:, y_0:y_1, x_0:x_1], axis=(1, 2))

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        xax, yax, wax = axes
        xax.step(xprof_clean, 'k-', linewidth=2)
        xax.step(range(x_0, x_1), xprof_clean[x_0:x_1], 'b-', linewidth=2)

        lim = xax.get_ylim()
        xax.set_xlabel("X (Axis 2)", fontsize=18)
        xax.plot([x_0, x_0], [xprof.min(), xprof.max()], 'r-')
        xax.plot([x_1-1, x_1-1], [xprof.min(), xprof.max()], 'r-')
        xax.set_ylim(lim)

        yax.step(yprof_clean, 'k-', linewidth=2)
        yax.step(range(y_0, y_1), yprof_clean[y_0:y_1], 'b-', linewidth=2)
        lim = yax.get_ylim()
        yax.set_xlabel("Y (Axis 1)", fontsize=18)
        yax.plot([y_0, y_0], [yprof.min(), yprof.max()], 'r-')
        yax.plot([y_1 - 1, y_1 - 1], [yprof.min(), yprof.max()], 'r-')
        yax.set_ylim(lim)

        wax.step(zprof_clean, 'k-', linewidth=2)
        wax.step(range(z_0, z_1), zprof_clean[z_0:z_1], 'b-', linewidth=2)
        lim = wax.get_ylim()
        wax.plot([z_0, z_0], [zprof.min(), zprof.max()], 'r-')
        wax.plot([z_1 - 1, z_1 - 1], [zprof.min(), zprof.max()], 'r-')
        wax.set_xlabel("Z (Axis 0)", fontsize=18)
        wax.set_ylim(lim)

        for ax_j in fig.axes:
            ax_j.set_yticks([])
            ax_j.tick_params(labelsize=16)
        fig.tight_layout()
        plt.show()


    return wcrop, ycrop, xcrop


def crop(fits_in, wcrop=None, ycrop=None, xcrop=None):
    """Crops an input data cube (FITS).

    The best crop parameters can be determined using get_crop_params. The
    arguments xcrop/ycrop 'auto' here to trim empty rows and columns, and wcrop
    can be set to 'auto' to trim to the WAVGOOD range.
    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        wcrop (int tuple): Wavelength range (Angstrom) to crop z-axis (axis 0)
            to. Use 'auto' to automatically trim to "WAVGOOD" range.
        ycrop (int tuple): Range to crop y-axis (axis 1) to. Use 'auto' to
            trim empty rows. See get_crop_params for more complete method.
        xcrop (int tuple): Range to crop x-axis (axis 2) to. Use 'auto' to
            trim empty rows. See get_crop_params for more complete method.

    Returns:
        HDU / HDUList*: Trimmed FITS object with updated header.
        *Return type matches type of fits_in argument.

    Examples:

        The parameter wcrop (wavelength crop) is in Angstrom, so to crop a
        data cube to the wavelength range 4200-4400A,the usage would be:

        >>> from astropy.io import fits
        >>> from cwitools.reduction import crop
        >>> myfits = fits.open("mydata.fits")
        >>> myfits_cropped = crop(myfits,wcrop=(4200,4400))

        Crop ranges for the x/y axes are given in image coordinates (px).
        They can be given either as straight-forward indices:

        >>> crop(myfits, xcrop=(10,60))

        Or using negative numbers to count backwards from the last index:

        >>> crop(myfits, ycrop=(10,-10))

    """

    #Extract info
    hdu = utils.extractHDU(fits_in)
    data = hdu.data.copy()
    header = hdu.header.copy().copy()

    #Get profiles of each axis
    data[np.isnan(data)] = 0

    #Allow any axis to have simple automatic mode.
    if 'auto' in [xcrop, ycrop, wcrop]:
        x_auto, y_auto, w_auto = get_crop_params(fits_in, zero_only=True)
        if xcrop == 'auto':
            xcrop = x_auto
        if ycrop == 'auto':
            ycrop = y_auto
        if wcrop == 'auto':
            wcrop = w_auto

    #If crop is not set, use entire axis
    if xcrop is None:
        xcrop = [0, data.shape[2]]

    if ycrop is None:
        ycrop = [0, data.shape[1]]

    if wcrop is None:
        zcrop = [0, data.shape[0]]
    else:
        zcrop = coordinates.get_indices(wcrop[0], wcrop[1], header)

    #Crop cube
    crop_data = data[zcrop[0]:zcrop[1], ycrop[0]:ycrop[1], xcrop[0]:xcrop[1]]

    #Change RA/DEC/WAV reference pixels
    header["CRPIX1"] -= xcrop[0]
    header["CRPIX2"] -= ycrop[0]
    header["CRPIX3"] -= zcrop[0]

    trimmed_hdu = utils.matchHDUType(fits_in, crop_data, header)

    return trimmed_hdu

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


def coadd(cube_list, cube_type=None, masks_in=None, var_in=None, pa=None, px_thresh=0.5,
          exp_thresh=0.1, verbose=False, plot=0, drizzle=0):
    """Coadd a list of fits images into a master frame.

    Args:

        cube_list (lists): Input files to be added, specified on one of 3 ways:
            a) A path to a CWITools .list file (must also provide -cube_type)
            b) A Python list of file paths
            c) A Python list of HDU/HDUList objects
        cube_type (str): If cube_list is given as a CWITools .list file, use
            this to specify the main cube type to coadd (e.g. icubes.fits)
        masks_in (list or str): Specification of 3D PCWI/KCWI pipeline masks to
            load and apply to data. Can be given in three ways:
              a) As a list of HDU-like objects (HDU or HDUList)
              b) As a list of file paths
              c) As a cube type (e.g. "mcubes.fits") - this option only works
                 if cube_list is given as a CWITools .list file.
        var_in (list or str): Specification of 3D PCWI/KCWI variance cubes to
            load and use for propagating error. Same rules apply as masks_in.
        px_thresh (float): Minimum fractional pixel overlap.
            This is the overlap between an input pixel and a pixel in the
            output frame. If a given pixel from an input frame covers less
            than this fraction of an output pixel, its contribution will be
            rejected.
        exp_thresh (float): Minimum exposure time, as fraction of maximum.
            If an area in the coadd has a stacked exposure time less than
            this fraction of the maximum overlapping exposure time, it will be
            trimmed from the coadd. Default: 0.1.
        pa (float): The desired position-angle of the output data.
        verbose (bool): Show progress bars and file names.
        drizzle (float): The drizzle factor to use, as a fraction of pixels size.
            E.g. 0.2 will shrink input pixels by 20%.


    Returns:
        astropy.io.fits.HDUList: The stacked FITS with new header.


    Raises:
        RuntimeError: If wavelength scales of input are not equal.

    """
    #
    # DETERMINE USE-CASE
    #

    # Scenario 1 - User provided a .list file
    if isinstance(cube_list, str) and ".list" in cube_list:

        if cube_type is None:
            raise SyntaxError("cube_type must be provided if coadding with a CWITools .list file")

        # Load cube list as dict and overwrite cube_list with list of paths
        int_clist = utils.parse_cubelist(cube_list)
        cube_list = utils.find_files(
            int_clist["ID_LIST"],
            int_clist["INPUT_DIRECTORY"],
            cube_type,
            depth=int_clist["SEARCH_DEPTH"]
            )
        int_hdus = [utils.extractHDU(x) for x in cube_list]

        # Load masks if mask cube type given
        if isinstance(masks_in, str):
            mask_list = utils.find_files(
                int_clist["ID_LIST"],
                int_clist["INPUT_DIRECTORY"],
                masks_in,
                depth=int_clist["SEARCH_DEPTH"]
                )
            mask_hdus = [utils.extractHDU(x) for x in mask_list]
        else:
            mask_hdus = None

        # Load variance if cube type given
        if isinstance(var_in, str):
            var_list = utils.find_files(
                int_clist["ID_LIST"],
                int_clist["INPUT_DIRECTORY"],
                var_in,
                depth=int_clist["SEARCH_DEPTH"]
                )
            var_hdus = [utils.extractHDU(x) for x in var_list]
        else:
            var_hdus = None

    # Scenario 2 - User provides a list of filenames or HDU-like objects
    # Masks and variance in this scenario must also be provided athis way
    elif isinstance(cube_list, list):
        int_hdus = [utils.extractHDU(x) for x in cube_list]

        if isinstance(masks_in, list):
            mask_hdus = [utils.extractHDU(x) for x in masks_in]
        else:
            mask_hdus = None

        if isinstance(var_in, list):
            var_hdus = [utils.extractHDU(x) for x in var_in]
        else:
            var_hdus = None

    else:
        raise SyntaxError("Something is wrong with the given input types for cube_list and/or\
        cube_type. Check and try again.")

    #
    # At this point, we have lists of 3D HDUs for int [msk, var]
    # Next step - prepare some data structures and variables

    drz_f = (1 - drizzle) / 2.0 # Fractional margin to add for drizzle factor
    usemask = mask_hdus is not None #Boolean flags for masking and error prop
    usevar = var_hdus is not None

    footprints = [] #On-sky footprints of each HDU
    wav0s = [] #Lower wavelength limits
    wav1s = [] #Upper wavelength limits
    pas = [] #Position angles
    wscales = [] #Wavelength scales/sampling rates

    #First pass through data to build up the above lists and meta-data
    for i, int_hdu in enumerate(int_hdus):

        #3D Header
        header3d = int_hdu.header

        #On-sky footprint from 2D WCS (from 2D header)
        footprint = WCS(coordinates.get_header2d(header3d)).calc_footprint()

        #Wavelength limits
        wav0 = header3d["CRVAL3"] - (header3d["CRPIX3"] - 1) * header3d["CD3_3"]
        wav1 = wav0 + header3d["NAXIS3"] * header3d["CD3_3"]

        #Position Angle
        if "ROTPA" in header3d:
            pa_i = header3d["ROTPA"]
        elif "ROTPOSN" in header3d:
            pa_i = header3d["ROTPOSN"]
        else:
            warnings.warn("No header key for PA (ROTPA or ROTPOSN) found.")
            pa_i = 0

        # Replace masked voxels with NaN values
        if usemask:
            msk_data = mask_hdus[i].data
            bin_mask = (msk_data == 1) & (msk_data >= 8)
            int_hdu.data[bin_mask] = np.nan

        #Add all of the above to lists
        wscales.append(header3d["CD3_3"])
        footprints.append(footprint)
        wav0s.append(wav0)
        wav1s.append(wav1)
        pas.append(pa_i)

    #If user does not provide a PA, set to mode of input to minimize rotations
    if pa is None:
        pa = mode(np.array(pas)).mode[0]

    # Check that the scale (Ang/px) of each input image is the same
    if len(set(wscales)) != 1:
        raise RuntimeError("ERROR: Wavelength axes must be equal in scale.")


    #
    # STAGE 1: WAVELENGTH ALIGNMENT
    #
    if verbose:
        utils.output("\tAligning wavelength axes...\n")

    # Get common wavelength scale and new wavelength axis
    cd33_0 = wscales[0]
    wav_new = np.arange(min(wav0s) - cd33_0, max(wav1s) + cd33_0, cd33_0)

    # Adjust each cube to be on new wavelength axis
    for i, int_hdu in enumerate(int_hdus):

        # Pad the end of the cube with zeros to reach same length as wav_new
        n_pad_w = len(wav_new) - int_hdu.header["NAXIS3"]

        # Get the wavelength offset between this cube and wav_new
        wav_shift_i = (wav0s[i] - wav_new[0]) / cd33_0

        # Split the wavelength difference into an integer and sub-pixel shift
        int_shift = int(wav_shift_i)
        subpx_shift = wav_shift_i - int_shift

        # Create convolution matrix for subpixel shift
        shift_kernel = np.array([subpx_shift, 1 - subpx_shift])

        #
        # Apply changes
        #

        # 1 - Pad data along z-axis to same length as wav_new
        int_hdu.data = np.pad(
            int_hdu.data,
            ((0, n_pad_w), (0, 0), (0, 0)),
            mode='constant'
            )

        # 2 - Perform integer shift with np.roll
        int_hdu.data = np.roll(int_hdu.data, int_shift, axis=0)

        # 3 - Shift data along axis by convolving with K
        int_hdu.data = np.apply_along_axis(
            lambda m, sk=shift_kernel: np.convolve(m, sk, mode='same'),
            axis=0,
            arr=int_hdu.data
            )

        # 4 - Update header's WCS info for axis 3
        int_hdu.header["NAXIS3"] = len(wav_new)
        int_hdu.header["CRVAL3"] = wav_new[0]
        int_hdu.header["CRPIX3"] = 1

        # Apply steps 1-4 to variance (square the kernel for convolution)
        if usevar:
            var_hdus[i].data = np.pad(
                var_hdus[i].data,
                ((0, n_pad_w), (0, 0), (0, 0)),
                mode='constant'
                )
            # 2 - Perform integer shift with np.roll
            var_hdus[i].data = np.roll(var_hdus[i].data, int_shift, axis=0)

            # 3 - Shift data along axis by convolving with K
            var_hdus[i].data = np.apply_along_axis(
                lambda m, sk=shift_kernel: np.convolve(m, sk**2, mode='same'),
                axis=0,
                arr=var_hdus[i].data
                )

            # 4 - Update header's WCS info for axis 3
            var_hdus[i].header["NAXIS3"] = len(wav_new)
            var_hdus[i].header["CRVAL3"] = wav_new[0]
            var_hdus[i].header["CRPIX3"] = 1

    #
    # Stage 2 - SPATIAL ALIGNMENT
    #
    utils.output("\tMapping pixels from input --> sky --> output frames.\n")

    #Take first 2D header as template for 2D coadd header and get 2D WCS
    hdr0 = coordinates.get_header2d(int_hdus[0].header)
    wcs0 = WCS(hdr0)

    #Get plate-scales and then update WCS to have 1:1 aspect ratio
    dx_0, dy_0 = proj_plane_pixel_scales(wcs0)
    if dx_0 > dy_0:
        wcs0.wcs.cd[:, 0] /= dx_0 / dy_0
    else:
        wcs0.wcs.cd[:, 1] /= dy_0 / dx_0

    #Rotate WCS to the input pa
    wcs0 = rotate(wcs0, pas[i] - pa)

    #Set new WCS - we will use it later to create the canvas
    wcs0.wcs.set()

    # We don't know which corner is which for an arbitrary rotation
    # So, map each vertex to the coadd space
    x_0, y_0 = 0, 0
    x_1, y_1 = 0, 0
    for f_p in footprints:
        ras, decs = f_p[:, 0], f_p[:, 1]
        x_all, y_all = wcs0.all_world2pix(ras, decs, 0)
        x_0 = min(np.min(x_all), x_0)
        y_0 = min(np.min(y_all), y_0)
        x_1 = max(np.max(x_all), x_1)
        y_1 = max(np.max(y_all), y_1)

    #Get required size of the canvas in x, y
    coadd_size_x = int(round((x_1 - x_0) + 1))
    coadd_size_y = int(round((y_1 - y_0) + 1))

    #Get RA/DEC of lower-left corner - to establish WCS reference point
    ra0, dec0 = wcs0.all_pix2world(x_0, y_0, 0)

    #Set the lower corner of the WCS and create a canvas
    wcs0.wcs.crpix[0] = 1
    wcs0.wcs.crval[0] = ra0
    wcs0.wcs.crpix[1] = 1
    wcs0.wcs.crval[1] = dec0
    wcs0.wcs.set()

    #Convert back from WCS to 2D header
    hdr0 = wcs0.to_header()

    #Get size of 3D canvas in wav
    coadd_size_w = len(wav_new)

    #
    # Now that spatial WCS has been figured out - regenerate 3D WCS/Header
    # Do this by copying an input 3D header and updating the necessary fields
    #
    coadd_hdr = int_hdus[0].header.copy()

    # Size of each axis
    coadd_hdr["NAXIS1"] = coadd_size_x
    coadd_hdr["NAXIS2"] = coadd_size_y
    coadd_hdr["NAXIS3"] = coadd_size_w

    # Reference image coordinate
    coadd_hdr["CRPIX1"] = hdr0["CRPIX1"]
    coadd_hdr["CRPIX2"] = hdr0["CRPIX2"]
    coadd_hdr["CRPIX3"] = 1

    # World coordinate position at reference image coordinate
    coadd_hdr["CRVAL1"] = hdr0["CRVAL1"]
    coadd_hdr["CRVAL2"] = hdr0["CRVAL2"]
    coadd_hdr["CRVAL3"] = wav_new[0]

    # Change along axes
    coadd_hdr["CD1_1"] = wcs0.wcs.cd[0, 0]
    coadd_hdr["CD1_2"] = wcs0.wcs.cd[0, 1]
    coadd_hdr["CD2_1"] = wcs0.wcs.cd[1, 0]
    coadd_hdr["CD2_2"] = wcs0.wcs.cd[1, 1]

    # Re-generate a WCS object from this coadd header and get on-sky footprint
    coadd_hdr2d = coordinates.get_header2d(coadd_hdr)
    coadd_wcs = WCS(coadd_hdr2d)
    coadd_fp = coadd_wcs.calc_footprint()

    #Get scales and pixel size of new canvas
    coadd_px_area = coordinates.get_pxarea_arcsec(coadd_hdr2d)

    # Create data structures to store coadded cube and corresponding exposure time mask
    coadd_data = np.zeros((coadd_size_w, coadd_size_y, coadd_size_x))
    coadd_exp = np.zeros_like(coadd_data)

    if usevar:
        coadd_var = np.zeros_like(coadd_data)

    if plot:

        fig1, axis = plt.subplots(1, 1)
        for f_p in footprints:
            axis.plot(-f_p[0:2, 0], f_p[0:2, 1], 'k-')
            axis.plot(-f_p[1:3, 0], f_p[1:3, 1], 'k-')
            axis.plot(-f_p[2:4, 0], f_p[2:4, 1], 'k-')
            axis.plot([-f_p[3, 0], -f_p[0, 0]], [f_p[3, 1], f_p[0, 1]], 'k-')
        for f_p in [coadd_fp]:
            axis.plot(-f_p[0:2, 0], f_p[0:2, 1], 'r-')
            axis.plot(-f_p[1:3, 0], f_p[1:3, 1], 'r-')
            axis.plot(-f_p[2:4, 0], f_p[2:4, 1], 'r-')
            axis.plot([-f_p[3, 0], -f_p[0, 0]], [f_p[3, 1], f_p[0, 1]], 'r-')

        fig1.show()
        plt.waitforbuttonpress()
        plt.close()
        plt.ion()

        grid = gridspec.GridSpec(2, 2)
        fig2 = plt.figure(figsize=(12, 12))
        input_ax = fig2.add_subplot(grid[:1, :])
        sky_ax = fig2.add_subplot(grid[1:, :1])
        coadd_ax = fig2.add_subplot(grid[1:, 1:])

    if verbose:
        pbar = tqdm(total=np.sum([x.data[0].size for x in int_hdus]))

    # Run through each input frame
    for i, int_hdu in enumerate(int_hdus):

        header_i = int_hdu.header
        header2d_i = coordinates.get_header2d(header_i)
        wcs2d_i = WCS(header2d_i)
        px_area_i = coordinates.get_pxarea_arcsec(header_i)

        if "TELAPSE" in header_i:
            t_exp_i = header_i["TELAPSE"]
        elif "EXPTIME" in header_i:
            t_exp_i = header_i["TELAPSE"]
        else:
            warnings.warn("No exposure time (TELAPSE/EXPTIME) keyword found in header. Skipping.")
            continue

        # Create intermediate frame to build up coadd contributions pixel-by-pixel
        build_frame = np.zeros_like(coadd_data)

        #Build frame for variance
        if usevar:
            var_build_frame = np.zeros_like(coadd_data)

        # Fract frame stores a coverage fraction for each coadd pixel
        fract_frame = np.zeros_like(coadd_data)

        # Get wavelength coverage of this FITS as binary mask
        wavmask_i = np.ones(len(wav_new), dtype=bool)
        wavmask_i[wav_new < wav0s[i]] = 0
        wavmask_i[wav_new > wav1s[i]] = 0

        # Convert to a flux-like unit if the input data is in counts
        if "electrons" in int_hdu.header["BUNIT"]:
            int_hdu.data /= t_exp_i
            if usevar:
                var_hdus[i].data /= t_exp_i**2 #Propagate error

        if plot:
            input_ax.clear()
            sky_ax.clear()
            coadd_ax.clear()
            input_ax.set_title("Input Frame Coordinates")
            sky_ax.set_title("Sky Coordinates")
            coadd_ax.set_title("Coadd Coordinates")
            coadd_ax.set_xlabel("X")
            coadd_ax.set_ylabel("Y")
            sky_ax.set_xlabel("RA (hh.hh)")
            sky_ax.set_ylabel("DEC (dd.dd)")
            y_u, x_u = int_hdu.data.shape[1:]
            input_ax.plot([0, x_u], [0, 0], 'k-')
            input_ax.plot([x_u, x_u], [0, y_u], 'k-')
            input_ax.plot([x_u, 0], [y_u, y_u], 'k-')
            input_ax.plot([0, 0], [y_u, 0], 'k-')
            input_ax.set_xlim([-5, x_u+5])
            input_ax.set_ylim([-5, y_u+5])
            #input_ax.plot(qXin,qYin,'ro')
            input_ax.set_xlabel("X")
            input_ax.set_ylabel("Y")
            y_u, x_u = coadd_data.shape[1:]
            coadd_ax.plot([0, x_u], [0, 0], 'r-')
            coadd_ax.plot([x_u, x_u], [0, y_u], 'r-')
            coadd_ax.plot([x_u, 0], [y_u, y_u], 'r-')
            coadd_ax.plot([0, 0], [y_u, 0], 'r-')
            coadd_ax.set_xlim([-0.5, x_u+1])
            coadd_ax.set_ylim([-0.5, y_u+1])
            for f_p in footprints[i:i+1]:
                sky_ax.plot(-f_p[0:2, 0], f_p[0:2, 1], 'k-')
                sky_ax.plot(-f_p[1:3, 0], f_p[1:3, 1], 'k-')
                sky_ax.plot(-f_p[2:4, 0], f_p[2:4, 1], 'k-')
                sky_ax.plot([-f_p[3, 0], -f_p[0, 0]], [f_p[3, 1], f_p[0, 1]], 'k-')
            for f_p in [coadd_fp]:
                sky_ax.plot(-f_p[0:2, 0], f_p[0:2, 1], 'r-')
                sky_ax.plot(-f_p[1:3, 0], f_p[1:3, 1], 'r-')
                sky_ax.plot(-f_p[2:4, 0], f_p[2:4, 1], 'r-')
                sky_ax.plot([-f_p[3, 0], -f_p[0, 0]], [f_p[3, 1], f_p[0, 1]], 'r-')

        # Loop through spatial pixels in this input frame
        for y_j in range(int_hdu.data.shape[1]):
            for x_k in range(int_hdu.data.shape[2]):

                #Get binary mask of good wavelength indices at this x,y position
                msk_jk = wavmask_i & ~np.isnan(int_hdu.data[:, y_j, x_k])

                # Define BL, TL, TR, BR corners of pixel as coordinates
                pix_verts = np.array([
                    [x_k - 0.5 + drz_f, y_j - 0.5 + drz_f],
                    [x_k - 0.5 + drz_f, y_j + 0.5 - drz_f],
                    [x_k + 0.5 - drz_f, y_j + 0.5 - drz_f],
                    [x_k + 0.5 - drz_f, y_j - 0.5 + drz_f]
                    ])

                # Convert these vertices to RA/DEC positions
                pix_verts_radec = wcs2d_i.all_pix2world(pix_verts, 0)

                # Convert the RA/DEC vertex values into coadd frame coordinates
                pix_verts_coadd = coadd_wcs.all_world2pix(pix_verts_radec, 0)

                #Create polygon object for projection onto coadd grid
                pix_projection = shapely_polygon(pix_verts_coadd)

                if plot:
                    input_ax.plot(pix_verts[:, 0], pix_verts[:, 1], 'kx')
                    sky_ax.plot(-pix_verts_radec[:, 0], pix_verts_radec[:, 1], 'kx')
                    coadd_ax.plot(pix_verts_coadd[:, 0], pix_verts_coadd[:, 1], 'kx')

                #Get bounds of pixel projection in coadd frame
                proj_bounds = list(pix_projection.exterior.bounds)

                # xb0 is x-bound-lower, yb1 is y-bound-upper, etc.
                xb0, yb0, xb1, yb1 = (int(round(pib)) for pib in proj_bounds)

                # Upper bounds need to be increased to include full pixel
                xb1 += 1
                yb1 += 1

                # Loop through relevant pixels in output/coadd frame ('coadd' denoted '_c')
                for x_c in range(xb0, xb1):
                    for y_c in range(yb0, yb1):

                        # Create shapely_polygon object for this coadd pixel
                        coadd_pixel = shapely_box(x_c - 0.5, y_c - 0.5, x_c + 0.5, y_c + 0.5)

                        # Get overlap between input pixel and this coadd pixel
                        overlap = pix_projection.intersection(coadd_pixel).area

                        # Convert to fraction of total input pixel area
                        overlap /= pix_projection.area

                        # Extract spectrum
                        spc_in = int_hdu.data[msk_jk, y_j, x_k]

                        # Extract variance
                        if usevar:
                            var_jk = var_hdus[i].data[msk_jk, y_j, x_k].copy()
                            var_jk *= (overlap**2)

                        #Update all relevant frames
                        try:
                            fract_frame[msk_jk, y_c, x_c] += overlap
                            build_frame[msk_jk, y_c, x_c] += overlap * spc_in
                            if usevar:
                                var_build_frame[msk_jk, y_c, x_c] += var_jk

                        except IndexError:
                            continue
                            #TODO - Fix the issue of edge pixels being trimmed
                            #Need to expand coadd canvas to include all pixels.
                if verbose:
                    pbar.update(1)

        if plot:
            fig2.canvas.draw()
            plt.waitforbuttonpress()

        #Calculate ratio of coadd pixel area to input pixel area
        px_area_ratio = coadd_px_area / px_area_i

        # Max value in fract_frame should be px_area_ratio; it's the biggest
        # fraction of an input pixel that can add to one coadd pixel
        # We want to use this map now to create a flat_frame - where the
        # values represent a covering fraction for each pixel
        # i.e. 0.1 = 10% of pixel area covered by input frames
        flat_frame = fract_frame / px_area_ratio

        #Replace zero-values with inf values to avoid division by zero
        flat_frame[flat_frame == 0] = np.inf

        # Perform flat field correction for pixels that are not fully covered
        build_frame /= flat_frame

        # Zero any pixels below user-set pixel threshold. Set flat value to inf
        build_frame[flat_frame < px_thresh] = 0

        # Propagate variance on previous two steps
        if usevar:
            var_build_frame /= (flat_frame)**2
            var_build_frame[flat_frame < px_thresh] = np.inf

        #Replace values < px_thresh with inf also
        flat_frame[flat_frame < px_thresh] = np.inf

        #Add to exposure mask
        coadd_exp += t_exp_i * (flat_frame < np.inf)

        # Add weight * data to coadd
        coadd_data += t_exp_i * build_frame

        # Propagate error on the above step
        if usevar:
            coadd_var += (t_exp_i**2) * var_build_frame

    if verbose:
        pbar.close()

    if plot:
        plt.close()

    utils.output("\tTrimming coadded canvas.\n")

    # Create 1D exposure time profiles
    exp_zprof = np.mean(coadd_exp, axis=(1, 2))
    exp_xprof = np.mean(coadd_exp, axis=(0, 1))
    exp_yprof = np.mean(coadd_exp, axis=(0, 2))

    # Normalize the profiles
    exp_zprof /= np.max(exp_zprof)
    exp_xprof /= np.max(exp_xprof)
    exp_yprof /= np.max(exp_yprof)

    # Convert 0s to +1NF in exposure time cube
    coadd_exp[coadd_exp == 0] = np.inf

    # Divide by total exposure time, or square for variance
    coadd_data /= coadd_exp
    if usevar:
        coadd_var /= coadd_exp**2

    #Exposure time threshold, relative to maximum exposure time, below which to crop.
    use_z = exp_zprof > exp_thresh
    use_x = exp_xprof > exp_thresh
    use_y = exp_yprof > exp_thresh

    #Trim the data
    coadd_data = coadd_data[use_z]
    coadd_data = coadd_data[:, use_y]
    coadd_data = coadd_data[:, :, use_x]

    #Trim variance
    if usevar:
        coadd_var = coadd_var[use_z]
        coadd_var = coadd_var[:, use_y]
        coadd_var = coadd_var[:, :, use_x]

    #Update the WCS to account for trimmed pixels
    coadd_hdr["CRPIX3"] -= np.argmax(use_z)
    coadd_hdr["CRPIX2"] -= np.argmax(use_y)
    coadd_hdr["CRPIX1"] -= np.argmax(use_x)

    # Create FITS object matching the input type (i.e. HDU or HDUList)
    coadd_fits = utils.matchHDUType(int_hdus[0], coadd_data, coadd_hdr)

    if usevar:
        coadd_var_fits = utils.matchHDUType(var_hdus[0], coadd_var, coadd_hdr)
        return coadd_fits, coadd_var_fits

    return coadd_fits

def air2vac(fits_in, mask=False):
    """Covert wavelengths in a cube from standard air to vacuum.

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        mask (bool): Set if the cube is a mask cube.

    Returns:
        HDU / HDUList*: Trimmed FITS object with updated header.
        *Return type matches type of fits_in argument.

    """

    hdu = utils.extractHDU(fits_in)
    hdu = hdu.copy()
    cube = np.nan_to_num(hdu.data, nan=0, posinf=0, neginf=0)
    hdr = hdu.header

    if hdr['CTYPE3'] == 'WAVE':
        utils.output("\tFITS already in vacuum wavelength.\n")
        return fits_in

    wave_air = coordinates.get_wav_axis(hdr)
    wave_vac = pyasl.airtovac2(wave_air)

    # resample to uniform grid
    cube_new = np.zeros_like(cube)
    for i in range(cube.shape[2]):
        for j in range(cube.shape[1]):

            spec0 = cube[:, j, i]
            if not mask:
                f_cubic = interp1d(
                    wave_vac,
                    spec0,
                    kind='cubic',
                    fill_value='extrapolate'
                    )
                spec_new = f_cubic(wave_air)
            else:
                f_pre = interp1d(
                    wave_vac,
                    spec0,
                    kind='previous',
                    bounds_error=False,
                    fill_value=128
                    )
                spec_pre = f_pre(wave_air)

                f_nex = interp1d(
                    wave_vac,
                    spec0,
                    kind='next',
                    bounds_error=False,
                    fill_value=128
                    )
                spec_nex = f_nex(wave_air)

                spec_new = np.zeros_like(spec0)
                for k in range(spec0.shape[0]):
                    spec_new[k] = max(spec_pre[k], spec_nex[k])

            cube_new[:, j, i] = spec_new

    hdr['CTYPE3'] = 'WAVE'
    hdu_new = utils.matchHDUType(fits_in, cube_new, hdr)

    return hdu_new


def heliocentric(fits_in, mask=False, return_vcorr=False, resample=True, vcorr=None,
                 barycentric=False):
    """Apply heliocentric correction to the cubes.

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        mask (bool): Set if the cube is a mask cube. This only works for
            resampled cubes.
        return_vcorr (bool): If set, return the correction velocity (in km/s)
            as well.
        resample (bool): Resample the cube to the original wavelength grid?
        vcorr (float): Use a different correction velocity.
        barycentric (bool): Use barycentric correction instead of helocentric.

    Returns:
        HDU / HDUList*: Trimmed FITS object with updated header.
        vcorr (float): (if vcorr is True) Correction velocity in km/s.
        *Return type matches type of fits_in argument.

    """

    hdu = utils.extractHDU(fits_in)
    hdu = hdu.copy()
    cube = np.nan_to_num(hdu.data, nan=0, posinf=0, neginf=0)
    hdr = hdu.header

    v_old = 0.
    if 'VCORR' in hdr:
        v_old = hdr['VCORR']
        utils.output("\tRolling back the existing correction with:\n")
        utils.output("\t\tVcorr = %.2f km/s.\n" % (v_old))

    if vcorr is None:
        targ = astropy.coordinates.SkyCoord(
            hdr['TARGRA'],
            hdr['TARGDEC'],
            unit='deg',
            obstime=hdr['DATE-BEG']
            )
        keck = astropy.coordinates.EarthLocation.of_site('Keck Observatory')
        if barycentric:
            vcorr = targ.radial_velocity_correction(kind='barycentric', location=keck)
        else:
            vcorr = targ.radial_velocity_correction(kind='heliocentric', location=keck)
        vcorr = vcorr.to('km/s').value

    utils.output("\tHelio/Barycentric correction:\n")
    utils.output("\t\tVcorr = %.2f km/s.\n" % (vcorr))

    v_tot = vcorr-v_old

    if not resample:

        hdr['CRVAL3'] = hdr['CRVAL3'] * (1 + v_tot / 2.99792458e5)
        hdr['CD3_3'] = hdr['CD3_3'] * (1 + v_tot / 2.99792458e5)
        hdr['VCORR'] = vcorr
        hdu_new = utils.matchHDUType(fits_in, cube, hdr)
        if not return_vcorr:
            return hdu_new
        return hdu_new, vcorr

    wav_old = coordinates.get_wav_axis(hdr)
    wav_hel = wave_old * (1 + v_tot / 2.99792458e5)

    # resample to uniform grid
    cube_new = np.zeros_like(cube)
    for i in range(cube.shape[2]):
        for j in range(cube.shape[1]):

            spc0 = cube[:, j, i]
            if not mask:
                f_cubic = interp1d(wav_hel, spc0, kind='cubic', fill_value='extrapolate')
                spec_new = f_cubic(wav_old)

            else:
                f_pre = interp1d(wav_hel, spc0, kind='previous', bounds_error=False, fill_value=128)
                spec_pre = f_pre(wav_old)
                f_nex = interp1d(wav_hel, spc0, kind='next', bounds_error=False, fill_value=128)
                spec_nex = f_nex(wav_old)

                spec_new = np.zeros_like(spc0)
                for k in range(spc0.shape[0]):
                    spec_new[k] = max(spec_pre[k], spec_nex[k])

            cube_new[:, j, i] = spec_new

    hdr['VCORR'] = vcorr
    hdu_new = utils.matchHDUType(fits_in, cube_new, hdr)

    if not return_vcorr:
        return hdu_new

    return hdu_new, vcorr

def update_cov_header(fits_in, params):
    """Update FITS header to the given parameters of the covariance curve.

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        params (list): List of covariance model parameters:
            alpha - coefficient of Log(K) term
            norm - normalization for curved regime (K < thresh)
            thresh - threshold in kernel size between flat/curved regime

    Returns:
        HDU / HDUList*: Modified HDU/HDUList

    """
    hdu = utils.extractHDU(fits_in)
    data = hdu.data.copy()
    hdr = hdu.header.copy()
    alpha, norm, thresh = params
    beta = norm * (1 + alpha * np.log(thresh))
    hdr["COV_ALPH"] = alpha
    hdr["COV_NORM"] = norm
    hdr["COV_THRE"] = thresh
    hdr["COV_BETA"] = beta
    fits_out = utils.matchHDUType(fits_in, data, hdr)
    return fits_out

def fit_covar_xy(fits_in, var, mask=None, wrange=None, xybins=None, nw=100, wavgood=True,
                 return_all=False, model_bounds=None, mask_sky=True, mask_neb=None, plot=False):
    """Fits a two-component model to the noise as a function of bin size.

    The model used can be found in modeling.covar_curve

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        var (np.array): Variance cube.
        mask (np.array): Mask cube, M, where M > 0 excludes pixels.
        xybins (np.array): List of spatial bin sizes. Default is a list of
            10 poiints evenly distributed between 1 and 1/5 of the shortest
            spatial axis.
        nw (int): Number of independent estimates in each bin size. This is
            done by grouping the independent wavelength layers.
        wrange (tuple): Lower and higher range in wavelength that the curve
            is extracted from.
        wavgood (bool): Shortcut to use the good wavelength range from header.
        model_bounds (float tuple): Lower and upper bounds on the parameters
            of the model (see modeling.covar_curve). The parameters and default
            bounds are as follows:
                alpha (0.1 - 10)
                norm (1 - 1) (by default, no re-normalization enabled)
                threshold (15 - 60)
                beta (1 - 5)
        mask_sky (bool): Set to TRUE to auto-mask some sky lines
        mask_neb (float): Provide redshift to mask common nebular lines
        return_all (bool): If set, also return the independently measured data
            points.

    Returns:
        HDU / HDUList*: Curve parameters recorded in the FITS header.
        param (np.array): Parameters (alpha, norm) that can be used to recover
            the rescaling curve. Only return if return_all == True.
        bin_sizes (np.array): Bin sizes for the independently measured data
            points. Only returned if return_all == True.
        noise_ratios (np.array): Rescaling factor (ratio of actual noise to
            naively propagated noise) for the independely measured data points.
            Only returned if return_all ==True.

    """

    hdu = utils.extractHDU(fits_in)
    data = hdu.data.copy()
    hdr = hdu.header.copy()

    var = var.copy()

    # Fitting
    if model_bounds is None:
        model_bounds = [(0.1, 10), (1, 1), (15, 60)]

    # Create empty mask if none given needed. Make sure dtype is integer
    if mask is None:
        mask = np.zeros_like(data, dtype=int)
    else:
        mask = mask.astype(int)

    # Apply mask
    var[mask != 0] = 0
    data[mask != 0] = 0

    # Filter data for bad values
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    var = np.nan_to_num(var, nan=0, posinf=0, neginf=0)

    # Get wavelength axis for masking
    wav_axis = coordinates.get_wav_axis(hdr)

    # Create binary mask of z-layers to exclude
    zmask = np.zeros_like(wav_axis, dtype=bool)
    if wrange is not None:
        zmask[(wav_axis < wrange[0]) | (wav_axis > wrange[1])] = 1
    if wavgood:
        zmask[(wav_axis < hdr['WAVGOOD0']) | (wav_axis > hdr['WAVGOOD1'])] = 1
    if mask_sky:
        skymask = utils.get_skymask(hdr)
        zmask = zmask | skymask
    if mask_neb is not None:
        nebmask = utils.get_nebmask(hdr, z=mask_neb, vel_window=2000)
        zmask = zmask | nebmask

    # Limit all data to non-masked wavelength layers
    wav_axis = wav_axis[~zmask]
    data = data[~zmask]
    var = var[~zmask]
    mask = mask[~zmask]

    # Bin the data, variance and mask cubes in spatial axes by binsize
    def resize(cube_in, var_in, mask_in, binsize):

        cube = cube_in.copy()
        mask = mask_in.copy()
        var = var_in.copy()

        if binsize == 1:
            return cube, var, mask

        binsize = int(binsize)

        # Adjust size of cube to be even multiple of binsize
        shape = cube.shape
        if shape[1] % binsize != 0:
            cube = cube[:, 0 : shape[1] - shape[1] % binsize, :]
            mask = mask[:, 0 : shape[1] - shape[1] % binsize, :]
            var = var[:, 0 : shape[1] - shape[1] % binsize, :]
        if shape[2] % binsize != 0:
            cube = cube[:, :, 0 : shape[2] - shape[2] % binsize]
            mask = mask[:, :, 0 : shape[2] - shape[2] % binsize]
            var = var[:, :, 0 : shape[2] - shape[2] % binsize]

        # Update shape
        shape = cube.shape

        # Create new shape for the purpose of rebinning
        shape_new = (
            shape[0],
            int(shape[1] / binsize),
            binsize,
            int(shape[2]/binsize),
            binsize
            )
        cube_reshape = cube.reshape(shape_new)
        var_reshape = var.reshape(shape_new)
        mask_reshape = mask.reshape(shape_new)

        # If a bin contains mask == 1, set whole binned mask voxel to 1
        for k in range(shape_new[0]):
            for i in range(shape_new[1]):
                for j in range(shape_new[3]):
                    msk_bin = mask_reshape[k, i, :, j, :]
                    if 1 in msk_bin:
                        mask_reshape[k, i, :, j, :] = 1

        #Recover final, binned versions
        cube_binned = cube_reshape.sum(-1).sum(2)
        var_binned = var_reshape.sum(-1).sum(2)
        mask_binned = mask_reshape.max(-1).max(2)

        return cube_binned, var_binned, mask_binned

    # Calculate noise ratio as a function of spatial bin size
    bin_sizes = []
    noise_ratios = []
    if xybins is None:
        bin_grid = np.arange(1, np.min(data.shape[1:3])/5).astype(int)
    else:
        bin_grid = np.array(xybins).astype(int)

    # Get indices of a set of 'nw' evenly-spaced wavelength layers throughout cube
    # This is done to extract a sub-cube made up of independent z-layers
    z_indices = np.arange(0, data.shape[0] - nw, nw).astype(int)

    # 'z_shift' shifts these indices along by 1 each time, selecting a different
    # sub-cube made up of independent z-layers
    for z_shift in tqdm(range(nw)):
        for bin_i in np.flip(bin_grid):

            #Extract sub cube and sub mask
            subcube = data[z_indices + z_shift, :, :].copy()
            submask = mask[z_indices + z_shift, :, :].copy()
            subvar = var[z_indices + z_shift, :, :].copy()

            #Bin
            cube_b, var_b, mask_b = resize(subcube, subvar, submask, bin_i)

            #Get binary mask of useable vox (mask is dtype int)
            use_vox = (mask_b == 0)

            #Skip if fewer than 10 useable voxels remain in binned cube
            if np.count_nonzero(use_vox) < 10:
                continue

            # Measure the error in the binned cube
            actual_err = np.std(cube_b[use_vox])
            propagated_err = np.sqrt(np.median(var_b[use_vox]))

            # Append bin size and noise ratio to lists
            if np.isfinite(actual_err / propagated_err):
                bin_sizes.append(bin_i)
                noise_ratios.append(actual_err / propagated_err)

    bin_sizes = np.array(bin_sizes)
    noise_ratios = np.array(noise_ratios)

    noise_ratios_clipped = []
    bin_sizes_clipped = []

    #Sigma-clip
    for b_s in bin_sizes:

        indices = bin_sizes == b_s
        noise_ratios_b = noise_ratios[indices]
        noise_ratios_b_clipped = sigmaclip(noise_ratios_b, low=3, high=3).clipped

        for nrc in noise_ratios_b_clipped:
            noise_ratios_clipped.append(nrc)
            bin_sizes_clipped.append(b_s)

    kernel_areas = np.array(bin_sizes_clipped)**2
    noise_ratios = np.array(noise_ratios_clipped)

    model_fit = modeling.fit_model1d(
        modeling.covar_curve,
        model_bounds,
        kernel_areas,
        noise_ratios
        )
    params = model_fit.x

    if plot:

        #Data for plotting

        model = modeling.covar_curve(model_fit.x, kernel_areas)
        residuals = (model - noise_ratios) / noise_ratios
        kareas_smooth = np.linspace(kernel_areas.min(), kernel_areas.max(), 1000)
        model_smooth = modeling.covar_curve(model_fit.x, kareas_smooth)

        #Plotting
        grid = gridspec.GridSpec(
            1, 2,
            width_ratios=[1, 0.5],
            bottom=0.15,
            top=0.95,
            left=0.12,
            right=0.95
            )
        fig = plt.figure(figsize=(20, 16))
        covarax = fig.add_subplot(grid[0, 0])
        histax = fig.add_subplot(grid[0, 1])
        covarax.plot(kernel_areas, noise_ratios, 'ko', alpha=0.2, label="Data")
        covarax.plot(kareas_smooth, model_smooth, 'r-', label="Model")
        covarax.plot([model_fit.x[2]]*2, [0, noise_ratios.max()], 'b--', label="Threshold")
        histax.hist(
            residuals,
            range=(-0.4, 0.4),
            bins=20,
            facecolor='k',
            edgecolor='w',
            alpha=0.75
            )
        covarax.legend(fontsize=12)
        covarax.set_xlim([kernel_areas.min() - 1, kernel_areas.max() + 1])
        covarax.set_ylim([noise_ratios.min(), noise_ratios.max()])
        axlabelsize = 18
        covarax.set_xlabel(r"$\mathrm{K~[px^2]}$", fontsize=axlabelsize)
        covarax.set_ylabel(r"$\mathrm{\sigma_{obs}/\sigma_{ideal}}$", fontsize=axlabelsize)
        histax.set_xlabel(r"$\mathrm{\frac{\sigma_{model}}{\sigma_{obs}} - 1}$",
                          fontsize=axlabelsize)
        histax.set_ylabel(r"$\mathrm{N}$", fontsize=axlabelsize)
        histax.set_yticks([])
        histax.set_xticks([-0.2, 0.2,])
        histax.set_xlim([-0.4, 0.4])
        for ax_i in fig.axes:
            ax_i.tick_params(labelsize=14)
        fig.tight_layout()
        fig.show()
        input("")

    #  Get output FITS with updated header
    fits_out = update_cov_header(fits_in, params)

    if return_all:
        return fits_out, params, kernel_areas, noise_ratios
    return fits_out
