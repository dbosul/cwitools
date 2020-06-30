"""Tools for extended data reduction."""

from cwitools import coordinates, modeling, utils, synthesis, extraction
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy import optimize
from scipy.ndimage.filters import convolve
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import correlate, medfilt
from scipy.stats import sigmaclip, mode
from skimage import measure, morphology
from shapely.geometry import box, Polygon
from tqdm import tqdm

import argparse
import astropy.coordinates
import astropy.stats
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import reproject
import sys
import time
import warnings

def slice_corr(fits_in, mask_reg=None):
    """Perform slice-by-slice median correction for scattered light.

    Args:
        fits_in (HDU or HDUList): The input data cube

    Returns:
        HDU or HDUList (same type as input): The corrected data

    """
    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data, hdu.header

    instrument = utils.get_instrument(header)
    if instrument == "PCWI":
        slice_axis = 1
    elif instrument == "KCWI":
        slice_axis = 2
    else:
        raise ValueError("Unrecognized instrument")

    slice_axis = np.nanargmin(data.shape)
    wav_axis = coordinates.get_wav_axis(header)
    nslices = data.shape[slice_axis]

    if mask_reg is not None:
        msk2d = extraction.reg2mask(fits_in, mask_reg)[0].data > 0
    else:
        msk2d = np.zeros_like(data[0], dtype=bool)

    #Run through slices
    for i in tqdm(range(nslices)):

        if slice_axis == 1:
            slice_2d = data[:, i, :]
            msk1d = msk2d[i, :]
        elif slice_axis == 2:
            slice_2d = data[:, :, i]
            msk1d = msk2d[:, i]
        else:
            raise RuntimeError("Shortest axis should be slice axis.")

        #Shrink mask if needed to obtain measurement
        while np.count_nonzero(msk1d == 0) < 5:
            msk1d = morphology.binary_erosion(msk1d)

        xdomain = np.arange(slice_2d.shape[1])
        bgmodel_z = np.zeros_like(slice_2d[:, 0])

        #Run through wavelength layers
        for wj in range(slice_2d.shape[0]):

            xprof = slice_2d[wj].copy()[msk1d == 0]

            clipped, lower, upper = sigmaclip(xprof, low=2, high=2)
            usex = (xprof >= lower) & (xprof <= upper)

            slice_2d[wj] -= np.median(xprof[usex])

    return fits_in


def estimate_variance(inputfits, window=50, nmin=30, snrmin=2.5, wmasks=[]):
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

    cube = inputfits[0].data.copy()
    varcube = np.zeros_like(cube)
    z, y, x = cube.shape
    Z = np.arange(z)
    wav_axis = coordinates.get_wav_axis(inputfits[0].header)
    cd3_3 = inputfits[0].header["CD3_3"]

    #Create wavelength masked based on input
    zmask = np.ones_like(wav_axis, dtype=bool)
    for (w0, w1) in wmasks:
        zmask[(wav_axis > w0) & (wav_axis < w1)] = 0
    nzmax = np.count_nonzero(zmask)

    #Loop over wavelength first to minimize repetition of wl-mask calculation
    for j, wav_j in enumerate(wav_axis):

        #Get initial width of white-light bandpass in px
        width_px = window / cd3_3

        #Create initial white-light mask, centered on j with above width
        vmask = zmask & (np.abs(Z - j) <= width_px / 2)

        #Grow until minimum number of valid wavelength layers included
        while np.count_nonzero(vmask) < min(nzmax, window / cd3_3):
            width_px += 2
            vmask = zmask & (np.abs(Z - j) <= width_px / 2)

        varcube[j] = np.var(cube[vmask], axis=0)

    #Adjust first estimate by rescaling, if set to do so
    varscale_f = scale_variance(cube, varcube, nmin=nmin, snrmin=snrmin)

    varcube *= varscale_f

    return varcube

def scale_variance(data, var, nmin=50, snrmin=3, plot=True):
    """TBD"""
    snr_range = (-5, 5)
    snr_bins = 100

    data = np.nan_to_num(data, nan=0)
    var = np.nan_to_num(var, nan=np.inf)

    scale_factor = 1
    scale_factor_change = 1.0
    std_fit = 99
    n = 0
    utils.output("\t%10s %15s %15s %15s\n" % ("iter", "scale_f", "std-dev", "1/std-dev"))
    while abs(std_fit - 1) >= 0.001:

        n += 1
        snr = data / np.sqrt(var)
        #Adjust SNR dist. using latest scale factor
        snr_scaled = snr * scale_factor

        #Segment into regions
        vox_msk = np.abs(snr_scaled) > snrmin
        vox_lab, num_reg = measure.label(vox_msk, return_num=True)

        #Measure sizes of regions above (in absolute terms) snr min
        reg_props = measure.regionprops_table(vox_lab, properties=['area', 'label'])
        large_regions = reg_props['area'] > nmin

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

        if 1:
            counts_all, edges_all = np.histogram(
                snr_scaled[~obj_mask],
                range=snr_range,
                bins=snr_bins
            )
            matplotlib.use('TkAgg')
            fig, ax  = plt.subplots(1, 1, figsize=(14,14))
            #ax.plot(centers, counts_all, 'k.--', alpha=0.5)
            #ax.plot(centers, counts, 'k.--')
            #ax.plot(centers[fit_cens], counts[fit_cens], 'kx')
            #ax.plot(centers, noisemodel1(centers), 'k-')
            #ax.plot(centers, noisemodel2(centers), 'r-')
            ax.hist(snr_scaled.flatten(),
                facecolor='k',
                range=snr_range,
                bins=snr_bins,
                alpha=0.4,
                label="All Data"
            )
            ax.hist(snr_scaled[~obj_mask].flatten(),
                facecolor='g',
                range=snr_range,
                bins=snr_bins,
                alpha=0.5,
                label="Systematics Removed"
            )
            ax.plot(centers, noisemodel0(centers), 'k--',
                alpha=0.5,
                linewidth=2.0,
                label="Standard Normal"
            )
            ax.plot(centers, noisemodel2(centers), 'k',
                linewidth=2.0,
                label="Best-fit Gaussian"
            )
            ax.set_yscale('log')
            ax.set_ylabel(r"$\mathrm{Log(N)}$", fontsize=32)
            ax.set_xlabel(r"$\mathrm{S/N}$", fontsize=32)
            ax.tick_params(labelsize=24)
            ax.legend(fontsize=20)
            fig.tight_layout()
            fig.show()
            input("")#plt.waitforbuttonpress()
            plt.close()

        new_scale_factor = 1 / std_fit
        utils.output("\t%10i %15.5f %15.5f %15.5f\n" % (n, scale_factor, std_fit, 1 / std_fit))

        scale_factor *= new_scale_factor


    return 1 / scale_factor**2

def xcor_crpix3(fits_list, xmargin=2, ymargin=2):
    """Get relative offsets in wavelength axis by cross-correlating sky spectra.

    Args:
        fits_list (Astropy.io.fits.HDUList list): List of sky cube FITS objects.
        xmargin (int): Margin to use along FITS axis 1 when summing spatially to
            create spectra. e.g. xmargin = 2 - exclude the edge 2 pixels left
            and right from contributing to the spectrum.
        ymargin (int): Margin to use along fits axis 2 when creating spevtrum.

    Returns:
        crpix3_corr (list): List of corrected CRPIX3 values.

    """
    #Extract wavelength axes and normalized sky spectra from each fits
    N = len(fits_list)
    wavs, spcs, crval3s, crpix3s = [], [], [], []
    for i, sky_fits in enumerate(fits_list):

        sky_data, sky_hdr = sky_fits[0].data, sky_fits[0].header
        sky_data = np.nan_to_num(sky_data, nan=0, posinf=0, neginf=0)

        wav = coordinates.get_wav_axis(sky_hdr)

        sky = np.sum(sky_data[:, ymargin:-ymargin, xmargin:-xmargin], axis=(1, 2))
        sky /= np.max(sky)

        spcs.append(sky)
        wavs.append(wav)
        crval3s.append(sky_hdr["CRVAL3"])
        crpix3s.append(sky_hdr["CRPIX3"])

    #Create common wavelength axis to interpolate sky spectra onto
    w0, w1 = np.min(wavs), np.max(wavs)
    dw_min = np.min([x[1] - x[0] for x in wavs])
    Nw = int((w1 - w0) / dw_min) + 1
    wav_common = np.linspace(w0, w1, Nw)

    #Interpolate (linearly) spectra onto common wavelength axis
    spc_interps = [interp1d(wavs[i], spcs[i])(wav_common) for i in range(N)]

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
    return crpix3s

def xcor_2d(hdu0_in, hdu1_in, crval=None, crpix=None, maxstep=None, box=None, upscale=1,
conv_filter=2., background_subtraction=False, background_level=None,
reset_center=False, method='interp-bicubic', output_flag=False, plot=0, debug=False):
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
        background_subtraction (bool): Set to True to apply background subtraction.
            If "background_level" is not specified, median is used.
        background_level (float tuple): Background value of the two images.
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
    
    plot=int(plot)

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
    elif crval is not None:
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

    img0 = np.nan_to_num(hdu0.data,nan=0,posinf=0,neginf=0)
    img1 = np.nan_to_num(img1,nan=0,posinf=0,neginf=0)

    sz0_sc = int(sz0[0] * upscale), int(sz0[1] * upscale) #Scaled size
    img1_expand = np.zeros((3 * sz0_sc[0], 3 * sz0_sc[1]))
    img1_expand[sz0_sc[0] : 2 * sz0_sc[0], sz0_sc[1] : 2 * sz0_sc[1]] = img1

    # +/- maxstep pix
    xcor_size = ((np.array(maxstep) - 1) * upscale + 1) + int(np.ceil(conv_filter))
    xcor_size = xcor_size.astype(int)
    xx = np.linspace(-xcor_size[0], xcor_size[0], 2 * xcor_size[0] + 1, dtype=int)
    yy = np.linspace(-xcor_size[1], xcor_size[1], 2 * xcor_size[1] + 1, dtype=int)
    dy, dx = np.meshgrid(yy, xx)

    xcor = np.zeros(dx.shape)
    box_sc = [b * upscale for b in box]
    box_sc = np.array(box_sc).astype(int)

    for ii in range(xcor.shape[0]):
        for jj in range(xcor.shape[1]):

            cut0 = img0[box_sc[1]:box_sc[3], box_sc[0]:box_sc[2]]
            cut1 = img1_expand[box_sc[1] - dy[ii,jj] + sz0_sc[0] : box_sc[3] - dy[ii,jj] + sz0_sc[0],
                               box_sc[0] - dx[ii,jj] + sz0_sc[1] : box_sc[2] - dx[ii,jj] + sz0_sc[1]]

            if background_subtraction:
                if background_level is None:
                    back_val0 = np.median(cut0[cut0 != 0])
                    back_val1 = np.median(cut1[cut1 != 0])
                else:
                    back_val0 = float(background_level[0])
                    back_val1 = float(background_level[1])

                cut0 = cut0 - back_val0
                cut1 = cut1 - back_val1
            else:
                if not background_level is None:
                    cut0[cut0 < background_level[0]] = 0
                    cut1[cut1 < background_level[1]] = 0

            cut0[cut0 < 0] = 0
            cut1[cut1 < 0] = 0
            mult = cut0 * cut1

            if np.sum(mult != 0) > 0:
                xcor[ii, jj] = np.sum(mult) / np.sum(mult!=0)

    # local maxima
    max_conv = ndimage.filters.maximum_filter(xcor, 2 * conv_filter + 1)
    maxima = (xcor == max_conv)
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    xindex, yindex = [],[]

    for dx, dy in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        xindex.append(x_center)
        y_center = (dy.start + dy.stop-1) / 2
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
        elif plot==2:
            fig, axes = plt.subplots(3, 2, figsize=(8, 12))
        else:
            raise ValueError('Allowed values for "plot": 0, 1, 2.')

        # xcor map
        if plot == 2:
            ax = axes[0, 0]
        elif plot == 1:
            ax = axes
        xplot = (np.append(xx, xx[1] - xx[0] + xx[-1]) - 0.5) / upscale
        yplot = (np.append(yy, yy[1] - yy[0] + yy[-1]) - 0.5) / upscale
        colormesh=ax.pcolormesh(xplot, yplot, xcor.T)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot([xplot.min(), xplot.max()], [0, 0], 'w--')
        ax.plot([0, 0], [yplot.min(), yplot.max()], 'w--')
        ax.set_xlabel('dx')
        ax.set_ylabel('dy')
        ax.set_title('XCOR_MAP')
        fig.colorbar(colormesh, ax=ax)

        if plot == 2:
            fig.delaxes(axes[0, 1])

            # adu0
            cut0_plot = img0[box_sc[1]:box_sc[3], box_sc[0]:box_sc[2]]
            ax = axes[1,0]
            imshow = ax.imshow(cut0_plot, origin='bottom')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Ref img')
            fig.colorbar(imshow, ax = ax)

            # adu1
            cut1_plot = img1_expand[box_sc[1] + sz0_sc[0] : box_sc[3] + sz0_sc[0],
                                    box_sc[0] + sz0_sc[1] : box_sc[2] + sz0_sc[1]]
            ax = axes[1, 1]
            imshow = ax.imshow(cut1_plot, origin='bottom')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Original img')
            fig.colorbar(imshow, ax=ax)


            # sub1
            ax = axes[2, 0]
            imshow = ax.imshow(cut1_plot - cut0_plot, origin = 'bottom')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Original sub')
            fig.colorbar(imshow, ax = ax)

    # closest local maxima
    if len(xindex) == 0:
        # Error handling
        if output_flag == True:
            return 0., 0., 0., 0., False
        else:
            # perhaps we can use the global maximum here, but it is also garbage...
            raise ValueError('Unable to find local maximum in the XCOR map.')

    max = np.max(max_conv[xindex, yindex])
    med = np.median(xcor)
    index = np.where(max_conv[xindex, yindex] > 0.3 * (max - med) + med)
    xindex = xindex[index]
    yindex = yindex[index]
    if len(xindex) == 0:
        # Error handling
        if output_flag == True:
            return 0., 0., 0., 0., False
        else:
            # perhaps we can use the global maximum here, but it is also garbage...
            raise ValueError('Unable to find local maximum in the XCOR map.')

    r = (xx[xindex]**2 + yy[yindex]**2)
    index = r.argmin()
    xshift = xx[xindex[index]] / upscale
    yshift = yy[yindex[index]] / upscale

    hdu1 = coordinates.scale_hdu(hdu1, 1 / upscale, header_only=True)
    hdu0 = coordinates.scale_hdu(hdu0, 1 / upscale, header_only=True)
    wcs0=WCS(hdu0.header)
    wcs1=WCS(hdu1.header)

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

    if plot!=0:
        if plot==1:
            ax = axes
        if plot==2:
            ax = axes[0, 0]
        ax.plot(xshift, yshift, '+', color='r', markersize=20)

        if plot==2:
            # sub2
            cut1_best = img1_expand[int((box[1] + sz0[0] - int(yshift)) * upscale) : int((box[3] + sz0[0] - int(yshift)) * upscale),
                                    int((box[0] + sz0[1] - int(xshift)) * upscale) : int((box[2] + sz0[1] - int(xshift)) * upscale)]
            ax=axes[2,1]
            imshow=ax.imshow(cut1_best-cut0_plot,origin='bottom')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Best sub')
            fig.colorbar(imshow,ax=ax)


        fig.tight_layout()
        plt.show()

    if output_flag == True:
        return crpix1_final, crpix2_final, crval1_final, crval2_final, True
    else:
        return crpix1_final, crpix2_final, crval1_final, crval2_final

def xcor_cr12(fits_in, fits_ref, wmask=[], maxstep=None,
ra=None, dec=None, box_size=None, crpix=None, 
pixscale=None, orientation=None, dimension=None, upscale=10., conv_filter=2.,
background_subtraction=False, background_level=None, reset_center=False,
method='interp-bicubic', plot=1):
    """Use cross-correlation to measure the values of CRPIX1/2 and CRVAL1/2.

    This function is a wrapper of xcor_2d() to optimize the reduction process.

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data to be shifted.
        fits_ref (astropy HDU / HDUList): Input HDU/HDUList with 3D data as reference.
        wmask (float tuple): Wavelength bins in which the cube is collapsed into a
            whitelight image.
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
        background_subtraction (bool): Apply background subtraction to the image?
            If "background_level" is not specified, it uses median as the background.
        background_level (float tuple): Background value of the two images. Pixels below
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
    hdu_img , _= synthesis.whitelight(hdu, wmask=wmask, mask_sky=True)
    hdu_img_ref , _ = synthesis.whitelight(hdu_ref, wmask=wmask, mask_sky=True)

    ### CHANGED - Use Astropy to get pixel sizes
    wcs = WCS(hdu_img.header)
    pixel_scales = proj_plane_pixel_scales(wcs)
    px = (pixel_scales[0] * u.deg).to(u.arcsec).value
    py = (pixel_scales[1] * u.deg).to(u.arcsec).value

    ### CHANGED - simplified a few lines to one here
    if pixscale is None:
        pixscale_x = pixscale_y = np.min([px, py])
    else:
        pixscale_x, pixscale_y = pixscale

    # Post projection image size
    if dimension is None:
        d_x = int(np.round(px * hdu_ref.shape[2] / pixscale_x))
        d_y = int(np.round(py * hdu_ref.shape[1] / pixscale_y))
        dimension = [d_x, d_y]

    # Construct WCS for the reference HDU in uniform grid
    hdrtmp = hdu_img_ref.header.copy()
    wcstmp = WCS(hdrtmp).copy()
    center = wcstmp.wcs_pix2world((wcstmp.pixel_shape[0] - 1) / 2.,
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
    old_cd12 = hdr0['CD1_2']
    old_cd21 = hdr0['CD2_1']
    old_cd22 = hdr0['CD2_2']
    hdr0['CD1_1'] = -pixscale_x / 3600
    hdr0['CD2_2'] = pixscale_y / 3600
    hdr0['CD1_2'] = 0.
    hdr0['CD2_1'] = 0.
    wcs0 = WCS(hdr0)

    # orientation
    if orientation==None:
        orientation = np.degrees(np.arctan(old_cd21 / (-old_cd11)))

    ### CHANGED - USE reduction.rotate() here to simplify code
    wcs_rot = rotate(wcs0, orientation)
    hdr_rot = wcs_rot.to_header()
    hdr0["CD1_1"]  = wcs_rot.wcs.cd[0,0]
    hdr0["CD1_2"]  = wcs_rot.wcs.cd[0,1]
    hdr0["CD2_1"]  = wcs_rot.wcs.cd[1,0]
    hdr0["CD2_2"]  = wcs_rot.wcs.cd[1,1]
    wcs0 = WCS(hdr0)
    
    ### CHANGED - Use new reproject_hdu wrapper
    hdu_img_ref0 = coordinates.reproject_hdu(hdu_img_ref, hdr0)
    
    # Reformat the box parameter
    if box_size is not None:
        box_x, box_y = wcs0.all_world2pix(ra, dec, 0)
        box = [box_x - int(box_size / pixscale_x / 2),
               box_y - int(box_size / pixscale_y /2), 
               box_x + int(box_size / pixscale_x /2), 
               box_y + int(box_size / pixscale_y /2)]
    else:
        box = None
    
    # CRs
    if crpix is not None:
        crval = [ra, dec]
    else: 
        crval = None

    # First iteration
    crpix1_tmp, crpix2_tmp, crval1_tmp, crval2_tmp, flag = xcor_2d(hdu_img_ref0, hdu_img,
        crval=crval,
        crpix=crpix,
        maxstep=maxstep,
        box=box,
        upscale=1,
        conv_filter=conv_filter,
        background_subtraction=background_subtraction,
        background_level=background_level,
        reset_center=reset_center,
        method=method,
        output_flag=True,
        plot=plot
    )
    if flag == False:
        if reset_center == False:
            utils.output('\tFirst attempt failed. Trying to recenter\n')
            crpix1_tmp, crpix2_tmp, crval1_tmp, crval2_tmp = xcor_2d(hdu_img_ref0,hdu_img,
                crval=crval,
                crpix=crpix,
                maxstep=maxstep,
                box=box,
                upscale=1,
                conv_filter=conv_filter,
                background_subtraction=background_subtraction,
                background_level=background_level,
                reset_center=True,
                method=method,
                output_flag=True,
                plot=plot
            )
        else:
            raise ValueError('Unable to find local maximum in the XCOR map.')

    utils.output('\tFirst iteration:\n')
    utils.output("\t\tCRPIX = %.2f, %.2f; CRVAL1 = %.4f, %.4f\n" % (crpix1_tmp, crpix2_tmp, crval1_tmp, crval2_tmp))

    # iteration 2: with upscale
    ### RECOMMENDED CHANGE
    # < What is the hard-coded value here, and can it be set to a variable? >
    ### END
    
    crpix1, crpix2, crval1, crval2 = xcor_2d(hdu_img_ref0, hdu_img,
        crval=[crval1_tmp, crval2_tmp], 
        crpix=[crpix1_tmp, crpix2_tmp], 
        maxstep=[2, 2],
        box=box,
        upscale=upscale,
        conv_filter=conv_filter,
        background_subtraction=background_subtraction,
        background_level=background_level,
        method=method,
        plot=plot,
        debug=True,
    )

    utils.output('\tSecond iteration:\n')
    utils.output("\t\tCRPIX = %.2f, %.2f; CRVAL1 = %.4f, %.4f\n" % (crpix1, crpix2, crval1, crval2))
    
    # get returning dataset
    #crpix1 = hdu_img.header['CRPIX1'] + dx2
    #crpix2 = hdu_img.header['CRPIX2'] + dy2
    #crval1 = hdu_img.header['CRVAL1']
    #crval2 = hdu_img.header['CRVAL2']

    return crpix1,crpix2,crval1,crval2

def fit_crpix12(fits_in, crval1, crval2, box_size=10, plot=False, iters=3, std_max=4):
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
    x0 = max(0, int(crpix1 - box_size_x / 2))
    x1 = min(cube.shape[2] - 1, int(crpix1 + box_size_x / 2 + 1))

    y0 = max(0, int(crpix2 - box_size_y / 2))
    y1 = min(cube.shape[1] - 1, int(crpix2 + box_size_y / 2 + 1))

    #Create data structures for fitting
    x_domain = np.arange(x0, x1)
    y_domain = np.arange(y0, y1)

    x_prof = np.sum(wl_img[y0:y1, x0:x1], axis=0)
    y_prof = np.sum(wl_img[y0:y1, x0:x1], axis=1)

    x_prof /= np.max(x_prof)
    y_prof /= np.max(y_prof)

    #Determine bounds for gaussian profile fit
    x_bounds = [
        (0, 10),
        (x0, x1),
        (0, std_max / x_scale)
    ]
    y_bounds = [
        (0, 10),
        (y0, y1),
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
        TL, TR = axes[0, :]
        BL, BR = axes[1, :]
        TL.set_title("Full Image", fontsize=24)
        TL.pcolor(wl_img, vmin=0, vmax=wl_img.max())
        TL.plot( [x0, x0], [y0, y1], 'w-')
        TL.plot( [x0, x1], [y1, y1], 'w-')
        TL.plot( [x1, x1], [y1, y0], 'w-')
        TL.plot( [x1, x0], [y0, y0], 'w-')
        TL.plot( crpix1,  crpix2, 'wx', markersize=15, markeredgewidth=4.0)
        TL.plot( x_center + 0.5, y_center + 0.5, 'rx', markersize=15, markeredgewidth=4.0)
        TL.set_aspect(y_scale/x_scale)

        TR.set_title("%.1f x %.1f Arcsec Box" % (box_size, box_size), fontsize=24)
        TR.pcolor(wl_img[y0:y1, x0:x1], vmin=0, vmax=wl_img.max())
        TR.plot( crpix1 + 0.5 - x0, crpix2 + 0.5 - y0, 'wx', markersize=15, markeredgewidth=4.0)
        TR.plot( x_center + 0.5 - x0, y_center + 0.5 - y0, 'rx', markersize=15, markeredgewidth=4.0)
        TR.set_aspect(y_scale/x_scale)

        BL.set_title("X Profile Fit", fontsize=24)
        BL.plot(x_domain, x_prof, 'k.-', linewidth=2, label="Data")
        BL.plot(x_domain, x_prof_model, 'r--', linewidth=2, label="Model")
        BL.plot( [x_center]*2, [0,1], 'r--')
        BL.legend(fontsize=18)

        BR.set_title("Y Profile Fit", fontsize=24)
        BR.plot(y_domain, y_prof, 'k.-', linewidth=2, label="Data")
        BR.plot(y_domain, y_prof_model, 'r--', linewidth=2, label="Model")
        BR.plot( [y_center]*2, [0,1], 'r--')
        BR.legend(fontsize=18)

        for ax in fig.axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.show()
        plt.waitforbuttonpress()
        plt.close()

    #Return
    return x_center + 1, y_center + 1

def rebin(inputfits, xybin=1, zbin=1, vardata=False):
    """Re-bin a data cube along the spatial (x,y) and wavelength (z) axes.

    Args:
        inputfits (astropy FITS object): Input FITS to be rebinned.
        xybin (int): Integer binning factor for x,y axes. (Def: 1)
        zbin (int): Integer binning factor for z axis. (Def: 1)
        vardata (bool): Set to TRUE if rebinning variance data. (Def: True)
        fileExt (str): File extension for output (Def: .binned.fits)

    Returns:
        astropy.io.fits.HDUList: The re-binned cube with updated WCS/Header.

    """


    #Extract useful structures
    data = inputfits[0].data.copy()
    head = inputfits[0].header.copy()

    #Get dimensions & Wav array
    z, y, x = data.shape
    wav = coordinates.get_wav_axis(head)

    #Get new sizes
    znew = int(z // zbin)
    ynew = int(y // xybin)
    xnew = int(x // xybin)

    #Perform wavelenght-binning first, if bin provided
    if zbin > 1:

        #Get new bin size in Angstrom
        zbinSize = zbin * head["CD3_3"]

        #Create new data cube shape
        data_zbinned = np.zeros((znew, y, x))

        #Run through all input wavelength layers and add to new cube
        for zi in range(znew * zbin):
            data_zbinned[int(zi // zbin)] += data[zi]

        #Normalize so that units remain as "erg/s/cm2/A"
        if vardata: data_zbinned /= zbin**2
        else: data_zbinned /= zbin

        #Update central reference and pixel scales
        head["CD3_3"] *= zbin
        head["CRPIX3"] /= zbin

    else:

        data_zbinned = data

    #Perform spatial binning next
    if xybin > 1:

        #Get new shape
        data_xybinned = np.zeros((znew, ynew, xnew))

        #Run through spatial pixels and add
        for yi in range(ynew * xybin):
            for xi in range(xnew * xybin):
                xindex = int(xi // xybin)
                yindex = int(yi // xybin)
                data_xybinned[:, yindex, xindex] += data_zbinned[:, yi, xi]

        #
        # No normalization needed for binning spatial pixels.
        # Units remain as 'per pixel' but pixel size changes.
        #

        #Update reference pixel
        head["CRPIX1"] /= float(xybin)
        head["CRPIX2"] /= float(xybin)

        #Update pixel scales
        for key in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]: head[key] *= xybin

    else: data_xybinned = data_zbinned

    binnedFits = fits.HDUList([fits.PrimaryHDU(data_xybinned)])
    binnedFits[0].header = head

    return binnedFits

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


    if type(pad) == int:
        pad = (pad, pad)

    hdu = utils.extractHDU(fits_in)
    data = hdu.data.copy()
    header = hdu.header.copy()

    # instrument
    inst = utils.get_instrument(header)

    hdu_2d , _ = synthesis.whitelight(hdu, mask_sky=True)

    #Extract data so that axes are [wav, in-slice, across-slice] = [w, y, x]
    if inst == 'KCWI':
        wl = hdu_2d.data
    elif inst == 'PCWI':
        wl = hdu_2d.data.T
        pad[0], pad[1] = pad[1], pad[0]
    else:
        raise ValueError('Instrument not recognized.')

    nslices = wl.shape[1] #Number of slice pixels
    npix = wl.shape[0] #Number of in-slice pixels

    # zpad
    wav_axis = coordinates.get_wav_axis(header)

    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    xprof = np.max(data, axis=(0, 1))
    yprof = np.max(data, axis=(0, 2))
    zprof = np.max(data, axis=(1, 2))

    w0, w1 = header["WAVGOOD0"], header["WAVGOOD1"]
    z0, z1 = coordinates.get_indices(w0, w1, header)
    wcrop = w0, w1 = [wav_axis[z0], wav_axis[z1]]

    # xpad
    xbad = xprof <= 0
    ybad = yprof <= 0

    x0 = int(np.round(xbad.tolist().index(False) + pad[0]))
    x1 = int(np.round(len(xbad) - xbad[::-1].tolist().index(False) - 1 - pad[0]))

    xcrop = [x0, x1]

    if zero_only:

        y0 = ybad.tolist().index(False) + pad[1]
        y1 = len(ybad) - ybad[::-1].tolist().index(False) - 1 - pad[1]

    else:

        bot_pads = np.repeat(np.nan, nslices)
        top_pads = np.repeat(np.nan, nslices)

        for i in range(nslices):

            stripe = wl[:, i]
            stripe_clean = stripe[stripe != 0]

            if len(stripe_clean) == 0:
                continue

            stripe_clean_masked, lo, hi = sigma_clip(stripe_clean,
                sigma = nsig,
                return_bounds = True
            )

            med = np.median(stripe_clean_masked.data[~stripe_clean_masked.mask])
            thresh = ((med - lo) + (hi - med)) / 2
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
        y0 = np.nanmedian(bot_pads) + pad[1]
        y1 = np.nanmedian(top_pads) - pad[1]

    #Round up to nearest index
    y0 = int(np.round(y0))
    y1 = int(np.round(y1))
    ycrop = [y0, y1]

    if inst=='PCWI':
        x0, x1, y0, y1 = y0, y1, x0, x1
        xcrop, ycrop = ycrop, xcrop

    utils.output("\tAutoCrop Parameters:\n")
    utils.output("\t\tx-crop: %02i:%02i\n" % (x0, x1))
    utils.output("\t\ty-crop: %02i:%02i\n" % (y0, y1))
    utils.output("\t\tz-crop: %i:%i (%i:%i A)\n" % (z0, z1, w0, w1))

    if plot:

        x0, x1 = xcrop
        y0, y1 = ycrop

        xprof_clean = np.max(data[z0:z1, y0:y1, :], axis=(0, 1))
        yprof_clean = np.max(data[z0:z1, :, x0:x1], axis=(0, 2))
        zprof_clean = np.max(data[:, y0:y1, x0:x1], axis=(1, 2))

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        xax, yax, wax = axes
        xax.step(xprof_clean, 'k-', linewidth=2)
        xax.step(range(x0, x1), xprof_clean[x0:x1], 'b-', linewidth=2)

        lim=xax.get_ylim()
        xax.set_xlabel("X (Axis 2)", fontsize=18)
        xax.plot([x0, x0], [xprof.min(), xprof.max()], 'r-' )
        xax.plot([x1-1, x1-1], [xprof.min(), xprof.max()], 'r-' )
        xax.set_ylim(lim)

        yax.step(yprof_clean, 'k-', linewidth=2)
        yax.step(range(y0, y1), yprof_clean[y0:y1], 'b-', linewidth=2)
        lim=yax.get_ylim()
        yax.set_xlabel("Y (Axis 1)", fontsize=18)
        yax.plot([y0, y0], [yprof.min(), yprof.max()], 'r-' )
        yax.plot([y1 - 1, y1 - 1], [yprof.min(), yprof.max()], 'r-' )
        yax.set_ylim(lim)

        wax.step(zprof_clean, 'k-', linewidth=2)
        wax.step(range(z0, z1), zprof_clean[z0:z1], 'b-', linewidth=2)
        lim=wax.get_ylim()
        wax.plot([z0, z0], [zprof.min(), zprof.max()], 'r-' )
        wax.plot([z1 - 1, z1 - 1], [zprof.min(), zprof.max()], 'r-' )
        wax.set_xlabel("Z (Axis 0)", fontsize=18)
        wax.set_ylim(lim)

        for ax in fig.axes:
            ax.set_yticks([])
            ax.tick_params(labelsize=16)
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
        data cube to the wavelength range 4200-4400A ,the usage would be:

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
    header = hdu.header.copy()

    wav_axis = coordinates.get_wav_axis(header)

    #Get profiles of each axis
    data[np.isnan(data)] = 0
    xprof = np.max(data, axis=(0, 1))
    yprof = np.max(data, axis=(0, 2))
    zprof = np.max(data, axis=(1, 2))

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
    if xcrop == None:
        xcrop = [0, data.shape[2]]

    if ycrop == None:
        ycrop = [0, data.shape[1]]

    if wcrop == None:
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
    mrot = np.array([[cosq, -sinq],
                     [sinq, cosq]])

    
    # reset center
    if keep_center:
        crpix_old = wcs.wcs.crpix
        crval_old = wcs.wcs.crval
        naxis = np.flip(wcs.array_shape)
        crpix = (np.array(naxis)+1)/2.
        crval = wcs.all_pix2world(crpix[0],crpix[1],1)
        crval = [i for i in crval]
        wcs.wcs.crpix=crpix
        wcs.wcs.crval=crval

    if wcs.wcs.has_cd():    # CD matrix
        newcd = np.dot(mrot, wcs.wcs.cd)
        wcs.wcs.cd = newcd
        wcs.wcs.set()
        return wcs
    elif wcs.wcs.has_pc():      # PC matrix + CDELT
        newpc = np.dot(mrot, wcs.wcs.get_pc())
        wcs.wcs.pc = newpc
        wcs.wcs.set()
        return wcs
    else:
        raise TypeError("Unsupported wcs type (need CD or PC matrix)")


def coadd(cube_list, cube_type=None,  masks_in=None, var_in=None, pa=None,
px_thresh=0.5, exp_thresh=0.1, verbose=False, plot=0, drizzle=0):
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
    if type(cube_list) is str and ".list" in cube_list:

        if cube_type is None:
            raise SyntaxError("cube_type must also be provided if coadding with a CWITools .list file")

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
        if type(masks_in) is str:
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
        if type(var_in) is str:
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
    elif type(cube_list) is list:
        int_hdus = [utils.extractHDU(x) for x in cube_list]

        if type(masks_in) is list:
            mask_hdus = [utils.extractHDU(x) for x in masks_in]
        else:
            mask_hdus = None

        if type(var_in) is list:
            var_hdus = [utils.extractHDU(x) for x in var_in]
        else:
            var_hdus = None

    else:
        raise SyntaxError("Something is wrong with the given input types for cube_list and/or cube_type. Check and try again.")

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
        header3D = int_hdu.header

        #2D Header & World Coordinate System
        header2D = coordinates.get_header2d(header3D)
        wcs2D = WCS(header2D)

        #On-sky footprint
        footprint = wcs2D.calc_footprint()

        #Wavelength limits
        wav0 = header3D["CRVAL3"] - (header3D["CRPIX3"] - 1) * header3D["CD3_3"]
        wav1 = wav0 + header3D["NAXIS3"] * header3D["CD3_3"]

        #Position Angle
        if "ROTPA" in header3D:
            pa_i = header3D["ROTPA"]
        elif "ROTPOSN" in header3D:
            pa_i = header3D["ROTPOSN"]
        else:
            warnings.warn("No header key for PA (ROTPA or ROTPOSN) found.")
            pa_i = 0

        # Replace masked voxels with NaN values
        if usemask:
            msk_data = mask_hdus[i].data
            bin_mask = (msk_data == 1) & (msk_data >= 8)
            int_hdu.data[bin_mask] = np.nan

        #Add all of the above to lists
        wscales.append(header3D["CD3_3"])
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
            mode = 'constant'
        )

        # 2 - Perform integer shift with np.roll
        int_hdu.data = np.roll(int_hdu.data, int_shift, axis=0)

        # 3 - Shift data along axis by convolving with K
        int_hdu.data = np.apply_along_axis(
            lambda m: np.convolve(m, shift_kernel, mode = 'same'),
            axis = 0,
            arr = int_hdu.data
        )

        # 4 - Update header's WCS info for axis 3
        int_hdu.header["NAXIS3"] = len(wav_new)
        int_hdu.header["CRVAL3"] = wav_new[0]
        int_hdu.header["CRPIX3"] = 1

        # Apply steps 1-4 to variance (square the kernel for convolution)
        if usevar:
            var_hdus[i].data  = np.pad(
                var_hdus[i].data ,
                ((0, n_pad_w), (0,0), (0,0)),
                mode='constant'
            )
            # 2 - Perform integer shift with np.roll
            var_hdus[i].data = np.roll(var_hdus[i].data, int_shift, axis=0)

            # 3 - Shift data along axis by convolving with K
            var_hdus[i].data = np.apply_along_axis(
                lambda m: np.convolve(m, shift_kernel**2, mode = 'same'),
                axis = 0,
                arr = var_hdus[i].data
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
    dx0, dy0 = proj_plane_pixel_scales(wcs0)
    if dx0 > dy0:
        wcs0.wcs.cd[:,0] /= dx0 / dy0
    else:
        wcs0.wcs.cd[:,1] /= dy0 / dx0

    #Rotate WCS to the input pa
    wcs0 = rotate(wcs0, pas[i] - pa)

    #Set new WCS - we will use it later to create the canvas
    wcs0.wcs.set()

    # We don't know which corner is which for an arbitrary rotation
    # So, map each vertex to the coadd space
    x0, y0 = 0, 0
    x1, y1 = 0, 0
    for fp in footprints:
        ras, decs = fp[:,0],fp[:,1]
        xs, ys = wcs0.all_world2pix(ras, decs, 0)
        x0 = min(np.min(xs), x0)
        y0 = min(np.min(ys), y0)
        x1 = max(np.max(xs), x1)
        y1 = max(np.max(ys), y1)

    #Get required size of the canvas in x, y
    coadd_size_x = int(round((x1 - x0) + 1))
    coadd_size_y = int(round((y1 - y0) + 1))

    #Get RA/DEC of lower-left corner - to establish WCS reference point
    ra0, dec0 = wcs0.all_pix2world(x0, y0, 0)
    px_scale_new = proj_plane_pixel_scales(wcs0)[0]

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
    coadd_hdr["CD1_1"]  = wcs0.wcs.cd[0,0]
    coadd_hdr["CD1_2"]  = wcs0.wcs.cd[0,1]
    coadd_hdr["CD2_1"]  = wcs0.wcs.cd[1,0]
    coadd_hdr["CD2_2"]  = wcs0.wcs.cd[1,1]

    # Re-generate a WCS object from this coadd header and get on-sky footprint
    coadd_hdr2D = coordinates.get_header2d(coadd_hdr)
    coadd_wcs = WCS(coadd_hdr2D)
    coadd_fp = coadd_wcs.calc_footprint()

    #Get scales and pixel size of new canvas
    coadd_px_area = coordinates.get_pxarea_arcsec(coadd_hdr2D)

    # Create data structures to store coadded cube and corresponding exposure time mask
    coadd_data = np.zeros((coadd_size_w, coadd_size_y, coadd_size_x))
    coadd_exp = np.zeros_like(coadd_data)

    if usevar:
        coadd_var = np.zeros_like(coadd_data)

    W, Y, X = coadd_data.shape

    if plot:

        fig1, ax = plt.subplots(1,1)
        for fp in footprints:
            ax.plot( -fp[0:2, 0], fp[0:2, 1],'k-')
            ax.plot( -fp[1:3, 0], fp[1:3, 1],'k-')
            ax.plot( -fp[2:4, 0], fp[2:4, 1],'k-')
            ax.plot([-fp[3, 0], -fp[0, 0]], [fp[3, 1], fp[0, 1]], 'k-')
        for fp in [coadd_fp]:
            ax.plot( -fp[0:2, 0], fp[0:2, 1],'r-')
            ax.plot( -fp[1:3, 0], fp[1:3, 1],'r-')
            ax.plot( -fp[2:4, 0], fp[2:4, 1],'r-')
            ax.plot([-fp[3, 0], -fp[0, 0]], [fp[3, 1], fp[0, 1]], 'r-')

        fig1.show()
        plt.waitforbuttonpress()
        plt.close()
        plt.ion()

        gs = gridspec.GridSpec(2, 2)
        fig2 = plt.figure(figsize=(12,12))
        input_ax  = fig2.add_subplot(gs[ :1, : ])
        sky_ax = fig2.add_subplot(gs[ 1:, :1 ])
        coadd_ax = fig2.add_subplot(gs[ 1:, 1: ])

    if verbose:
        pbar = tqdm(total=np.sum([x.data[0].size for x in int_hdus]))

    # Run through each input frame
    for i, int_hdu in enumerate(int_hdus):

        header_i = int_hdu.header
        header2D_i = coordinates.get_header2d(header_i)
        wcs2D_i = WCS(header2D_i)
        px_area_i = coordinates.get_pxarea_arcsec(header_i)

        if "TELAPSE" in header_i:
            t_exp_i = header_i["TELAPSE"]
        elif "EXPTIME" in header_i:
            t_exp_i = header_i["TELAPSE"]
        else:
            warnings.warn("No exposure time (TELAPSE or EXPTIME) keyword found in header. Skipping file.")
            continue

        #Get shape of current cube
        w, y, x = int_hdu.data.shape

        # Create intermediate frame to build up coadd contributions pixel-by-pixel
        build_frame = np.zeros_like(coadd_data)

        #Build frame for variance
        if usevar:
            var_build_frame = np.zeros_like(coadd_data)

        # Fract frame stores a coverage fraction for each coadd pixel
        fract_frame = np.zeros_like(coadd_data)

        # Get wavelength coverage of this FITS as binary mask
        wavmask_i = np.ones(len(wav_new), dtype = bool)
        wavmask_i[wav_new < wav0s[i]] = 0
        wavmask_i[wav_new > wav1s[i]] = 0

        # Convert to a flux-like unit if the input data is in counts
        if "electrons" in int_hdu.header["BUNIT"]:

            int_hdu.data /= t_exp_i

            #Propagate error
            if usevar:
                var_hdus[i].data /= t_exp_i**2

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
            xU,yU = x,y
            input_ax.plot( [0,xU], [0,0], 'k-')
            input_ax.plot( [xU,xU], [0,yU], 'k-')
            input_ax.plot( [xU,0], [yU,yU], 'k-')
            input_ax.plot( [0,0], [yU,0], 'k-')
            input_ax.set_xlim( [-5,xU+5] )
            input_ax.set_ylim( [-5,yU+5] )
            #input_ax.plot(qXin,qYin,'ro')
            input_ax.set_xlabel("X")
            input_ax.set_ylabel("Y")
            xU,yU = X,Y
            coadd_ax.plot( [0,xU], [0,0], 'r-')
            coadd_ax.plot( [xU,xU], [0,yU], 'r-')
            coadd_ax.plot( [xU,0], [yU,yU], 'r-')
            coadd_ax.plot( [0,0], [yU,0], 'r-')
            coadd_ax.set_xlim( [-0.5,xU+1] )
            coadd_ax.set_ylim( [-0.5,yU+1] )
            for fp in footprints[i:i+1]:
                sky_ax.plot( -fp[0:2,0],fp[0:2,1],'k-')
                sky_ax.plot( -fp[1:3,0],fp[1:3,1],'k-')
                sky_ax.plot( -fp[2:4,0],fp[2:4,1],'k-')
                sky_ax.plot( [ -fp[3,0], -fp[0,0] ] , [ fp[3,1], fp[0,1] ],'k-')
            for fp in [coadd_fp]:
                sky_ax.plot( -fp[0:2,0],fp[0:2,1],'r-')
                sky_ax.plot( -fp[1:3,0],fp[1:3,1],'r-')
                sky_ax.plot( -fp[2:4,0],fp[2:4,1],'r-')
                sky_ax.plot( [ -fp[3,0], -fp[0,0] ] , [ fp[3,1], fp[0,1] ],'r-')

        # Loop through spatial pixels in this input frame
        for yj in range(y):
            for xk in range(x):

                #Get binary mask of good wavelength indices at this x,y position
                msk_jk = wavmask_i & ~np.isnan(int_hdu.data[:, yj, xk])

                # Define BL, TL, TR, BR corners of pixel as coordinates
                pix_verts =  np.array([
                    [xk - 0.5 + drz_f, yj - 0.5 + drz_f],
                    [xk - 0.5 + drz_f, yj + 0.5 - drz_f],
                    [xk + 0.5 - drz_f, yj + 0.5 - drz_f],
                    [xk + 0.5 - drz_f, yj - 0.5 + drz_f]
                ])

                # Convert these vertices to RA/DEC positions
                pix_verts_radec = wcs2D_i.all_pix2world(pix_verts, 0)

                # Convert the RA/DEC vertex values into coadd frame coordinates
                pix_verts_coadd = coadd_wcs.all_world2pix(pix_verts_radec, 0)

                #Create polygon object for projection onto coadd grid
                pix_projection = Polygon(pix_verts_coadd)

                if plot:
                    input_ax.plot(pix_verts[:,0], pix_verts[:,1], 'kx')
                    sky_ax.plot(-pix_verts_radec[:,0], pix_verts_radec[:,1], 'kx')
                    coadd_ax.plot(pix_verts_coadd[:,0], pix_verts_coadd[:,1], 'kx')

                #Get bounds of pixel projection in coadd frame
                proj_bounds = list(pix_projection.exterior.bounds)

                # xb0 is x-bound-lower, yb1 is y-bound-upper, etc.
                xb0, yb0, xb1, yb1 = (int(round(pib)) for pib in proj_bounds)

                # Upper bounds need to be increased to include full pixel
                xb1 += 1
                yb1 += 1

                # Loop through relevant pixels in output/coadd frame
                for xc in range(xb0, xb1):
                    for yc in range(yb0, yb1):

                        #Get corners of pixel
                        xc0 = xc - 0.5
                        xc1 = xc + 0.5
                        yc0 = yc - 0.5
                        yc1 = yc + 0.5

                        # Define BL, TL, TR, BR corners of pixel as coordinates
                        out_px_verts =  np.array([
                            [xc0, yc0],
                            [xc0, yc1],
                            [xc1, yc1],
                            [xc1, yc0]
                        ])

                        # Create Polygon object for this coadd pixel
                        coadd_pixel = box(xc0, yc0, xc1, yc1)

                        # Get overlap between input pixel and this coadd pixel
                        overlap = pix_projection.intersection(coadd_pixel).area

                        # Convert to fraction of total input pixel area
                        overlap /= pix_projection.area

                        # Extract spectrum
                        spc_in = int_hdu.data[msk_jk, yj, xk]

                        # Extract variance
                        if usevar:
                            var_jk = var_hdus[i].data[msk_jk, yj, xk].copy()
                            var_jk *= (overlap**2)

                        #Update all relevant frames
                        try:
                            fract_frame[msk_jk, yc, xc] += overlap
                            build_frame[msk_jk, yc, xc] += overlap * spc_in
                            if usevar:
                                var_build_frame[msk_jk, yc, xc] += var_jk

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

        flat_frame[flat_frame < px_thresh] = np.inf

        # Create 3D mask of non-zero voxels from this frame
        M = flat_frame < np.inf

        # Add weight * data to coadd
        coadd_data += t_exp_i * build_frame

        # Propagate error on the above step
        if usevar:
            coadd_var += (t_exp_i**2) * var_build_frame

        #Add to exposure mask
        coadd_exp += t_exp_i * M
        coadd_exp2D = np.sum(coadd_exp, axis=0)

    if verbose:
        pbar.close()

    utils.output("\tTrimming coadded canvas.\n")

    if plot:
        plt.close()

    # Create 1D exposure time profiles
    exp_zprof = np.mean(coadd_exp, axis=(1, 2))
    exp_xprof = np.mean(coadd_exp, axis=(0, 1))
    exp_yprof = np.mean(coadd_exp, axis=(0, 2))

    # Normalize the profiles
    exp_zprof /= np.max(exp_zprof)
    exp_xprof /= np.max(exp_xprof)
    exp_yprof /= np.max(exp_yprof)

    # Convert 0s to 1s in exposure time cube
    ee = coadd_exp.flatten()
    ee[ee == 0] = 1
    coadd_exp = np.reshape(ee, coadd_data.shape)

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

    #Get 'bottom/left/blue corner of cropped data
    W0 = np.argmax(use_z)
    X0 = np.argmax(use_x)
    Y0 = np.argmax(use_y)

    #Update the WCS to account for trimmed pixels
    coadd_hdr["CRPIX3"] -= W0
    coadd_hdr["CRPIX2"] -= Y0
    coadd_hdr["CRPIX1"] -= X0

    # Create FITS object matching the input type (i.e. HDU or HDUList)
    coadd_fits = utils.matchHDUType(int_hdus[0], coadd_data, coadd_hdr)

    if usevar:
        coadd_var_fits = utils.matchHDUType(var_hdus[0], coadd_var, coadd_hdr)
        return coadd_fits, coadd_var_fits

    else:
        return coadd_fits

def air2vac(fits_in, mask = False):
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
    cube = np.nan_to_num(hdu.data, nan = 0, posinf = 0, neginf = 0)
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

            if mask == False:
                f_cubic = interp1d(wave_vac, spec0,
                    kind = 'cubic',
                    fill_value = 'extrapolate'
                )
                spec_new = f_cubic(wave_air)
            else:
                f_pre = interp1d(wave_vac, spec0,
                    kind = 'previous',
                    bounds_error = False,
                    fill_value = 128
                )
                spec_pre = f_pre(wave_air)

                f_nex = interp1d(wave_vac, spec0,
                    kind = 'next',
                    bounds_error = False,
                    fill_value = 128
                )
                spec_nex = f_nex(wave_air)

                spec_new = np.zeros_like(spec0)
                for k in range(spec0.shape[0]):
                    spec_new[k] = max(spec_pre[k], spec_nex[k])

            cube_new[:, j, i] = spec_new

    hdr['CTYPE3'] = 'WAVE'
    hdu_new = utils.matchHDUType(fits_in, cube_new, hdr)

    return hdu_new


def heliocentric(fits_in, mask=False, return_vcorr=False, resample=True,
vcorr=None, barycentric=False):
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
    cube = np.nan_to_num(hdu.data,  nan = 0,  posinf = 0,  neginf = 0)
    hdr = hdu.header

    v_old = 0.
    if 'VCORR' in hdr:
        v_old = hdr['VCORR']
        utils.output("\tRolling back the existing correction with:\n")
        utils.output("\t\tVcorr = %.2f km/s.\n" % (v_old))

    if vcorr is None:
        targ = astropy.coordinates.SkyCoord(hdr['TARGRA'],  hdr['TARGDEC'],
            unit = 'deg',
            obstime = hdr['DATE-BEG']
        )
        keck = astropy.coordinates.EarthLocation.of_site('Keck Observatory')
        if barycentric:
            vcorr = targ.radial_velocity_correction(kind = 'barycentric',
                location = keck
            )
        else:
            vcorr = targ.radial_velocity_correction(kind = 'heliocentric',
                location = keck
            )
        vcorr = vcorr.to('km/s').value

    utils.output("\tHelio/Barycentric correction:\n")
    utils.output("\t\tVcorr = %.2f km/s.\n" % (vcorr))

    v_tot = vcorr-v_old

    if resample == False:

        hdr['CRVAL3'] = hdr['CRVAL3'] * (1 + v_tot / 2.99792458e5)
        hdr['CD3_3'] = hdr['CD3_3'] * (1 + v_tot / 2.99792458e5)
        hdr['VCORR'] = vcorr
        hdu_new = utils.matchHDUType(fits_in, cube, hdr)
        if not return_vcorr:
            return hdu_new
        else:
            return hdu_new, vcorr

    else:

        wave_old = coordinates.get_wav_axis(hdr)
        wave_hel = wave_old * (1 + v_tot / 2.99792458e5)

        # resample to uniform grid
        cube_new = np.zeros_like(cube)
        for i in range(cube.shape[2]):
            for j in range(cube.shape[1]):

                spec0 = cube[:,  j,  i]
                if mask == False:
                    f_cubic = interp1d(wave_hel,  spec0,
                        kind = 'cubic',
                        fill_value = 'extrapolate'
                    )
                    spec_new = f_cubic(wave_old)

                else:
                    f_pre = interp1d(wave_hel,  spec0,
                        kind = 'previous',
                        bounds_error = False,
                        fill_value = 128
                    )
                    spec_pre = f_pre(wave_old)
                    f_nex = interp1d(wave_hel,  spec0,
                        kind = 'next',
                        bounds_error = False,
                        fill_value = 128
                    )
                    spec_nex = f_nex(wave_old)

                    spec_new = np.zeros_like(spec0)
                    for k in range(spec0.shape[0]):
                        spec_new[k] = max(spec_pre[k],  spec_nex[k])

                cube_new[:,  j,  i] = spec_new

        hdr['VCORR'] = vcorr
        hdu_new = utils.matchHDUType(fits_in, cube_new, hdr)

        if not return_vcorr:
            return hdu_new

        else:
            return hdu_new,  vcorr

def cov_curve(npix, alpha, norm=1):
    """Calculate the theoretical covariance rescaling curve from modeled parameters 
        (see O'Sullivan+20).

    Args:
        npix (np.array): Number of pixels that are binned spatially. 
        alpha (float): Parameter that quantifies how much pixels are correlated. 
        norm (float): Normalizing paramter.

    Returns:
        factor (np.array): The theoretical rescaling factor for each npix.

    """
    return (1+alpha*np.log(npix))*norm

def update_cov_header(fits_in, alpha, norm=1):
    """Update FITS header to the given parameters of the covariance curve. 
    
    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        alpha (float): Parameter that quantifies how much pixels are correlated.
        norm (float): Normalizing paramter.
        
    Returns:
        HDU / HDUList*: Modified HDU/HDUList
    
    """
    hdu = utils.extractHDU(fits_in)
    data = hdu.data.copy()
    hdr = hdu.header.copy()
    hdr["COV_A"] = alpha
    hdr["COV_B"] = norm
    fits_out = utils.matchHDUType(fits_in, data, hdr)
    return fits_out

def get_cov(fits_in, var, mask=None, wrange=[], xbins=None, nw=100, wavegood=True, 
           niter=5, nsig=3., return_all=False):
    
    """Extract the covariance rescaling curve from the observed data and 
        variance cubes.

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        var (np.array): Variance cube.
        mask (np.array): Mask cube. 
        wrange (tuple): Lower and higher range in wavelength that the curve 
            is extracted from.
        xbins (np.array): List of pixel bin sizes. Default is a list of 10 poiints
            evenly distributed between 1 and 1/5 of the shortest spatial axis.
        nw (int): Number of independent estimates in each bin size. This is
            done by grouping the independent wavelength layers. 
        wavegood (bool): Shortcut to use the good wavelength range from header.
        niter (int): Number of iterations to fit the rescaling curve. 
        nsig (float): Number of sigma for sigma-rejection. 
        return_all (bool): If set, also return the independently measured data 
            points.

    Returns:
        HDU / HDUList*: Curve parameters recorded in the FITS header.
        param (np.array): Parameters (alpha, norm) that can be used to recover
            the rescaling curve. Only return if return_all == True.
        bin_all (np.array): Bin sizes for the independently measured data 
            points. Only return if return_all == True.
        fac_all (np.array): Rescaling factor for the independely measured data
            points. Only return if return_all ==True.

    """
    
    # initial readings
    hdu = utils.extractHDU(fits_in)
    data = hdu.data.copy()
    hdr = hdu.header.copy()
    
    var = var.copy()
    
    if mask is not None:
        var[mask != 0] = 0
        data[mask != 0] = 0
        
    
    #Filter data for bad values
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    var = np.nan_to_num(var, nan=0, posinf=0, neginf=0)

    #Get wavelength axis for masking
    wav_axis = coordinates.get_wav_axis(hdr)

    # Extract relevant cube
    zmask = np.zeros_like(wav_axis, dtype=bool)
    if len(wrange) != 0:
        zmask[(wav_axis < wrange[0]) | (wav_axis > wrange[1])] = 1
    if wavegood==True:
        zmask[(wav_axis < hdr['WAVGOOD0']) | (wav_axis > hdr['WAVGOOD1'])] = 1
    wav_axis = wav_axis[~zmask]
    data = data[~zmask]
    var = var[~zmask]
    
    # resize the old cube
    def resize(cube, binsize):
        
        if binsize==1:
            return cube
        
        cube = cube.copy()
        bs = int(binsize)
        
        # trim off edges
        sh = cube.shape
        if sh[1]%bs != 0:
            cube = cube[:, 0:sh[1]-sh[1]%bs, :]
        if sh[2]%bs != 0:
            cube = cube[:, :, 0:sh[2]-sh[2]%bs]
        sh = cube.shape
        
        shape = (sh[0], int(sh[1]/bs), bs,
             int(sh[2]/bs), bs)
        cube_reshape = cube.reshape(shape)
        
        # remove bins with 0
        for k in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[3]):
                    tmp = cube_reshape[k,i,:,j,:]
                    if 0 in tmp:
                        cube_reshape[k,i,:,j,:] = 0
        return cube_reshape.sum(-1).sum(2)

    # get independent scaling measurements
    bin_all = []
    fac_all = []
    if xbins is None:
        bin_grid = np.linspace(1, np.min(data.shape[1:3])/5, 10).astype(int)
    else:
        bin_grid = np.array(xbins).astype(int)
    index_z = np.arange(0, data.shape[0]-nw, nw).astype(int)
    for k in tqdm(range(nw)):
        for i in np.flip(bin_grid):
        
            cube_bin = resize(data[index_z+k, :, :], i)
            if np.sum(cube_bin != 0)<10:
                continue
            v_p = np.std(cube_bin[cube_bin != 0])

            var_bin = resize(var[index_z+k, :, :], i)
            v_t = np.sqrt(np.median(var_bin[cube_bin != 0]))

            bin_all.append(i)
            fac_all.append(v_p/v_t)
    bin_all = np.array(bin_all)**2
    fac_all = np.array(fac_all)            
            
    # fitting
    bin_fit = bin_all.copy()
    fac_fit = fac_all.copy()
    for i in range(niter):
        param, pcov = optimize.curve_fit(cov_curve, bin_fit, fac_fit)
        curve = cov_curve(bin_fit, *param)
        rms =  np.sqrt(np.mean(((fac_fit / curve) - 1)**2))
        index = np.abs((fac_fit / curve) - 1) > (nsig * rms)
        if np.sum(index)==0:
            break
        bin_fit = bin_fit[~index]
        fac_fit = fac_fit[~index]
        
    # output
    fits_out = update_cov_header(fits_in, *param)
    
    if return_all:
        return fits_out, param, bin_all, fac_all 
    return fits_out

