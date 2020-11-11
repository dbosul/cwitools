"""Reduction tools related to variance estimation."""

#Standard Imports

#Third-party Imports
from astropy.modeling import models, fitting
from scipy.stats import sigmaclip
from skimage import measure
from tqdm import tqdm

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

#Local Imports
from cwitools import coordinates, modeling, utils

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
    hdu = utils.extract_hdu(inputfits)

    varcube = np.zeros_like(hdu.data)
    z_indices = np.arange(hdu.data.shape[0])
    wav_axis = coordinates.get_wav_axis(hdu.header)

    #Create wavelength masked based on input
    zmask = np.ones_like(wav_axis, dtype=bool)
    if wmasks is not None:
        for pair in wmasks:
            zmask[(wav_axis > pair[0]) & (wav_axis < pair[1])] = 0
    nzmax = np.count_nonzero(zmask)

    #Loop over wavelength first to minimize repetition of wl-mask calculation
    for z_j in z_indices:

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

    if var_rescale_factor < 1:
        print("\nWARNING: INPUT VARIANCE APPEARS TO BE OVER-ESTIMATED, AND WILL BE SCALED DOWN (SCALING FACTOR: %.3f). IT IS IMPORTANT THAT YOU VERIFY THAT THIS IS APPROPRIATE - INCORRECTLY SCALING DOWN THE VARIANCE CAN LEAD TO FALSE POSITIVES AND INCORRECT RESULTS." % var_rescale_factor)

    return var * var_rescale_factor, var_rescale_factor

def fit_covar_xy(fits_in, var, mask=None, wrange=None, xybins=None, n_w=10, wavgood=True,
                 return_all=False, model_bounds=None, mask_sky=True, mask_neb=None, plot=False):
    """Fits a two-component model to the noise as a function of bin size.

    The model used can be found in modeling.covar_curve

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        var (np.array): Variance cube.
        mask (np.array): Mask cube, M, where M > 0 excludes pixels.
        xybins (np.array): List of spatial bin sizes. Default is a list of
            10 points evenly distributed between 1 and 1/5 of the shortest
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

    hdu = utils.extract_hdu(fits_in)
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
        nebmask = utils.get_nebmask(hdr, redshift=mask_neb, vel_window=2000)
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
    z_indices = np.arange(0, data.shape[0] - n_w, n_w).astype(int)

    # 'z_shift' shifts these indices along by 1 each time, selecting a different
    # sub-cube made up of independent z-layers
    for z_shift in tqdm(range(n_w)):
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
    hdu = utils.extract_hdu(fits_in)
    data = hdu.data.copy()
    hdr = hdu.header.copy()
    alpha, norm, thresh = params
    beta = norm * (1 + alpha * np.log(thresh))
    hdr["COV_ALPH"] = alpha
    hdr["COV_NORM"] = norm
    hdr["COV_THRE"] = thresh
    hdr["COV_BETA"] = beta
    fits_out = utils.match_hdu_type(fits_in, data, hdr)
    return fits_out
