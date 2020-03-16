"""Tools for variance estimation and scaling."""
from cwitools import coordinates
from scipy.stats import sigmaclip

import numpy as np
import warnings

def estimate_variance(inputfits, window=50, sclip=None, wmasks=[], fmin=0.9):
    """Estimates the 3D variance cube of an input cube.

    Args:
        inputfits (astropy.io.fits.HDUList): FITS object to estimate variance of.
        window (int): Wavelength window (Angstrom) to use for local 2D variance estimation.
        wmasks (list): List of wavelength tuples to exclude when estimating variance.
        sclip (float): Sigmaclip threshold to apply when comparing layer-by-layer noise.
        fMin (float): The minimum rescaling factor (Default 0.9)

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
    varcube = rescale_var(varcube, cube, fmin=fmin, sclip=sclip)

    rescaleF = np.var(cube) / np.mean(varcube)
    varcube *= rescaleF

    return varcube

def rescale_var(varcube, datacube, fmin=0.9, sclip=4):
    """Rescale a variance cube layer-by-layer to reflect the noise of a data cube.

    Args:
        varcube (NumPy.ndarray): Variance cube to rescale.
        datacube (NumPy.ndarray): Data cube corresponding to variance cube.
        fmin (float): Minimum rescaling factor (Default=0.9)

    Returns:

        NumPy ndarray: Rescaled variance cube

    Examples:

        >>> from astropy.io import fits
        >>> from cwitools.variance import rescale_variance
        >>> data = fits.open("data.fits")
        >>> var  = fits.getdata("variance.fits")
        >>> var_rescaled = rescale_variance(var, data)

    """
    for wi in range(varcube.shape[0]):

        useXY = varcube[wi] > 0

        layer_data = datacube[wi][useXY]
        layer_var = varcube[wi][useXY]

        # if sclip != None:
        #     layer_data = sigmaclip(layer_data, low=sclip, high=sclip).clipped
        #     layer_var = sigmaclip(layer_var, low=sclip, high=sclip).clipped

        rsFactor = np.var(layer_data) / np.mean(layer_var)
        rsFactor = max(rsFactor, fmin)

        varcube[wi] *= rsFactor

    return varcube
