from scipy.stats import sigmaclip

import numpy as np
import warnings

def estimate_variance(inputfits, zwindow=10, rescale=True, sigmaclip=4,
zmask=(0, 0), fmin=0.9, fmax=10):
    """Estimates the 3D variance cube of an input cube.

    Args:
        inputfits (astropy.io.fits.HDUList): FITS object to estimate variance of.
        zWindow (int): Size of z-axis bins to use for 2D variance estimation. Default: 10.
        rescale (bool): Set to TRUE to perform layer-by-layer rescaling of 2D variance.
        sigmaclip (float): Threshold (in stddevs) for sigma-clipping data before estimation.
        zmask (int tuple): Wavelength layers to exclude while estimating variance.
        fMin (float): The minimum rescaling factor (Default 0.9)
        fMax (float): The maximum rescaling factor (Default: 10)
        fileExt (str): The extension to use for the output cube (Default .var.fits)

    Returns:

        NumPy ndarray: Estimated variance cube

    Examples:

        >>> from astropy.io import fits
        >>> from cwitools.variance import estimate_variance
        >>> myfits = fits.open("mydata.fits")
        >>> varcube = estimate_variance(myfits)
        >>> varfits = fits.HDUList([fits.primaryHDU(varcube)])
        >>> varfits[0].header = myfits[0].header
        >>> varfits.writeto("mydata_var.fits")

    """

    cube = inputfits[0].data
    z0, z1 = zmask
    dz = zwindow

    #Output warning
    if z1 - z0 >= dz:
        warnings.warn("""Your z-mask is large relative to your zwindow size.\
        \nVariance estimate may be unreliable.""")

    #Parse boolean input
    rescale = True if rescale == "True" else False

    #Run sigma-clip if set
    if sigmaclip > 0:
        cube = sigma_clip(cube, sigma=sigmaclip).data

    #Make first estimate by binning data
    varcube = np.zeros_like(cube)
    i = 0
    a, b = (i * dz), (i + 1) * dz
    while b < cube.shape[0]:
        varcube[a:b] = np.var(cube[a:b], axis=0)
        i += 1
        a, b = (i * dz), (i + 1) * dz
    varcube[a:] = np.var(cube[a:], axis=0)

    #Adjust first estimate by rescaling, if set to do so
    if rescale:
        varcube = rescale_var(varcube, cube, fmin=fmin, fmax=fmax)

    return varcube

def rescale_var(varcube, datacube, fmin=0.9, fmax=10):
    """Rescale a variance cube layer-by-layer to reflect the noise of a data cube.

    Args:
        varcube (NumPy.ndarray): Variance cube to rescale.
        datacube (NumPy.ndarray): Data cube corresponding to variance cube.
        fmin (float): Minimum rescaling factor (Default=0.9)
        fmax (float): Maximum rescaling factor (Default=10.0)

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

        sig = np.sqrt(varcube[wi])
        useXY = sig > 0
        varNorm = np.var(datacube[wi][useXY] / sig[useXY])

        #Normalize so that variance of layer as a whole is ~1
        #
        # Note: this assumes most of the 3D field is empty of real signal.
        # Z and XY Masks should be supplied if that is not the case
        #
        rsFactor = (1 / varNorm)

        rsFactor = max(rsFactor, fmin)
        rsFactor = min(rsFactor, fmax)

        varcube[wi] *= rsFactor

    return varcube
