from astropy import convolution
from cwitools.modeling import fwhm2sigma
import numpy as np

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
