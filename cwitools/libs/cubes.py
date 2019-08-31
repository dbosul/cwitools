"""CWITools library for common data-cube manipulation routines."""

from astropy.io import fits
import numpy as np
import os
import sys

def get_indices(w1,w2,header):
    """Returns wavelength indices for two given wavelengths in Angstrom

    Args:
        w1 (float): Lower wavelength, in Angstrom.
        w2 (float): Upper wavelength, in Angstrom.
        header (astropy.io.fits.Header): FITS header for this data cube.

    Returns:
        int tuple: The lower and upper wavelength indices for this range.

    Example:

        To find the indices corresponding to the wavelength range 4100-4200A:

        >>> from astropy.io import fits
        >>> from cwitools import cubes
        >>> mydata,header = fits.getdata("mydata.fits",header=True)
        >>> a,b = cubes.get_band(4100,4200,header)
        >>> mydata_trimmed = mydata[a:b]

    """
    w0,dw,p0 = header["CRVAL3"],header["CD3_3"],header["CRPIX3"]
    w0 -= p0*dw
    return ( int((w1-w0)/dw), int((w2-w0)/dw) )

def get_header1d(header3d):
    """Remove the spatial axes from a a 3D FITS Header.

    Args:
        header3d (astropy.io.fits.Header): Input 3D header.

    Returns:
        astropy.io.fits.Header: 1D Header object (wavelength axis only).

    Example:

        If you wanted to collapse a cube to a spectrum and save it:

        >>> from astropy.io import fits
        >>> from cwitools import cubes
        >>> import numpy as np
        >>> mydata,header = fits.getdata("mydata.fits",header=True)
        >>> myspec_data = np.sum(mydata,axis=(1,2)) #Sum over spatial axes
        >>> myspec_header = cubes.get_header1d(header)
        >>> myspec_fits = cubes.make_fits(myspec_data,myspec_header)
        >>> myspec_fits.writeto("myspec.fits")

        Note: example does not properly handle any physical unit conversion!

    """

    hdr1D = header3d.copy()
    for key,val in list(header3d.items()):
        if '1' in key or '2' in key:
            del hdr1D[key]
        elif '3' in key:
            hdr1D[key.replace('3','1')] = val
            del hdr1D[key]
    del hdr1D["NAXIS1"]
    hdr1D.insert(2,"NAXIS1")

    hdr1D["NAXIS1"]  = header3d["NAXIS3"]
    hdr1D["NAXIS"]   = 1
    hdr1D["WCSDIM"]  = 1
    return hdr1D

def get_header2d(header3d):
    """Remove the spectral axis from a 3D FITS Header

        Args:
            header3d (astropy.io.fits.Header): Input 3D header.

        Returns:
            astropy.io.fits.Header: 2D Header object (spatial axes only)

        Example:

            If you wanted to collapse a cube to a 2D image and save it:

            >>> from astropy.io import fits
            >>> from cwitools import cubes
            >>> import numpy as np
            >>> mydata,header = fits.getdata("mydata.fits",header=True)
            >>> image_data = np.sum(mydata,axis=(0)) #Sum over wavelength
            >>> image_header = cubes.get_header2d(header)
            >>> image_fits = cubes.make_fits(image_data,image_header)
            >>> image_fits.writeto("image2D.fits")

            Note again - this example doesn't convert to surface-brightness or
            do any proper handling of units. That's up to you!

    """
    hdr2D = header3d.copy()
    for key in list(hdr2D.keys()):
        if '3' in key:
            del hdr2D[key]
    hdr2D["NAXIS"]   = 2
    hdr2D["WCSDIM"]  = 2

    return hdr2D

def get_wavaxis(header):
    """Returns a NumPy array representing the wavelength axis of a cube.

    Args:
        header (astropy.io.fits.Header): Can be 1D (spectrum) or 3D (cube) header.

    Returns:
        numpy.ndarray: Wavelength axis for this data.

    Examples:

        If you wanted to plot your spectrum vs. wavelength in matplotlib:

        >>> import matplotlib.pyplot as plt
        >>> from cwitools import cubes
        >>> from astropy.io import fits
        >>> spec,header = fits.getdata("myspectrum.fits",header=True)
        >>> wav_axis = cubes.get_wavaxis(header)
        >>> fig,ax = plt.subplots(1,1)
        >>> ax.plot(wav_axis,spec)
        >>> fig.show()


    """
    if header["NAXIS"]==3: return np.array([ header["CRVAL3"] + (i-header["CRPIX3"])*header["CD3_3"] for i in range(header["NAXIS3"])])
    elif header["NAXIS"]==1: return np.array([ header["CRVAL1"] + (i-header["CRPIX1"])*header["CD1_1"] for i in range(header["NAXIS1"])])

def make_fits(data,header):
    """A convenient wrapper for making a new FITS object with astropy.

    Args:
        data (numpy.ndarray): The data for the new fits
        header (astropy.io.fits.Header): The associated header

    Returns:
        astropy.io.fits.HDUList: The new hdulist/fits object.

    """
    hdu = fits.PrimaryHDU(data)
    hdu.header = header
    newfits = fits.HDUList([hdu])

    return newfits
