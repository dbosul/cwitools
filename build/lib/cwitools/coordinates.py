"""Tools for working with headers and world coordinate systems."""
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.wcs.utils import proj_plane_pixel_scales

import numpy as np

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

    #Delete all header keywords for axes 1 and 2 (of 3)
    for key, val in header3d.items():

        if '1' in key or '2' in key:
            del hdr1D[key]

        elif '3' in key:
            hdr1D[key.replace('3','1')] = val
            del hdr1D[key]

    #Delete old NAXIS1
    del hdr1D["NAXIS1"]

    #Replace NAXIS1 in appropriate position with old NAXIS3 value
    hdr1D.insert(2, "NAXIS1")
    hdr1D["NAXIS1"] = header3d["NAXIS3"]

    #Update that the WCS is now one-dimensional
    hdr1D["NAXIS"]  = 1
    hdr1D["WCSDIM"] = 1

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

    #Delete all axis 3 keywords
    for key in hdr2D.keys():
        if '3' in key:
            del hdr2D[key]

    #Reduce dimensionality to 2
    hdr2D["NAXIS"]   = 2
    hdr2D["WCSDIM"]  = 2

    return hdr2D

def get_pkpc_per_px(wcs, redshift=0):
    """Return the physical size of pixels in proper kpc. Assumes 1:1 aspect ratio.

    Args:
        wcs (astropy.wcs.WCS): A 2D or 3D astropy WCS object.
        redshift (float): Cosmological redshift of the field/target.

    Returns:
        float: Proper kiloparsecs per pixel

    Examples:

        Note that this method assumes the spatial axes are equal in scale and
        that the WCS is either (deg, deg, wavelength) or (deg, deg).

        >>> from astropy.io import fits
        >>> from astropy.wcs import WCS
        >>> from cwitools.coordinates import get_pkpc_px
        >>> z_target = 1.5
        >>> data, header = fits.getdata("targetdata.fits", header=True)
        >>> wcs = WCS(header)
        >>> px_scale_pkpc = pkpc_per_px(wcs, redshift=z_target)

    """
    #Get platescale in arcsec/px (assumed to be 1:1 aspect ratio)
    arcmin_per_px = (proj_plane_pixel_scales(wcs)[1] * u.deg).to(u.arcmin)

    #Get pkpc/arcsec from cosmology
    pkpc_per_arcmin = cosmo.kpc_proper_per_arcmin(redshift)

    #Get pkpc/pixel by combining
    pkpc_per_px = (arcmin_per_px * pkpc_per_arcmin).value

    return pkpc_per_px


def get_indices(w1, w2, header):
    """Returns wavelength layer indices for two given wavelengths in Angstrom.

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
    w0, dw, p0 = header["CRVAL3"], header["CD3_3"], header["CRPIX3"]
    w0 -= p0 * dw
    return (int((w1 - w0) / dw), int((w2 - w0) / dw))


def get_wav_axis(header):
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
        >>> wav_axis = cubes.get_wav_axis(header)
        >>> fig,ax = plt.subplots(1,1)
        >>> ax.plot(wav_axis,spec)
        >>> fig.show()

        Note: input header must be 3D (X, Y, WAV) or 1D (WAV).

    """

    #Select the appropriate axis.
    if header["NAXIS"] == 3:
        axis = 3
    elif header["NAXIS"] == 1:
        axis = 1
    else:
        raise ValueError("Header must be 1D or 3D to get wavelength axis.")

    #Get keywords defining wavelength axis
    Nwav = header["NAXIS{0}".format(axis)]
    wav0 = header["CRVAL{0}".format(axis)]
    dwav = header["CD{0}_{0}".format(axis)]
    pix0 = header["CRPIX{0}".format(axis)]

    #Calculate and return
    return np.array([wav0 + (i - pix0) * dwav for i in range(Nwav)])
