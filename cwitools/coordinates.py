"""Tools for working with headers and world coordinate systems."""
from astropy import units as u
from astropy.cosmology import WMAP9
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
<<<<<<< HEAD
from astropy.cosmology import WMAP9

=======
from cwitools import utils
>>>>>>> v0.6_dev2
import numpy as np

def get_flam2sb(header):
    """Get the conversion factor from FLAM units to surface brightness.

    Conversion is between erg/s/cm2/angstrom and erg/s/cm2/arcsec2.

    Args:
        header: Header for 3D data.

    Returns:
        float: conversion factor from FLAM to SB units

    """
    return get_pxsize_angstrom(header) / get_pxarea_arcsec(header)

def get_pxsize_angstrom(header):
    """Get the pixel/wavelenght-layer size in units of Angstrom.

    Args:
        header: Header for 3D data.

    Returns:
        float: size of the z-pixels (i.e. wavelength layers,) in Angstrom.

    """
    if header["NAXIS"] != 3:
        raise ValueError("Function only takes 3D input.")
    pxscales = proj_plane_pixel_scales(WCS(header))
    wscale = (pxscales[2] * u.meter).to(u.angstrom).value
    return wscale

def get_pxarea_arcsec(header):
    """Get the pixel area in arcsec2.

    Args:
        header: Header for 3D or 2D data.

    Returns:
        float: size of the spaxels in arcseconds squared.

    """
    if header["NAXIS"] == 3:
        header = get_header2d(header)
    elif header["NAXIS"] != 2:
        raise ValueError("Function only takes 2D or 3D input.")
    yscale, xscale = proj_plane_pixel_scales(WCS(header))
    yscale = (yscale * u.deg).to(u.arcsec).value
    xscale = (xscale * u.deg).to(u.arcsec).value
    pxsize = yscale * xscale
    return pxsize

<<<<<<< HEAD
def get_rgrid(fits_in, pos, unit='px', redshift=None,
postype='image', cosmo=WMAP9):
=======
def get_rgrid(fits_in, pos, unit='px', redshift=None, postype='image',
cosmo=WMAP9):
>>>>>>> v0.6_dev2
    """Get a 2D grid of radius from x,y in specified units.

    Args:
        fits_in (HDU or HDUList): HDU or HDUList containing 2D or 3D data.
        pos (float tuple): The position to center on, in image coordinates.
        unit (str): The desired units for the output grid.
            'px' - pixels
            'arcsec' - arcseconds
            'pkpc' - proper kiloparsecs
            'ckpc' - comoving kiloparsecs
        redshift (float): The redshift of the source, required to calculate
            the grid in units of pkpc or ckpc.
        postype (str): The type of coordinate given for the 'pos' argument.
            'radec' - a tuple of (RA, DEC) coordinates, in decimal degrees
            'image' - a tuple of image coordinates, in pixels
        cosmo (FlatLambdaCDM): The cosmology to use, as one of Astropy's
            cosmologies (astropy.cosmology.FlatLambdaCDM). Default is WMAP9.

    Returns:
        numpy.ndarray: 2D array of distance from `pos` in the requested units.

    """
    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data, hdu.header

    if unit not in ['px', 'arcsec', 'pkpc', 'ckpc']:
        raise ValueError("Unit must be 'px', 'arcsec', 'pkpc', or 'ckpc'")

    #Determine nature of input
    naxis = header["NAXIS"]
    if naxis == 3:
        header2d = get_header2d(header)
        img2d = np.mean(data, axis=0)
    elif naxis == 2:
        header2d = header
        img2d = data
    else:
        raise ValueError("Function only takes 2D or 3D input.")

    #If RA/DEC position given, convert to image coordinates
    if postype == 'radec':
        wcs2d = WCS(header2d)
        pos = tuple(float(x) for x in wcs2d.all_world2pix(pos[0], pos[1], 0))
    elif postype != 'image':
        raise ValueError("postype argument must be 'image' or 'radec'")

    #Get meshgrid of x and y positions
    xx, yy = np.indices(img2d.shape, dtype=float)

    #Center on source
    xx -= pos[0]
    yy -= pos[1]

    #Convert x/y grids to arcsec if arcsec OR physical units requested
    if unit in ['arcsec', 'pkpc', 'ckpc']:
        yscale, xscale = proj_plane_pixel_scales(WCS(header2d))
        yy *= (yscale * u.deg).to(u.arcsec).value
        xx *= (xscale * u.deg).to(u.arcsec).value

    #Now calculate radial distance
    rr = np.sqrt(xx**2 + yy**2)

    #If physical units requested, convert the rr grid from arcsec to kpc
    if unit in ['pkpc', 'ckpc']:
        if redshift is None:
            raise ValueError("Redshift must be provided to calculate kpc units.")
        #Get kpc/arcsec from cosmology
        if unit == 'pkpc':
            kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift) /  60.0
        elif type == 'ckpc':
            kpc_per_arcsec = cosmo.kpc_comoving_per_arcmin(redshift) / 60.0
        else:
            raise ValueError("Type must be 'proper' or 'comoving'")
        rr *= kpc_per_arcsec

    #Return distance meshgrid
    return rr

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

def get_kpc_per_px(header, redshift=0, type='proper', cosmo=WMAP9):
    """Return the physical size of pixels in proper kpc. Assumes 1:1 aspect ratio.

    Args:
        header (astropy.hdu.header): Header of a 2D or 3D Astropy HDU.
        redshift (float): Cosmological redshift of the field/target.
        type (str): Type of kiloparsec ('proper' or 'comoving') to return.
        cosmo (FlatLambdaCDM): Cosmology to use, as one of the inbuilt
            astropy.cosmology.FlatLambdaCDM instances (default WMAP9)

    Returns:
        float: Proper or comoving kiloparsecs per pixel

    Examples:

        Note that this method assumes the spatial axes are equal in scale and
        that the WCS is either (deg, deg, wavelength) or (deg, deg).

        >>> from astropy.io import fits
        >>> from cwitools.coordinates import get_kpc_per_px
        >>> z_target = 1.5
        >>> data, header = fits.getdata("targetdata.fits", header=True)
        >>> px_scale_pkpc = get_kpc_per_px(header, redshift=z_target)

    """
    wcs = WCS(header)

    #Get platescale in arcsec/px (assumed to be 1:1 aspect ratio)
    arcmin_per_px = (proj_plane_pixel_scales(wcs)[1] * u.deg).to(u.arcmin)

    #Get kpc/arcsec from cosmology
    if type == 'proper':
        kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(redshift)
    elif type == 'comoving':
        kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(redshift)
    else:
        raise ValueError("Type must be 'proper' or 'comoving'")

    #Get kpc/pixel by combining
    kpc_per_px = (arcmin_per_px * kpc_per_arcmin).value

    return kpc_per_px


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
