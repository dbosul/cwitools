"""Tools for working with headers and world coordinate systems."""

#Standard Imports
import warnings

#Third-party Imports
from astropy import units as u
from astropy.cosmology import WMAP9
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
import reproject

#Local Imports
from cwitools import utils

def reproject_hdu(hdu1, header, method="interp-bicubic"):
    """Reproject the WCS and data of one HDU to match another using 'Reproject'.

    Note: this is just a convenience wrapper of tools from the Reproject package.

    Args:
        hdu1 (HDU): The HDU to be reprojected.
        header (astropy.FITS.header): The header to be matched to.
        method (str): Method to use from 'Reproject' package.
            "interp-nearest-neighbor"
            "interp-bilinear"
            "interp-bicubic" (Default)
            "exact"

    Returns:
        astropy.fits.HDU: The scaled HDU        

    """

    if 'interp' in method:
        tmp = method.split('-')
        order = '-'.join(tmp[1:])
        scaled_data, _ = reproject.reproject_interp(hdu1, header, order=order)
    elif 'exact' in method:
        scaled_data, _ = reproject.reproject_exact(hdu1, header)
    else:
        raise ValueError('Reprojection method not recognized.')

    scaled_header = hdu1.header.copy()

    for wcs_key in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                    'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                    'NAXIS1', 'NAXIS1']:
        scaled_header[wcs_key] = header[wcs_key]

    hdu_out = utils.match_hdu_type(hdu1, scaled_data, scaled_header)
    return hdu_out


def scale_hdu(hdu, upscale, header_only=False, reproject_mode="interp-bicubic"):
    """Scale the data and/or header in an HDU up by a factor.

    Note: this is just a convenience wrapper of tools from the Reproject package.

    Args:
        hdu (HDU): The HDU to be scaled up.
        upscale (float): The factor to scale the data/WCS up by.
        header_only (bool): Set to only scale the Header (i.e. WCS)
        reproject_mode (str): Method to use from 'Reproject' package.
            "interp-nearest-neighbor"
            "interp-bilinear"
            "interp-bicubic" (Default)
            "exact"

    Returns:
        astropy.fits.HDU: The scaled HDU
        
    Example:
        
        To zoom in by a factor of 2:
        
        >>> hdu_new = scale_hdu(hdu_old, 2)
        
        To zoom out by a factor of 2:
        
        >>> hdu_new = scale_hdu(hdu_old, 0.5)
        
        If you only need the header without actually projecting the data,
        
        >>> hdu_new = scale_hdu(hdu_old, factor)
        >>> header_new = hdu_new.header
    

    """

    hdu_up = hdu.copy()

    if upscale == 1:
        warnings.warn("Scale factor given as 1. There will be no change.")
        return hdu

    hdr_up = hdu_up.header.copy()

    hdr_up['NAXIS1'] = int(hdr_up['NAXIS1'] * upscale)
    hdr_up['NAXIS2'] = int(hdr_up['NAXIS2'] * upscale)
    hdr_up['CRPIX1'] = (hdr_up['CRPIX1'] - 0.5) * upscale + 0.5
    hdr_up['CRPIX2'] = (hdr_up['CRPIX2'] - 0.5) * upscale + 0.5

    for cd_key in ['CD1_1', 'CD2_1', 'CD1_2', 'CD2_2']:
        hdr_up[cd_key] /= upscale

    if not header_only:
        if 'interp' in reproject_mode:
            tmp = reproject_mode.split('-')
            order = '-'.join(tmp[1:])
            hdu_up.data, _ = reproject.reproject_interp(hdu, hdr_up, order=order)
        elif 'exact' in reproject_mode:
            hdu_up.data, _ = reproject.reproject_exact(hdu, hdr_up)
        else:
            raise ValueError('Reprojectio method not recognized.')

    hdu_up.header = hdr_up

    return hdu_up


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


def get_rgrid(fits_in, pos, unit='px', redshift=None, pos_type='image', cosmo=WMAP9):
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
        pos_type (str): The type of coordinate given for the 'pos' argument.
            'radec' - a tuple of (RA, DEC) coordinates, in decimal degrees
            'image' - a tuple of image coordinates, in pixels
        cosmo (FlatLambdaCDM): The cosmology to use, as one of Astropy's
            cosmologies (astropy.cosmology.FlatLambdaCDM). Default is WMAP9.

    Returns:
        numpy.ndarray: 2D array of distance from `pos` in the requested units.

    """
    hdu = utils.extract_hdu(fits_in)
    data, header = hdu.data.copy(), hdu.header.copy()

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
    if pos_type == 'radec':
        wcs2d = WCS(header2d)
        pos = tuple(float(x) for x in wcs2d.all_world2pix(pos[0], pos[1], 0))
    elif pos_type != 'image':
        raise ValueError("pos_type argument must be 'image' or 'radec'")

    #Get meshgrid of x and y positions
    ygrid, xgrid = np.indices(img2d.shape, dtype=float)

    #Center on source
    xgrid -= pos[0]
    ygrid -= pos[1]

    #Convert x/y grids to arcsec if arcsec OR physical units requested
    if unit in ['arcsec', 'pkpc', 'ckpc']:
        xscale, yscale = proj_plane_pixel_scales(WCS(header2d))
        ygrid *= (yscale * u.deg).to(u.arcsec).value
        xgrid *= (xscale * u.deg).to(u.arcsec).value

    #Now calculate radial distance
    rgrid = np.sqrt(xgrid**2 + ygrid**2)

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
        rgrid *= kpc_per_arcsec.value

    #Return distance meshgrid
    return rgrid

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
    header1d = header3d.copy()

    #Delete all header keywords for axes 1 and 2 (of 3)
    for key, val in header3d.items():

        if '1' in key or '2' in key:
            del header1d[key]

        elif '3' in key:
            header1d[key.replace('3', '1')] = val
            del header1d[key]

    #Delete old NAXIS1
    del header1d["NAXIS1"]

    #Replace NAXIS1 in appropriate position with old NAXIS3 value
    header1d.insert(2, "NAXIS1")
    header1d["NAXIS1"] = header3d["NAXIS3"]

    #Update that the WCS is now one-dimensional
    header1d["NAXIS"] = 1
    header1d["WCSDIM"] = 1

    return header1d

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
    header2d = header3d.copy()

    #Delete all axis 3 keywords
    for key in header2d.keys():
        if '3' in key:
            del header2d[key]

    #Reduce dimensionality to 2
    header2d["NAXIS"] = 2
    header2d["WCSDIM"] = 2

    return header2d

def get_kpc_per_px(header, redshift=0, unit='pkpc', cosmo=WMAP9):
    """Return the physical size of pixels in proper kpc. Assumes 1:1 aspect ratio.

    Args:
        header (astropy.hdu.header): Header of a 2D or 3D Astropy HDU.
        redshift (float): Cosmological redshift of the field/target.
        unit (str): Proper ('pkpc') or comoving ('ckpc') kiloparsecs. Default: pkpc.
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
    if unit == 'pkpc':
        kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(redshift)
    elif unit == 'ckpc':
        kpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(redshift)
    else:
        raise ValueError("Type must be 'proper' or 'comoving'")

    #Get kpc/pixel by combining
    kpc_per_px = (arcmin_per_px * kpc_per_arcmin).value

    return kpc_per_px


def get_indices(wav1, wav2, header, bounded=True):
    """Returns wavelength layer indices for two given wavelengths in Angstrom.

    Args:
        wav1 (float): Lower wavelength, in Angstrom.
        wav2 (float): Upper wavelength, in Angstrom.
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
    wav0, dwav, pix0 = header["CRVAL3"], header["CD3_3"], header["CRPIX3"]
    wav0 -= pix0 * dwav

    index_lo = int(round((wav1 - wav0) / dwav))
    index_hi = int(round((wav2 - wav0) / dwav))

    if bounded:
        index_lo = max(0, index_lo)
        index_hi = min(header["NAXIS3"] - 1, index_hi)

    return index_lo, index_hi



def get_wav_axis(header):
    """Returns a NumPy array representing the wavelength axis of a cube.

    Args:
        header (astropy.io.fits.Header): header that contains wavelength
            or velocity axis that is specified in 'CTYPE' keywords in any 
            dimension.

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

    """

    #Select the appropriate axis.
    naxis = header['NAXIS']
    flag = False
    for i in range(naxis):
        #Keyword entry
        card = "CTYPE{0}".format(i+1)
        if not card in header:
            raise ValueError("Header must contain 'CTYPE' keywords.")
        
        #Possible wave types.
        if header[card] in ['AWAV', 'WAVE', 'VELO']:
            axis = i+1
            flag = True
            break

    #No wavelength axis
    if flag == False:
        raise ValueError("Header must contain a wavelength/velocity axis.")

    #Get keywords defining wavelength axis
    nwav = header["NAXIS{0}".format(axis)]
    wav0 = header["CRVAL{0}".format(axis)]
    dwav = header["CD{0}_{0}".format(axis)]
    pix0 = header["CRPIX{0}".format(axis)]

    #Calculate and return
    return np.array([wav0 + (i - pix0) * dwav for i in range(nwav)])
