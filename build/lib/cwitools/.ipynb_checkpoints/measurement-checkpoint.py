"""Tools for kinematic calculations."""
import numpy as np
from astropy import units as u
from astropy.cosmology import WMAP9
from astropy.wcs import WCS
from cwitools import utils, coordinates

def first_moment(x, y, y_var=[], get_err=False, method='basic', m1_init=None,
    window_size=25, window_min=10, window_step=1):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        y_var (np.array): Variance on y. Taken as var(y) if not provided.
        get_err (bool): Set to TRUE to return (moment, error) tuple
        method (str): The method to use for calculating the moment.
            'basic': Use all input x and y data.
            'clw': Use the closing-window method (O'Sullivan+20, FLASHES Survey)
        m1_init (float): Initial estimate for m1, required if using 'clw' method
            or restricting calculatio to a certain window size.
        window_size (float): The default window size for the moment calculation
            if using m1_init, or the initial window size of using 'clw' method.
        window_min (float): The minimum window size if using 'clw' method.
        window_step (float): The decrement in window size each iteration if
            using the 'clw' method.

    Returns:
        float: The first moment in x.
        float: (if get_err == True) The estimated error on the first moment

    """

    #Estimate variance if no variance given
    y_var = np.var(y) if y_var == [] else y_var

    #Basic and positive methods are mostly the same
    if method == 'basic':

        #If an initial value is given, use only the window around m1_init
        if m1_init != None:
            usex = np.abs(x - m1_init) <= window_size
            x = x[usex]
            y = y[usex]

        #Perform moment calculation as normal
        num = np.sum(x * y) #Numerator
        den = np.sum(y) #Denominator

        m1 = num / den
        m1_err = np.sqrt(np.sum(y_var * (den * x - num)**2 )) / den**2

    #If iterative closing-window method is selected
    elif method == 'clw':

        #Start guess at center of array of no initial value given
        m1 = x[int(len(x) / 2)] if m1_init == None else m1_init

        #Initialize window at maximum size
        window = window_size

        # Loop with decreasing window size until minimum is reached
        while window > window_min:

             #Get indices of values to use for this calculation
             use = ( np.abs(x - m1) < window/2 ) & (y > 0)

             usex = x[use]
             usey = y[use]

             #Update moment calculation using current window
             num = np.sum(usex * usey) #Numerator
             den = np.sum(usey) #Denominator
             m1 = num / den
             m1_err = np.sqrt(np.sum(y_var * (den * x - num)**2 )) / den**2

             #Update window size
             window -= window_step

    if get_err:
        return m1, m1_err

    else:
        return m1


def second_moment(x, y, m1=None, y_var=[], get_err=False):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        m1 (float): The first moment. Calculated if not provided.
        y_var (np.array): Variance on y. Taken as var(y) if not provided.
        get_err (bool): Set to TRUE to return (moment, error), tuple

    Returns:
        float: The second moment in x.
        float: The error on the second moment (if get_err == True)

    """
    m1num = np.sum(x * y) #Numerator of first moment calculation
    m1den = m2den = np.sum(y) #Denominator of first/second moment calculation

    m1 = m1num / m1den if m1 == None else m1 #Calculate if not given

    m2num = np.sum(np.power(x-m1, 2) * y) #Numerator of second moment calc.
    m2den = np.sum((x - m1) * y) #Term needed for eq.

    m2 = np.sqrt(m2num / m1den) #Second moment


    if not(get_err):
        return m2

    else:

        R = np.sum((x - m1)*y) #Term needed for eq.
        dm2_dIj = (m1den * x - m1num) / (m1den**2) #Another term needed

        #Estimate if no variance given
        y_var = np.var(y) if y_var == [] else y_var

        #Two squared terms that are multiplied by variance
        term1 = (1 / (2 * m2den * m2den * m2))**2
        term2 = (m2den * np.power(x - m1, 2) + 2 * m2den * dm2_dIj * R - m2num)**2

        #Error on second moment
        m2_err = np.sqrt( term1*np.sum(y_var*term2) )

        return m2, m2_err

def luminosity(fits_in, redshift=None, mask=None, cosmo=WMAP9):
    """Measure the integrated luminosity from 1D, 2D or 3D data.

    Args:
        fits_in (astropy HDU or HDUList): The input data, 2D or 3D.
            If input is 2D, it is assumed to have units erg/s/cm2/arcsec2.
            If input is 1D or 3D, units are assumed to be erg/s/cm2/angstrom
        redshift (float): The redshift of the source.
        mask (numpy.ndarray): A binary mask of the 3D region to use, where a
            value of 1 means `include` and 0 means `exclude.' If none provided,
            the entire image or data cube is summed.
        cosmology (astropy FlatLambdaCDM): An astropy cosmology instance. WMAP9
            is used by default.
    Returns:
        float: The integrated luminosity of the source in erg/s.

    """

    #Extract data and header
    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data, hdu.header

    #Replace mask with array of all 1s if none given
    mask = np.ones_like(data, dtype=bool) if mask is None else mask

    #Check dimensions of mask match (only matters for user-provided masks)
    if mask.shape != data.shape:
        raise ValueError("mask must match dimensions of input data.")

    #Apply mask
    data[mask == 0] = 0

    #If the input data is 3D, convert to SB map first
    ndims = len(data.shape)
    if ndims == 3 or ndims == 1:
        #input units are FLAM erg/s/cm2/angstrom
        px_size_ang = coordinates.get_pxsize_angstrom(header)
        flux_total = np.sum(data) * px_size_ang #erg/s/cm2
    elif ndims == 2:
        #input units are erg/s/cm2/arcsec
        px_area_arcsec = coordinates.get_pxarea_arcsec(header)
        flux_total = np.sum(data) * px_area_arcsec #erg/s/cm2
    else:
        raise ValueError("Input data must be 1D, 2D or 3D.")

    #Calculate distance to source in cm
    lum_dist = cosmo.luminosity_distance(redshift).to(u.cm).value
    flux_to_lum = 4 * np.pi * (lum_dist**2) #units of 'cm2'

    #Convert total flux to luminosity and return
    lum_tot = flux_total * flux_to_lum
    return lum_tot


def moment2d(xx, yy, p, q, fxy):
    """Calculate image moment of order p in x and q in y.

    Mpq = SUM(xx^p * yy^q * fxy) / SUM(fxy)

    Args:
        xx (numpy.ndarray): 2D meshgrid of x-values
        yy (numpy.ndarray): 2D meshgrid of y-values
        p (int): Moment order in x.
        q (int): Moment order in y.
        fxy (numpy.ndarray): Array of weights representing the 2D distribution
            being measured; e.g. flux or surface brightness.

    Returns:
        float: The value of the requested image moment.

    """
    return np.sum(np.power(xx, p) * np.power(yy, q) * fxy) / np.sum(fxy)

def specific_ang_momentum(vel_map, flx_map):
    """Calculate the specific angular momentum of an object.

    j is defined here as SUM_xy(flux * |R_perp X v_z|) / SUM_xy(flux)
    where SUM_xy is summing over the spatial axes, R_perp is the projected
    radius from the flux-weighted center of mass of the emission, v_z is
    the average line-of-sight velocity in a spaxel and the X denotes the cross
    product.

    Args:
        vel_map (numpy.ndarray): 2D map of average velocity.
        dsp_map (numpy.ndarray): 2D map of velocity dispersion.
        flx_map (numpy.ndarray): 2D map of surface brightness/flux.

    Returns:
        float: The specific angular momentum, as defined above.
    """

    #Get mask of Nan values and replace with 0
    nan_msk = np.isnan(vel_map)
    vel_map[nan_msk] = 0

    #Get 2D array of flux weights
    flux_weights = (nan_msk) * flx_map

    #Create 2D meshgrids of x-position and y-position
    x, y = vel_map.shape
    X, Y = np.arange(x), np.arange(y)
    yy, xx = np.meshgrid(Y, X)

    #Calculate relevant moments for specific angular momentum calculation
    xcen = moment2d(xx, yy, 1, 0, flux_weights)
    ycen = moment2d(xx, yy, 0, 1, flux_weights)

    #Get meshgrid of distance from x, ycentroid
    rr = np.sqrt((xx - xcen)**2 + (yy - ycen)**2)

    #Calculate and return
    return np.sum(flux_weights * rr * np.abs(vel_map)) / np.sum(flux_weights)

def asymmetry(sb_map, obj_mask=None):
    """Calculate the spatial asymmetry (alpha) of a 2D or 3D object.

    Args:
        sb_map (numpy.ndarray): The 2D surface brightness map.
        obj_mask (numpy.ndarray): A 2D mask delineating the object.

    Returns:
        float: The calculated elliptical eccentricity of the object
    """

    #Generate blank mask if none given
    if obj_mask is None:
        obj_mask = np.ones_like(sb_map)

    #Apply mask
    sb_map[obj_mask == 0] = 0

    #Get spatial meshgrids
    xx, yy = np.indices(sb_map)

    #Calculate image moments
    M10 = moment2d(xx, yy, 1, 0, sb_map)
    M01 = moment2d(xx, yy, 0, 1, sb_map)

    #Get x and y centroids
    x_cen = M10
    y_cen = M01

    #Get x and y meshgrids centered on object
    xx_obj = xx - x_cen
    yy_obj = yy - y_cen
    rr_obj = np.sqrt(xx_obj**2 + yy_obj**2)

    #
    # Measure moments as in ArrigoniBattaia et al. 2019
    #
    M20 = moment2d(xx_obj, yy_obj, 2, 0, sb_map)
    M02 = moment2d(xx_obj, yy_obj, 0, 2, sb_map)
    M11 = moment2d(xx_obj, yy_obj, 1, 1, sb_map)
    Q = M20  - M02
    U = 2 * M11
    alpha = (1 - np.sqrt( Q**2 + U**2 ))/(1 + np.sqrt(Q**2 + U**2))

    return alpha


def eccentricity(sb_map, obj_mask=None):
    """Calculate the spatial eccentricity of a 2D or 3D object.

    Args:
        sb_map (numpy.ndarray): The 2D surface brightness map.
        obj_mask (numpy.ndarray): A 2D mask delineating the object.

    Returns:
        float: The calculated elliptical eccentricity of the object
    """

    #Get asymmetry first
    alpha = asymmetry(sb_map, obj_mask=obj_mask)

    #Convert from alpha (= minor/major axis ratio) to elliptical eccentricity
    eccentricity = np.sqrt(1 - alpha**2)

    return eccentricity

def centroid2d(fits_in, obj_mask=None, obj_id=1, coords='image'):
    """Measure the spatial centroid of an extended object.

    Args:
        fits_in (HDU or HDUList): 2D or 3D flux-like data.
        obj_mask (numpy.ndarray): 2D or 3D data with labelled object regions.
        obj_id (int or list): Integer or list of integers of object IDs to use.
        coords (str): Desired coordinate system for the output.
            'image': Center of mass is returned as an (x, y) tuple
            'radec': Center of mass is returned as an (RA, DEC) tuple

    Returns:
        float tuple: The center of mass in the requested coordinate system.
    """
    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data.copy(), hdu.header

    bin_mask = extraction.obj2binary(obj_mask, obj_id)

    #Remove non-object regions
    data[bin_mask == 0] = 0

    #Get 2D flux-like map, i.e. centroid weights.
    ndims = len(data.shape)
    if ndims == 3:
        weights2d = np.sum(data, axis=0)
    elif ndims == 2:
        weights2d = data
    else:
        raise ValueError("Input must be 2D or 3D in shape.")

    #Get spatial meshgrids
    xx, yy = np.indices(weights2d)

    #Calculate image moments
    x_cen = moment2d(xx, yy, 1, 0, weights2d)
    y_cen = moment2d(xx, yy, 0, 1, weights2d)

    if coords == 'image':
        return x_cen, y_cen

    elif coords == 'radec':
        if ndims == 3:
            wcs2d = WCS(coordinates.get_header2d(header))
        else:
            wcs2d = WCS(header)
        ra, dec = wcs2d.all_pix2world(x_cen, y_cen, 0)
        return ra, dec
    else:
        raise ValueError("coords argument must be 'image' or 'radec'")

def area(obj_in, obj_id=1, unit='px2'):
    """Measure the spatial (projected) area of a 2D or 3D object.

    Args:
        obj_in (astropy HDU or HDUList): 2D or 3D mask of object(s) with header.
        obj_id (int): The ID of the object to measure. Default is 1.
        unit (str): Output unit of radius measurement.
            'px2': square pixels
            'arcsec2': square arcseconds
        redshift (float): The redshift of the object(s)
        cosmology (FlatLambdaCDM): Cosmology to use, as one of the inbuilt
            ~astropy.cosmology.FlatLambdaCDM instances (default WMAP9)

    Returns:
        float: The effective radius in the requested units

    """
    obj_hdu = utils.extractHDU(obj_in)
    obj_mask, header = obj_hdu.data, obj_hdu.header

    bin_mask = obj_mask == obj_id
    ndims = len(bin_mask.shape)
    if ndims == 3:
        nspaxels = np.count_nonzero(np.max(bin_mask, axis=0))
    elif ndims == 2:
        nspaxels = np.count_nonzero(bin_mask)
    else:
        raise ValueError("Object mask must be 2D or 3D")

    if unit == 'arcsec2':

        px_size_arcsec2 = coordinates.get_pxarea_arcsec(header)
        return nspaxels * px_size_arcsec2

    elif unit == 'px':
        return nspaxels
    else:
        raise ValueError("Unit must be 'px2' or 'arcsec2'")

def effective_radius(obj_in, obj_id=1, unit='px', redshift=None, cosmo=WMAP9):
    """Determines the effective radius (sqrt(Area/pi)) of a 2D or 3D object.

    Args:
        obj_in (astropy HDU or HDUList): 2D or 3D mask of object(s) with header.
        obj_id (int): The ID of the object to measure. Default is 1.
        unit (str): Output unit of radius measurement.
            'px': pixels
            'arcsec': arcseconds
            'pkpc': proper kiloparsec
            'ckpc': comoving kiloparsec
        redshift (float): The redshift of the object(s)
        cosmology (FlatLambdaCDM): Cosmology to use, as one of the inbuilt
            ~astropy.cosmology.FlatLambdaCDM instances (default WMAP9)

    Returns:
        float: The effective radius in the requested units
    """
    if 'kpc' in unit and redshift is None:
        raise ValueError("Redshift must be provided to convert to kiloparsecs.")

    obj_hdu = utils.extractHDU(obj_in)
    obj_mask, header = obj_hdu.data, obj_hdu.header

    bin_mask = obj_mask == obj_id
    ndims = len(bin_mask.shape)
    if ndims == 3:
        nspaxels = np.count_nonzero(np.max(bin_mask, axis=0))
    elif ndims == 2:
        nspaxels = np.count_nonzero(bin_mask)
    else:
        raise ValueError("Object mask must be 2D or 3D")

    r_eff_px = np.sqrt(nspaxels / np.pi)

    if unit in ['arcsec', 'pkpc', 'ckpc']:
        px_size_arcsec2 = coordinates.get_pxarea_arcsec(header)
        r_eff_arcsec = r_eff_px * np.sqrt(px_size_arcsec2)

        if unit == 'pkpc':
            pkpc_per_arcsec = cosmo.kpc_proper_per_arcmin(redshift) / 60.0
            r_eff_pkpc = r_eff_arcsec * pkpc_per_arcsec
            return r_eff_pkpc
        elif unit == 'ckpc':
            ckpc_per_arcsec = cosmo.kpc_comoving_per_arcmin(redshift) / 60.0
            r_eff_ckpc = r_eff_arcsec * ckpc_per_arcsec
            return r_eff_ckpc
        else:
            return r_eff_arcsec
    elif unit == 'px':
        return r_eff_px
    else:
        raise ValueError("Unit must be 'px', 'arcsec', 'pkpc' or 'ckpc'")


def max_radius(fits_in, obj_mask, obj_id=1, unit='px', redshift=None,
cosmo=WMAP9):
    """Determines the maximum radial extent of a 2D/3D object from its centroid.

    Args:
        fits_in (HDU or HDUList): 2D or 3D flux-like data with header.
        obj_mask (numpy.ndarray): 2D or 3D data with labelled object regions.
        obj_id (int): Integer ID of the object to be measured.
        unit (str): Output unit of radius measurement.
            'px': pixels
            'arcsec': arcseconds
            'pkpc': proper kiloparsec
            'ckpc': comoving kiloparsec
        cosmology (FlatLambdaCDM): Cosmology to use, as one of the inbuilt
            ~astropy.cosmology.FlatLambdaCDM instances (default WMAP9)

    Returns:
        float: The maximum radius in the requested units
    """

    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data, hdu.header

    if obj_mask.shape != data.shape:
        raise ValueError("Object mask and data should match in dimensions.")

    #Get centroid and radius meshgrid centered on it in desired units
    centroid = centroid2d(fits_in, obj_mask, obj_id)
    rr_obj = coordinates.get_rgrid(fits_in, centroid[0], centroid[1],
        unit=unit
    )

    #Get 2D mask of object
    ndims = len(obj_mask.shape)
    bin_mask = obj_mask == obj_id
    if ndims == 3:
        obj2d = np.max(bin_mask, axis=0)
    elif ndims == 2:
        obj2d = bin_mask
    else:
        raise ValueError("Object mask must be 2D or 3D")

    #Find maximum value of R within object mask and return
    r_max = np.max(rr_obj[obj2d == 1])
    return r_max

def rms_radius(fits_in, obj_mask, obj_id=1, unit='px', redshift=None,
cosmo=WMAP9):
    """Determines the flux-weighted RMS radius of an object.

    Args:
        fits_in (HDU or HDUList): 2D or 3D flux-like data with header.
        obj_mask (numpy.ndarray): 2D or 3D data with labelled object regions.
        obj_id (int): Integer ID of the object to be measured.
        unit (str): Output unit of radius measurement.
            'px': pixels
            'arcsec': arcseconds
            'pkpc': proper kiloparsec
            'ckpc': comoving kiloparsec
        cosmology (FlatLambdaCDM): Cosmology to use, as one of the inbuilt
            ~astropy.cosmology.FlatLambdaCDM instances (default WMAP9)

    Returns:
        float: The flux-weighted RMS radius in the requested units
    """
    if 'kpc' in unit and redshift is None:
        raise ValueError("Redshift must be provided to convert to kiloparsecs.")

    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data, hdu.header

    if obj_mask.shape != data.shape:
        raise ValueError("Object mask and data should match in dimensions.")

    #Get centroid and radius meshgrid centered on it in desired units
    centroid = centroid2d(fits_in, obj_mask, obj_id)
    rr_obj = coordinates.get_rgrid(fits_in, centroid[0], centroid[1],
        unit=unit
    )

    #Get 2D mask of object
    ndims = len(obj_mask.shape)
    bin_mask = obj_mask == obj_id
    if ndims == 3:
        obj2d = np.max(bin_mask, axis=0)
        data2d = np.sum(data, axis=0)
    elif ndims == 2:
        obj2d = bin_mask
        data2d = data
    else:
        raise ValueError("Object mask must be 2D or 3D")

    #Calculate flux-weighed RMS radius and return
    weights2d = data2d * obj2d
    r_rms = np.sqrt(np.sum(weights2d * rr_obj**2) / np.sum(weights2d))

    return r_rms

def rms_velocity(fits_in):
    """Obtain the RMS velocity from a velocity map.

    Args:
        fits_in (HDU or HDUList): HDU or HDUList containing velocity map.

    Returns:
        float: The RMS velocity of the input.

    """
    hdu = utils.extractHDU(fits_in)
    vmap, header = hdu.data, hdu.header

    spx_mask = np.isnan(vmap) | np.isinf(vmap)
    velocities = vmap[~spx_mask].flatten()
    rms_vel = np.sqrt(np.sum(np.power(velocities, 2)))

    return rms_vel
