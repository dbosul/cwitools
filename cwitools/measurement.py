"""Tools for kinematic calculations."""
#Standard Imports

#Third-party Imports
import numpy as np
from astropy import units as u
from astropy.cosmology import WMAP9
from astropy.wcs import WCS

#Local Imports
from cwitools import utils, coordinates, extraction

def first_moment(x, y, y_var=None, get_err=False, method='basic', mu1_init=None, window_size=25,
                 window_min=10, window_step=1):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        y_var (np.array): Variance on y. Taken as var(y) if not provided.
        get_err (bool): Set to TRUE to return (moment, error) tuple
        method (str): The method to use for calculating the moment.
            'basic': Use all input x and y data.
            'clw': Use the closing-window method (O'Sullivan+20, FLASHES Survey)
        mu1_init (float): Initial estimate for mu1, required if using 'clw' method
            or restricting calculatio to a certain window size.
        window_size (float): The default window size for the moment calculation
            if using mu1_init, or the initial window size of using 'clw' method.
        window_min (float): The minimum window size if using 'clw' method.
        window_step (float): The decrement in window size each iteration if
            using the 'clw' method.

    Returns:
        float: The first moment in x.
        float: (if get_err == True) The estimated error on the first moment

    Example:

        To get the first moment of a spectrum 'spec' with wavelength axis 'wav'

        >>> mu1 = measurement.first_moment(wav, spec)

        To get the error on the measurement, provide the input variance using the y_var argument,
        available:

        >>> mu1, mu1_err = measurement.first_moment(wav, spec, y_var=spec_var)

        or, if you do not have error on the input flux, it can be (roughly) esitmated from the data:

        >>> mu1, mu1_err = measurement.first_moment(wav, spec, get_err=True)
    """

    #If variance given, assume they want error returned
    if y_var is not None:
        get_err = True

    #Estimate variance if no variance given
    y_var = np.var(y) if y_var is None else y_var

    #Basic and positive methods are mostly the same
    if method == 'basic':

        #If an initial value is given, use only the window around mu1_init
        if mu1_init is not None:
            usex = np.abs(x - mu1_init) <= window_size
            x = x[usex]
            y = y[usex]

        #Perform moment calculation as normal
        num = np.sum(x * y) #Numerator
        den = np.sum(y) #Denominator

        mu1 = num / den
        mu1_err = np.sqrt(np.sum(y_var * (den * x - num)**2)) / den**2

    #If iterative closing-window method is selected
    elif method == 'clw':

        #Start guess at center of array of no initial value given
        mu1 = x[int(len(x) / 2)] if mu1_init is None else mu1_init

        #Initialize window at maximum size
        window = window_size

        # Loop with decreasing window size until minimum is reached
        while window > window_min:

            #Get indices of values to use for this calculation
            use = (np.abs(x - mu1) < window/2) & (y > 0)

            usex = x[use]
            usey = y[use]

            #Update moment calculation using current window
            num = np.sum(usex * usey) #Numerator
            den = np.sum(usey) #Denominator
            mu1 = num / den
            mu1_err = np.sqrt(np.sum(y_var * (den * x - num)**2)) / den**2

            #Update window size
            window -= window_step

    if get_err:
        return mu1, mu1_err

    return mu1


def second_moment(x, y, mu1=None, y_var=None, get_err=False):
    """Calculate first moment.

    Args:
        x (np.array): Input coordinate values (e.g. wavelength).
        y (np.array): Input weights (i.e. intensity)
        mu1 (float): The first moment. Calculated if not provided.
        y_var (np.array): Variance on y. Taken as var(y) if not provided.
        get_err (bool): Set to TRUE to return (moment, error), tuple

    Returns:
        float: The second moment in x.
        float: The error on the second moment (if get_err == True)

    Example:

        To get the first moment of a spectrum 'spec' with wavelength axis 'wav'

        >>> mu2 = measurement.second_moment(wav, spec)

        If you want to use a custom value for the first moment, provide it with 'mu1='

        >>> mu2, mu2_err = measurement.second_moment(wav, spec, mu1=4240)

        As with first_moment, error can be estimated if variance is provided or roughly estimated
        from the input data itself.

    """

    #If variance given, assume they want error returned
    if y_var is not None:
        get_err = True

    #Calculate mu1 if not given
    mu1 = np.sum(x * y) / np.sum(y) if mu1 is None else mu1

    #Numerator and denominator of mu2 calculation
    num = np.sum(np.power(x - mu1, 2) * y)
    den = np.sum(y)

    #Second moment
    mu2 = np.sqrt(num / den)

    #Calculate uncertainty if requested
    if get_err:

        #Estimate if no variance given
        y_var = np.var(y) if y_var is None else y_var

        #Squared residuals array
        rsquared = np.power(x - mu1, 2)

        #Numerator and denominator terms in summation under square root
        sqrt_num = np.power(rsquared * den - num, 2) * y_var
        sqrt_den = np.power(den, 4)

        #Calculate in full now
        mu2_err = np.sqrt(np.sum(sqrt_num / sqrt_den)) / (2 * mu2)

        return mu2, mu2_err

    return mu2



def luminosity(fits_in, redshift=None, mask=None, cosmo=WMAP9, var_data=None):
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
        var_data (numpy.ndarray): Array of same dimensions as data and mask,
            containing variance estimates. Used to propagate error on luminosity.

    Returns:
        float: The integrated luminosity of the source in erg/s.
        float: The error on the luminosity calculation.
        
    """

    #Extract data and header
    hdu = utils.extract_hdu(fits_in)
    data, header = hdu.data.copy(), hdu.header.copy()

    #Replace mask with array of all 1s if none given
    mask = np.ones_like(data, dtype=bool) if mask is None else mask
    usevar = var_data is not None

    #Check dimensions of mask match (only matters for user-provided masks)
    if mask.shape != data.shape:
        raise ValueError("mask must match dimensions of input data.")

    if usevar:
        var = var_data.copy()
        mask[np.isnan(var) | np.isinf(var)] = 0

    #Apply mask
    mask[np.isnan(data)| np.isinf(data)] = 0

    data[mask == 0] = 0
    if usevar:
        var[mask == 0] = 0


    #If the input data is 3D, convert to SB map first
    ndims = len(data.shape)
    if ndims in [1, 3]:
        #input units are FLAM erg/s/cm2/angstrom
        px_size_ang = coordinates.get_pxsize_angstrom(header)
        unit_conv = px_size_ang

    elif ndims == 2:
        #input units are erg/s/cm2/arcsec
        px_area_arcsec = coordinates.get_pxarea_arcsec(header)
        unit_conv = px_area_arcsec #erg/s/cm2

    else:
        raise ValueError("Input data must be 1D, 2D or 3D.")

    flux_total = np.sum(data) * unit_conv #erg/s/cm2

    if usevar:
        flux_var_total = np.sum(var) * (unit_conv**2)

        if "COV_ALPH" in header:

            alpha = header["COV_ALPH"]
            norm = header["COV_NORM"]
            thresh = header["COV_THRE"]
            nmask = np.count_nonzero(mask)

            if nmask > thresh:
                err_ratio = norm * (1 + alpha * np.log(thresh))
            else:
                err_ratio = norm * (1 + alpha * np.log(nmask))

            flux_var_total *= (err_ratio**2)

    #Calculate distance to source in cm
    lum_dist = cosmo.luminosity_distance(redshift).to(u.cm).value
    flux_to_lum = 4 * np.pi * (lum_dist**2) #units of 'cm2'

    #Convert total flux to luminosity and return
    lum_tot = flux_total * flux_to_lum

    if usevar:
        var_tot = flux_var_total * (flux_to_lum**2)
    else:
        var_tot = np.var(data) * np.sum(mask) * (flux_to_lum**2)

    lum_err = np.sqrt(var_tot)

    return lum_tot, lum_err


def moment2d(x_grid, y_grid, p, q, fxy):
    """Calculate image moment of order p in x and q in y.

    Mpq = SUM(x_grid^p * y_grid^q * fxy) / SUM(fxy)

    Args:
        x_grid (numpy.ndarray): 2D meshgrid of x-values
        y_grid (numpy.ndarray): 2D meshgrid of y-values
        p (int): Moment order in x.
        q (int): Moment order in y.
        fxy (numpy.ndarray): Array of weights representing the 2D distribution
            being measured; e.g. flux or surface brightness.

    Returns:
        float: The value of the requested image moment.

    """
    return np.sum(np.power(x_grid, p) * np.power(y_grid, q) * fxy) / np.sum(fxy)

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
    y_grid, x_grid = np.meshgrid(np.arange(vel_map.shape[1]), np.arange(vel_map.shape[0]))

    #Calculate relevant moments for specific angular momentum calculation
    x_cen = moment2d(x_grid, y_grid, 1, 0, flux_weights)
    y_cen = moment2d(x_grid, y_grid, 0, 1, flux_weights)

    #Get meshgrid of distance from x, ycentroid
    r_grid = np.sqrt((x_grid - x_cen)**2 + (y_grid - y_cen)**2)

    #Calculate and return
    return np.sum(flux_weights * r_grid * np.abs(vel_map)) / np.sum(flux_weights)

def asymmetry(sb_map, obj_mask=None):
    """Calculate the spatial asymmetry (alpha) of a 2D or 3D object.

    Args:
        sb_map (numpy.ndarray): The 2D surface brightness map.
        obj_mask (numpy.ndarray): A 2D mask delineating the object.

    Returns:
        float: The calculated asymmetry parameter (alpha), represening the minor/major axis ratio.
    """

    #Generate blank mask if none given
    if obj_mask is None:
        obj_mask = np.ones_like(sb_map)

    #Apply mask
    sb_map[obj_mask == 0] = 0

    #Get spatial meshgrids
    x_grid, y_grid = np.indices(sb_map.shape)

    #Get x and y centroids
    x_cen = moment2d(x_grid, y_grid, 1, 0, sb_map)
    y_cen = moment2d(x_grid, y_grid, 0, 1, sb_map)

    #Get x and y meshgrids centered on object
    dx_obj = x_grid - x_cen
    dy_obj = y_grid - y_cen

    #
    # Measure moments as in ArrigoniBattaia et al. 2019
    #
    moment20 = moment2d(dx_obj, dy_obj, 2, 0, sb_map)
    moment02 = moment2d(dx_obj, dy_obj, 0, 2, sb_map)
    moment11 = moment2d(dx_obj, dy_obj, 1, 1, sb_map)

    stokes_q = moment20  - moment02
    stokes_u = 2 * moment11

    return (1 - np.sqrt(stokes_q**2 + stokes_u**2))/(1 + np.sqrt(stokes_q**2 + stokes_u**2))


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
    return np.sqrt(1 - alpha**2)

def major_pa(fits_in, obj_mask=None, obj_id=1, var_data=None, coords='image'):
    """Calculate the position angle of the major axis of an extended object.

    Args:
        fits_in (HDU or HDUList): 2D or 3D flux-like data.
        obj_mask (numpy.ndarray): 2D or 3D data with labelled object regions.
        obj_id (int or list): Integer or list of integers of object IDs to use.
        var_data (numpy.ndarray): Variance image.
        coords (str): Desired coordinate system for the output.
            'image': Counterclockwise from up. Non-square pixel sampling is NOT
                considered.
            'wcs': East to North. Non-square pixel sampling is considered.

    Returns:
        float: The position angle in degrees of the object's major axis.
            The output angles are restricted in -90 to +90.
        (float: Error of the position angle if variance image is provided.)
    """
    hdu = utils.extract_hdu(fits_in)
    data, header = hdu.data.copy(), hdu.header.copy()

    if obj_mask is not None:
        bin_mask = extraction.obj2binary(obj_mask, obj_id)

        #Remove non-object regions
        data[bin_mask == 0] = 0

    #Get centroid
    x_cen, y_cen = centroid2d(fits_in, obj_mask, obj_id, coords='image')

    #Grid
    x_grid, y_grid = np.indices(fits_in.shape)

    #2nd moments
    x2_mean = moment2d(x_grid, y_grid, 2, 0, data)
    y2_mean = moment2d(x_grid, y_grid, 0, 2, data)
    xy_mean = moment2d(x_grid, y_grid, 1, 1, data)

    #Angle
    if x2_mean!=y2_mean:
        tan = 2 * xy_mean / (x2_mean - y2_mean)
        theta_mean=np.arctan(2*xy_mean/(x2_mean-y2_mean))/2
    else:
        theta_mean=np.pi/4

    #2nd Moments along the major and minor axes
    x_theta_2 = (np.cos(theta_mean)**2 * x2_mean +
                 np.sin(theta_mean)**2 * y2_mean +
                 2 * np.cos(theta_mean) * np.sin(theta_mean) * xy_mean)
    y_theta_2 = (np.sin(theta_mean)**2 * x2_mean +
                 np.cos(theta_mean)**2 * y2_mean -
                 2 * np.cos(theta_mean) * np.sin(theta_mean) * xy_mean)

    #Determine which is the major axis
    if x_theta_2 > y_theta_2:
        theta_mean = theta_mean + np.pi / 2.

    #Error
    if var_data is not None:
        sig = np.sqrt(var_data)

        sig_x2 = (np.sqrt(np.sum(data[bin_mask]**2 * x_grid[bin_mask]**4)) /
                  np.sum(data[bin_mask]))
        sig_y2 = (np.sqrt(np.sum(data[bin_mask]**2 * y_grid[bin_mask]**4)) /
                  np.sum(data[bin_mask]))
        sig_xy = (np.sqrt(np.sum(data[bin_mask]**2 * x_grid[bin_mask]**2 *
                  y_grid[bin_mask]**2)) / np.sum(data[bin_mask]))
        sig_tan = (2 / np.abs(x2_mean - y2_mean) * np.sqrt(sig_xy**2 +
                  xy_mean**2 * (sig_x2**2 + sig_y2**2)))
        sig_theta = sig_tan / (2 * (1 + tan**2))


    #Coordinates
    if coords == 'image':
        final_pa, final_pa_err = np.degrees(theta_mean), np.degrees(sig_theta)
    elif coords == 'wcs':
        #Non-square pixels
        dx = np.sqrt(header['CD1_1']**2 + header['CD2_1']**2)
        dy = np.sqrt(header['CD1_2']**2 + header['CD2_2']**2)
        theta_mean = theta_mean * dx / dy
        sig_theta = sig_theta * dx / dy

        #Frame PA
        theta_frame = np.arctan(header['CD1_2'] / header['CD2_2'])
        theta_mean = theta_mean + theta_frame

        final_pa, final_pa_err = np.degrees(theta_mean), np.degrees(sig_theta)

    else:
        raise ValueError("coords argument must be 'image' or 'wcs'")

    #Remove periodicity
    while theta_mean >= np.pi/2.:
        theta_mean = theta_mean - np.pi
    while theta_mean < -np.pi/2:
        theta_mean = theta_mean + np.pi

    #Return
    if var_data is not None:
        return np.degrees(theta_mean), np.degrees(sig_theta)
    else:
        return np.degrees(theta_mean)






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
    hdu = utils.extract_hdu(fits_in)
    data, header = hdu.data.copy(), hdu.header.copy()

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
    x_grid, y_grid = np.indices(weights2d.shape)

    #Calculate image moments
    x_cen = moment2d(x_grid, y_grid, 1, 0, weights2d)
    y_cen = moment2d(x_grid, y_grid, 0, 1, weights2d)

    if coords == 'image':
        return x_cen, y_cen

    if coords == 'radec':
        if ndims == 3:
            wcs2d = WCS(coordinates.get_header2d(header))
        else:
            wcs2d = WCS(header)
        ra, dec = wcs2d.all_pix2world(x_cen, y_cen, 0)
        return ra, dec

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
    obj_hdu = utils.extract_hdu(obj_in)
    obj_mask, header = obj_hdu.data.copy(), obj_hdu.header.copy()

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

    if unit == 'px':
        return nspaxels

    raise ValueError("Unit must be 'px2' or 'arcsec2'")

def eff_radius(obj_in, obj_id=1, unit='px', redshift=None, cosmo=WMAP9):
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

    obj_hdu = utils.extract_hdu(obj_in)
    obj_mask, header = obj_hdu.data.copy(), obj_hdu.header.copy()

    bin_mask = extraction.obj2binary(obj_mask, obj_id)
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
            return r_eff_pkpc.value

        if unit == 'ckpc':
            ckpc_per_arcsec = cosmo.kpc_comoving_per_arcmin(redshift) / 60.0
            r_eff_ckpc = r_eff_arcsec * ckpc_per_arcsec
            return r_eff_ckpc.value

        return r_eff_arcsec

    if unit == 'px':
        return r_eff_px

    raise ValueError("Unit must be 'px', 'arcsec', 'pkpc' or 'ckpc'")


def max_radius(fits_in, obj_mask, obj_id=1, unit='px', redshift=None, cosmo=WMAP9, pos=None):
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

    hdu = utils.extract_hdu(fits_in)
    data = hdu.data.copy()

    if obj_mask.shape != data.shape:
        raise ValueError("Object mask and data should match in dimensions.")

    #Get centroid and radius meshgrid centered on it in desired units
    if pos is None:
        centroid = centroid2d(fits_in, obj_mask, obj_id)
        rr_obj = coordinates.get_rgrid(fits_in, centroid, unit=unit, redshift=redshift)
    else:
        rr_obj = coordinates.get_rgrid(fits_in, pos, unit=unit, redshift=redshift, pos_type='radec')

    #Get 2D mask of object
    ndims = len(obj_mask.shape)
    bin_mask = extraction.obj2binary(obj_mask, obj_id)
    if ndims == 3:
        obj2d = np.max(bin_mask, axis=0)
    elif ndims == 2:
        obj2d = bin_mask
    else:
        raise ValueError("Object mask must be 2D or 3D")

    #Find maximum value of R within object mask and return
    return np.max(rr_obj[obj2d == 1])

def rms_radius(fits_in, obj_mask, obj_id=1, unit='px', redshift=None, cosmo=WMAP9, pos=None):
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

    hdu = utils.extract_hdu(fits_in)
    data = hdu.data.copy()

    if obj_mask.shape != data.shape:
        raise ValueError("Object mask and data should match in dimensions.")

    #Get centroid and radius meshgrid centered on it in desired units
    if pos is None:
        centroid = centroid2d(fits_in, obj_mask, obj_id)
        rr_obj = coordinates.get_rgrid(fits_in, centroid, unit=unit, redshift=redshift)
    else:
        rr_obj = coordinates.get_rgrid(fits_in, pos, unit=unit, redshift=redshift, pos_type='radec')

    #Get 2D mask of object
    ndims = len(obj_mask.shape)
    bin_mask = extraction.obj2binary(obj_mask, obj_id)
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
    hdu = utils.extract_hdu(fits_in)
    vmap = hdu.data.copy()

    spx_mask = np.isnan(vmap) | np.isinf(vmap)
    velocities = vmap[~spx_mask].flatten()
    rms_vel = np.sqrt(np.sum(np.power(velocities, 2)))

    return rms_vel
