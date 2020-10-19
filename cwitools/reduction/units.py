"""Reduction tools related to unit conversions and corrections."""

#Standard Imports

#Third-party Imports
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
import astropy.coordinates
import astropy.stats
import astropy.units as u
import numpy as np

#Local Imports
from cwitools import coordinates, utils

def air2vac(fits_in, mask=False):
    """Covert wavelengths in a cube from standard air to vacuum.

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        mask (bool): Set if the cube is a mask cube.

    Returns:
        HDU / HDUList*: Trimmed FITS object with updated header.
        *Return type matches type of fits_in argument.

    """

    hdu = utils.extract_hdu(fits_in)
    hdu = hdu.copy()
    cube = np.nan_to_num(hdu.data, nan=0, posinf=0, neginf=0)
    hdr = hdu.header

    if hdr['CTYPE3'] == 'WAVE':
        utils.output("\tFITS already in vacuum wavelength.\n")
        return fits_in

    wave_air = coordinates.get_wav_axis(hdr)
    wave_vac = pyasl.airtovac2(wave_air)

    # resample to uniform grid
    cube_new = np.zeros_like(cube)
    for i in range(cube.shape[2]):
        for j in range(cube.shape[1]):

            spec0 = cube[:, j, i]
            if not mask:
                f_cubic = interp1d(
                    wave_vac,
                    spec0,
                    kind='cubic',
                    fill_value='extrapolate'
                    )
                spec_new = f_cubic(wave_air)
            else:
                f_pre = interp1d(
                    wave_vac,
                    spec0,
                    kind='previous',
                    bounds_error=False,
                    fill_value=128
                    )
                spec_pre = f_pre(wave_air)

                f_nex = interp1d(
                    wave_vac,
                    spec0,
                    kind='next',
                    bounds_error=False,
                    fill_value=128
                    )
                spec_nex = f_nex(wave_air)

                spec_new = np.zeros_like(spec0)
                for k in range(spec0.shape[0]):
                    spec_new[k] = max(spec_pre[k], spec_nex[k])

            cube_new[:, j, i] = spec_new

    hdr['CTYPE3'] = 'WAVE'
    hdu_new = utils.match_hdu_type(fits_in, cube_new, hdr)

    return hdu_new


def heliocentric(fits_in, mask=False, return_vcorr=False, resample=True, vcorr=None,
                 barycentric=False):
    """Apply heliocentric correction to the cubes. 
    *Note that this only works for KCWI data because the location of the Keck 
    Observatory is hard-coded in the function.*

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        mask (bool): Set if the cube is a mask cube. This only works for
            resampled cubes.
        return_vcorr (bool): If set, return the correction velocity (in km/s)
            as well.
        resample (bool): Resample the cube to the original wavelength grid?
        vcorr (float): Use a different correction velocity.
        barycentric (bool): Use barycentric correction instead of heliocentric.

    Returns:
        HDU / HDUList*: Trimmed FITS object with updated header.
        vcorr (float): (if vcorr is True) Correction velocity in km/s.
        *Return type matches type of fits_in argument.
        
    Examples: 
        
        To apply heliocentric correction,
        
        >>> hdu_new = heliocentric(hdu_old)
        
        However, this resamples the wavelengths back to the original grid. To
        use the new grid without resampling the data,
        
        >>> hdu_new = heliocentric(hdu_old, resample=False)

    """

    hdu = utils.extract_hdu(fits_in)
    hdu = hdu.copy()
    cube = np.nan_to_num(hdu.data, nan=0, posinf=0, neginf=0)
    hdr = hdu.header

    v_old = 0.
    if 'VCORR' in hdr:
        v_old = hdr['VCORR']
        utils.output("\tRolling back the existing correction with:\n")
        utils.output("\t\tVcorr = %.2f km/s.\n" % (v_old))

    if vcorr is None:
        targ = astropy.coordinates.SkyCoord(
            hdr['TARGRA'],
            hdr['TARGDEC'],
            unit='deg',
            obstime=hdr['DATE-BEG']
            )
        keck = astropy.coordinates.EarthLocation.of_site('Keck Observatory')
        if barycentric:
            vcorr = targ.radial_velocity_correction(kind='barycentric', location=keck)
        else:
            vcorr = targ.radial_velocity_correction(kind='heliocentric', location=keck)
        vcorr = vcorr.to('km/s').value

    utils.output("\tHelio/Barycentric correction:\n")
    utils.output("\t\tVcorr = %.2f km/s.\n" % (vcorr))

    v_tot = vcorr-v_old

    if not resample:

        hdr['CRVAL3'] = hdr['CRVAL3'] * (1 + v_tot / 2.99792458e5)
        hdr['CD3_3'] = hdr['CD3_3'] * (1 + v_tot / 2.99792458e5)
        hdr['VCORR'] = vcorr
        hdu_new = utils.match_hdu_type(fits_in, cube, hdr)
        if not return_vcorr:
            return hdu_new
        return hdu_new, vcorr

    wav_old = coordinates.get_wav_axis(hdr)
    wav_hel = wav_old * (1 + v_tot / 2.99792458e5)

    # resample to uniform grid
    cube_new = np.zeros_like(cube)
    for i in range(cube.shape[2]):
        for j in range(cube.shape[1]):

            spc0 = cube[:, j, i]
            if not mask:
                f_cubic = interp1d(wav_hel, spc0, kind='cubic', fill_value='extrapolate')
                spec_new = f_cubic(wav_old)

            else:
                f_pre = interp1d(wav_hel, spc0, kind='previous', bounds_error=False, fill_value=128)
                spec_pre = f_pre(wav_old)
                f_nex = interp1d(wav_hel, spc0, kind='next', bounds_error=False, fill_value=128)
                spec_nex = f_nex(wav_old)

                spec_new = np.zeros_like(spc0)
                for k in range(spc0.shape[0]):
                    spec_new[k] = max(spec_pre[k], spec_nex[k])

            cube_new[:, j, i] = spec_new

    hdr['VCORR'] = vcorr
    hdu_new = utils.match_hdu_type(fits_in, cube_new, hdr)

    if not return_vcorr:
        return hdu_new

    return hdu_new, vcorr

def bunit_to_sb(header):
    """Get the conversion factor and new unit string to convert from FLAM to SB units.

    Args:
        header (astropy FITS Header): The Header of the input HDU. Input units must be:
            'FLAM'/'FLAM16' or similar (e.g. 'FLAM18')
            OR
            a string parseable by astropy.units.Unit(), made up of 'erg', 's', 'cm', 'arcsec',
            'angstrom' and a coefficient (e.g. "erg s-1 cm-2 arcsec-2")

    Returns:
        float: The coefficient to multiply the data by for this conversion
        str: The new BUNIT header value

    Example:
        To get the conversion from an input HDU with units of erg/s/cm2/angstrom to SB units
        (i.e. erg/s/cm2/arcsec), we do:

        >>> coeff, new_bunit = bunit_to_sb(hdu.header)

        Then to apply the conversion:

        >>> hdu.data *= coeff
        >>> hdu.header["BUNIT"] = new_bunit
    """
    bunit_in = get_bunit(header)

    #Initiliaze coeff and bunit as 1 and input
    coeff = 1
    bunit_out = bunit_in

    #Multiply by pixel size if units are per area
    if "arcsec-2" not in bunit_in:
        coeff /= coordinates.get_pxarea_arcsec(header)
        bunit_out = multiply_bunit(bunit_out, "arcsec-2")

    #Divide by binsize in Angstrom if not already per Angstrom
    if "angstrom-1" in bunit_in:
        coeff *= coordinates.get_pxsize_angstrom(header)
        bunit_out = multiply_bunit(bunit_out, "angstrom")

    return coeff, bunit_out

def bunit_to_flam(header):
    """Get the conversion factor and new unit string to convert from SB to FLAM units.

    Args:
        header (astropy FITS Header): The Header of the input HDU. Input units must be:
            'FLAM'/'FLAM16' or similar (e.g. 'FLAM18')
            OR
            a string parseable by astropy.units.Unit(), made up of 'erg', 's', 'cm', 'arcsec',
            'angstrom' and a coefficient (e.g. "erg s-1 cm-2 arcsec-2")

    Returns:
        float: The coefficient to multiply the data by for this conversion
        str: The new BUNIT header value

    Example:
        To get the conversion from an input HDU with units of erg/s/cm2/arcsec2 to FLAM
        units (i.e. erg/s/cm2/angstrom), we do:

        >>> coeff, new_bunit = bunit_to_flam(hdu.header)

        Then to apply the conversion:

        >>> hdu.data *= coeff
        >>> hdu.header["BUNIT"] = new_bunit
    """
    bunit_in = get_bunit(header)

    #Initiliaze coeff and bunit as 1 and input
    coeff = 1
    bunit_out = bunit_in

    #Multiply by pixel size if units are per area
    if "arcsec-2" in bunit_in:
        coeff *= coordinates.get_pxarea_arcsec(header)
        bunit_out = multiply_bunit(bunit_out, "arcsec2")

    #Divide by binsize in Angstrom if not already per Angstrom
    if "angstrom-1" not in bunit_in:
        coeff /= coordinates.get_pxsize_angstrom(header)
        bunit_out = multiply_bunit(bunit_out, "angstrom-1")

    return coeff, bunit_out

def bunit_to_dict(bunit_str):
    """Convert BUNIT string to a dictionary of 'unit:power' key:value pairs (e.g. cm^2 -> {'cm':2})

    Args:
        st (str): The input BUNIT string

    Returns:
        dict: The output dictionary of {unit:power} e.g. cm^2 = {'cm':2}

    """
    numchar = [str(i) for i in range(10)]
    numchar.append('+')
    numchar.append('-')
    dictout = {}

    st_list = bunit_str.split()
    for st_element in st_list:
        flag = 0
        i = 0
        for i, char in enumerate(st_element):
            if char in numchar:
                flag = 1
                break
        if i == 0:
            key = st_element
            power_st = '1'
        elif flag == 0:
            key = st_element
            power_st = '1'
        else:
            key = st_element[0:i]
            power_st = st_element[i:]

        dictout[key] = float(power_st)

    return dictout

def get_bunit(hdr):
    """"Get BUNIT string that meets FITS standard."""
    bunit = multiply_bunit(hdr['BUNIT'])
    return bunit

def multiply_bunit(bunit, multi=''):
    """Unit conversions and multiplications.

    Args:
        bunit (str): The BUNIT Header string, can be the following KCWI/PCWI defaults:
            'electrons' or 'variance': non-flux calibrated units
            'FLAM': meaning erg/s/cm2/angstrom
            'FLAM**2': flux variance units
            'FLAMXX': meaning 10^(-XX) erg/s/cm2/angstrom (e.g. FLAM16)
            'FLAMXX**2': variance for the above units
            OR
            any unit string composed only of the units listed below for 'mul_units' and a
            coefficient. Format must be parseable by astropy.units.Unit()

        multi (str): The units to multiply the input units by, limited to the following:
            - a coefficient (e.g. '1e-16')
            - 'erg' for energy
            - 's' for time
            - 'cm' for area
            - 'angstrom' for wavelength
            - 'arcsec' for angular area
            Note: 'A' is not 'Angstrom for astropy.units, so use 'angstrom' instead.

    Returns:
        str: The updated BUNIT string

    Example:

        To convert the default flux units of a KCWI cube (FLAM16) to erg/s/cm2/arcsec2:

        >>> bunit_flam16 = hdu.header["BUNIT"] #Input unit is 1e-16 erg/s/cm2/angstrom
        >>> bunit_sb = multiply_bunit(bunit_flam16, 'arcsec^-2 angstrom', coeff=1e16)

    """

    # Electrons
    electron_power = 0.
    if 'electrons' in bunit:
        bunit = bunit.replace('electrons', '1')
        electron_power = 1

    elif 'variance' in bunit:
        bunit = bunit.replace('variance', '1')
        electron_power = 2

    # unconventional expressions
    elif 'FLAM' in bunit:

        addpower = 1
        if '**2' in bunit:
            addpower = 2
            bunit = bunit.replace('**2', '')
        power = 0 if bunit == 'FLAM' else float(bunit.replace('FLAM', ''))

        v_0 = u.erg / u.s / u.cm**2 / u.angstrom * 10**(-power)
        v_0 = v_0**addpower

    else:
        v_0 = u.Unit(bunit)

    if isinstance(multi, str):
        multi = u.Unit(multi)
    else:
        multi = multi

    vout = (v_0 * multi)

    # Convert to quantity
    if isinstance(vout, (u.core.Unit, u.core.CompositeUnit)):
        vout = u.Quantity(1, vout)

    stout = "{0.value:.0e} {0.unit:FITS}".format(vout)
    stout = stout.replace('1e+00 ', '')
    stout = stout.replace('10**', '1e')

    # electrons
    if electron_power > 0:
        stout = stout +' electrons'+'{0:.0f}'.format(electron_power)+' '

    # sort
    def unit_key(str_in):
        """Docstring TBC"""
        flag = 5
        if str_in[0] in [str(i) for i in np.arange(10)]:
            flag = 0
        elif 'erg' in str_in or 'electrons' in str_in:
            flag = 1
        elif str_in[0] == 's':
            flag = 2
        elif 'cm' in str_in:
            flag = 3
        elif 'arcsec' in str_in:
            flag = 4
        return flag

    st_list = stout.split()
    st_list.sort(key=unit_key)
    stout = ' '.join(st_list)

    return stout
