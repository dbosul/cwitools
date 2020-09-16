"""Reduction tools related to unit conversions and corrections."""

#Standard Imports

#Third-party Imports
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
import astropy.coordinates
import astropy.stats
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

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        mask (bool): Set if the cube is a mask cube. This only works for
            resampled cubes.
        return_vcorr (bool): If set, return the correction velocity (in km/s)
            as well.
        resample (bool): Resample the cube to the original wavelength grid?
        vcorr (float): Use a different correction velocity.
        barycentric (bool): Use barycentric correction instead of helocentric.

    Returns:
        HDU / HDUList*: Trimmed FITS object with updated header.
        vcorr (float): (if vcorr is True) Correction velocity in km/s.
        *Return type matches type of fits_in argument.

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
