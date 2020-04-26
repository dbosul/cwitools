from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from cwitools.coordinates import *
from cwitools.tests import test_data

import numpy as np
import os
import unittest


class CoordinatesTestCases(unittest.TestCase):

    def test_getflam2sb(self):
        header = fits.getheader(test_data.coadd_path)
        res = get_flam2sb(header)
        self.assertTrue(type(res) is float)

    def test_get_pxsize_angstrom(self):
        header = fits.getheader(test_data.coadd_path)
        res = get_pxsize_angstrom(header)
        self.assertTrue(type(res) is float)

    def test_get_pxarea_arcsec(self):
        header = fits.getheader(test_data.coadd_path)
        res = get_pxarea_arcsec(header)
        self.assertTrue(type(res) is float)

    def test_get_rgrid(self):
        test_fits = fits.open(test_data.coadd_path)
        data = test_fits[0].data
        center = data.shape[0] / 2, data.shape[1] / 2
        #Call all versions of function to test different units
        res_px = get_rgrid(test_fits, [x0,y0], unit='px')
        res_as = get_rgrid(test_fits, [x0,y0], unit='arcsec')
        res_pk = get_rgrid(test_fits, [x0,y0], unit='pkpc', redshift=2.5)
        res_ck = get_rgrid(test_fits, [x0,y0], unit='ckpc', redshift=2.5)
        #Just make sure calls all completed
        self.assertTrue(True)

    def test_get_header1d(self):
        test_fits = fits.open(test_data.coadd_path)
        header1d = get_header1d(test_fits[0].header)
        wcs1d = WCS(header1d)
        test_fits.close()
        self.assertEqual(wcs1d.naxis, 1)

    def test_get_header2d(self):
        test_fits = fits.open(test_data.coadd_path)
        header2d = get_header2d(test_fits[0].header)
        wcs2d = WCS(header2d)
        test_fits.close()
        self.assertEqual(wcs2d.naxis, 2)

    def test_get_kpc_per_px(self):
        header = fits.getheader(test_data.coadd_path)
        pkpc = get_kpc_per_px(test_fits[0].header,
            type='proper',
            redshift=2.
        )
        ckpc = get_kpc_per_px(test_fits[0].header,
            type='comoving',
            redshift=2.
        )
        typetest = (type(pkpc) is float) and (type(ckpc) is float)
        self.assertTrue(typetest)

    def test_get_indices(self):
        header = fits.getheader(test_data.coadd_path)
        wav0 = header["CRVAL3"] - header["CRPIX3"] * header["CD3_3"]
        wav_a, wav_b = wav0, wav0 + 10 * header["CD3_3"]
        a, b = get_indices(wav_a, wav_b, header)
        test_fits.close()
        self.assertTrue((a == 0) & (b == 10))
        
    def test_get_wav_axis(self):
        test_fits = fits.open(test_data.coadd_path)
        header = test_fits[0].header
        wav0 = header["CRVAL3"] - header["CRPIX3"] * header["CD3_3"]
        wav_axis = get_wav_axis(header)
        test_fits.close()
        self.assertEqual(wav_axis[0], wav0)






if __name__ == '__main__':

    unittest.main()

    test_fits.close()
