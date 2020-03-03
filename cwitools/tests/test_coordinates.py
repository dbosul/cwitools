from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from cwitools import coordinates
from cwitools.tests import test_data

import numpy as np
import os
import unittest


class CoordinatesTestCases(unittest.TestCase):

    def test_get_wav_axis(self):
        test_fits = fits.open(test_data.coadd_path)
        header = test_fits[0].header
        wav0 = header["CRVAL3"] - header["CRPIX3"] * header["CD3_3"]
        wav_axis = coordinates.get_wav_axis(header)
        test_fits.close()
        self.assertEqual(wav_axis[0], wav0)


    def test_get_header1d(self):
        test_fits = fits.open(test_data.coadd_path)
        header1d = coordinates.get_header1d(test_fits[0].header)
        wcs1d = WCS(header1d)
        test_fits.close()
        self.assertEqual(wcs1d.naxis, 1)

    def test_get_header2d(self):
        test_fits = fits.open(test_data.coadd_path)
        header2d = coordinates.get_header2d(test_fits[0].header)
        wcs2d = WCS(header2d)
        test_fits.close()
        self.assertEqual(wcs2d.naxis, 2)

    def test_get_indices(self):
        test_fits = fits.open(test_data.coadd_path)
        header = test_fits[0].header
        wav0 = header["CRVAL3"] - header["CRPIX3"] * header["CD3_3"]
        wav_a, wav_b = wav0, wav0 + 10 * header["CD3_3"]
        a, b = coordinates.get_indices(wav_a, wav_b, header)
        test_fits.close()
        self.assertTrue((a == 0) & (b == 10))

    def test_get_pkpc_per_px(self):
        test_fits = fits.open(test_data.coadd_path)
        wcs = WCS(test_fits[0].header)
        as_per_px = proj_plane_pixel_scales(wcs)[1] * 3600.0
        pkpc_per_px = coordinates.get_pkpc_per_px(wcs, redshift=2.5)
        #Just assert that any value which is ballpark accurate is returned
        #Exact accuracy will vary depending on cosmology
        pkpc_per_as = pkpc_per_px / as_per_px
        test_fits.close()
        self.assertTrue(7.0 < pkpc_per_as < 9.0)

if __name__ == '__main__':

    unittest.main()

    test_fits.close()
