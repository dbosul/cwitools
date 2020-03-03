from astropy.io import fits
from astropy.wcs import WCS
from cwitools import subtraction
from cwitools.tests import test_data

import numpy as np
import os
import unittest

class CoordinatesTestCases(unittest.TestCase):


    def test_psf_sub_1d(self):
        test_fits = fits.open(test_data.icubes_path)
        ra, dec = test_data.ra, test_data.dec
        wcs = WCS(test_fits[0].header)
        x0, y0, w0 = wcs.all_world2pix(ra, dec, 4500, 0)
        sub, model = subtraction.psf_sub_1d(test_fits,
            (float(x0), float(y0)),
            fit_rad = 1.5,
            sub_rad = 5.5,
            slice_rad = 3,
        )
        test_fits.close()
        self.assertTrue(1)

    def test_psf_sub_2d(self):
        test_fits = fits.open(test_data.icubes_path)
        ra, dec = test_data.ra, test_data.dec
        wcs = WCS(test_fits[0].header)
        x0, y0, w0 = wcs.all_world2pix(ra, dec, 4500, 0)
        sub, model = subtraction.psf_sub_2d(test_fits,
            (float(x0), float(y0)),
            fit_rad = 1.5,
            sub_rad = 5.5
        )
        test_fits.close()
        self.assertTrue(1)

    def test_psf_sub_all(self):
        test_fits = fits.open(test_data.icubes_path)
        reg_path = test_data.reg_path
        ra, dec = test_data.ra, test_data.dec
        wcs = WCS(test_fits[0].header)
        x0, y0, w0 = wcs.all_world2pix(ra, dec, 4500, 0)
        sub, model = subtraction.psf_sub_all(test_fits,
            fit_rad = 1.5,
            sub_rad = 5.5,
            reg = reg_path,
            method = '2d'
        )
        test_fits.close()
        self.assertTrue(1)
if __name__ == '__main__':

    unittest.main()

    test_fits.close()
