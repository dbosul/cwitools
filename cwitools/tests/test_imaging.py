from astropy.io import fits
from astropy.wcs import WCS
from cwitools import coordinates
from cwitools import imaging
from cwitools.tests import test_data

import numpy as np
import os
import unittest


class CoordinatesTestCases(unittest.TestCase):

    def test_get_cutout(self):
        test_fits = fits.open(test_data.coadd_path)
        test_ra, test_dec, test_z = test_data.ra, test_data.dec, test_data.z
        cutout_fits = imaging.get_cutout(test_fits, test_ra, test_dec, 250,
            z=test_z
        )
        test_fits.close()
        self.assertTrue(1)

    def test_smooth_nd(self):
        test_fits = fits.open(test_data.coadd_path)
        cube = test_fits[0].data
        smooth = imaging.smooth_nd(cube, 2.5)
        test_fits.close()
        self.assertTrue(1)

    def test_get_mask(self):
        test_fits = fits.open(test_data.coadd_path)
        test_reg = test_data.reg_path
        image = np.sum(test_fits[0].data, axis=0)
        h2d = coordinates.get_header2d(test_fits[0].header)
        mask = imaging.get_mask(image, h2d, reg=test_reg)
        test_fits.close()
        self.assertTrue(1)

    def test_get_pseudo_nb(self):
        test_fits = fits.open(test_data.coadd_path)
        wcs = WCS(test_fits[0].header)
        x0, y0, w0 = wcs.all_world2pix(test_data.ra, test_data.dec, 4350, 0)
        pnb = imaging.get_pseudo_nb(test_fits, 4350, 20,
            pos=(x0, y0),
            sub_r=5
         )
        test_fits.close()
        self.assertTrue(1)

    def test_slice_fix(self):
        test_fits = fits.open(test_data.coadd_path)
        img = np.sum(test_fits[0].data, axis=0)
        test_fits.close()
        img_fixed = imaging.slice_fix(img)
        self.assertTrue(1)

if __name__ == '__main__':

    unittest.main()

    test_fits.close()
