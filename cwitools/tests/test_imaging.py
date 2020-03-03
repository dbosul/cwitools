from astropy.io import fits
from cwitools.analysis import imaging
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

if __name__ == '__main__':

    unittest.main()

    test_fits.close()
