from astropy.io import fits
from cwitools.analysis import imaging

import numpy as np
import os
import unittest

def get_test_fits():
    #Create test data cube
    test_path = __file__.replace("tests/test_imaging.py", "data/test_cube.fits")
    test_fits = fits.open(test_path)
    return test_fits

class CoordinatesTestCases(unittest.TestCase):

    def test_get_cutout(self):
        test_fits = get_test_fits()
        test_ra, test_dec = 168.218550543, 015.356529219
        test_z = 2.790
        cutout_fits = imaging.get_cutout(test_fits, test_ra, test_dec, 250,
            z=test_z
        )
        test_fits.close()
        self.assertTrue(1)

if __name__ == '__main__':

    unittest.main()

    test_fits.close()
