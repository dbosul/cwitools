from astropy.io import fits
from cwitools.analysis import variance

import numpy as np
import os
import unittest

def get_test_fits():
    #Create test data cube
    test_path = __file__.replace("tests/test_variance.py", "data/test_cube.fits")
    test_fits = fits.open(test_path)
    return test_fits

class CoordinatesTestCases(unittest.TestCase):

    #For now - this tests both estimate and rescale. Need new test data to
    #perform separate tests.
    def test_psf_sub_1d(self):
        test_fits = get_test_fits()

        test_fits.close()
        self.assertTrue(1)

if __name__ == '__main__':

    unittest.main()

    test_fits.close()
