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
    def test_estimate_variance(self):
        test_fits = get_test_fits()
        var_cube = variance.estimate_variance(test_fits)
        var_np = np.var(test_fits[0].data)
        var_est = np.mean(var_cube)
        ratio = var_est / var_np
        test_fits.close()
        self.assertTrue(0.9 <= ratio <= 2)

if __name__ == '__main__':

    unittest.main()

    test_fits.close()
