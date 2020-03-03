from astropy.io import fits
from cwitools import variance
from cwitools.tests import test_data

import numpy as np
import os
import unittest


class CoordinatesTestCases(unittest.TestCase):

    #For now - this tests both estimate and rescale. Need new test data to
    #perform separate tests.
    def test_estimate_variance(self):
        test_fits = fits.open(test_data.coadd_path)
        var_cube = variance.estimate_variance(test_fits)
        var_np = np.var(test_fits[0].data)
        var_est = np.mean(var_cube)
        ratio = var_est / var_np
        test_fits.close()
        self.assertTrue(0.9 <= ratio <= 2)

if __name__ == '__main__':

    unittest.main()

    test_fits.close()
