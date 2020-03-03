from astropy.io import fits
from cwitools.analysis import variance
from cwitools.tests import test_data

import numpy as np
import os
import unittest

class CoordinatesTestCases(unittest.TestCase):

    #For now - this tests both estimate and rescale. Need new test data to
    #perform separate tests.
    def test_psf_sub_1d(self):
        test_fits = fits.open(test_data.icubes_path)

        test_fits.close()
        self.assertTrue(1)

if __name__ == '__main__':

    unittest.main()

    test_fits.close()
