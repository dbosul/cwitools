from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from cwitools import reduction
import numpy as np
import os
import unittest

def get_test_fits():
    #Create test data cube
    test_path = __file__.replace("tests/test_reduction.py", "data/test_cube.fits")
    test_fits = fits.open(test_path)
    return test_fits

class ReductionTestCases(unittest.TestCase):

    #Cross-correlate a fits with itself and assert output equals input
    def test_align_crpix3(self):
        test_list = [get_test_fits()]*2
        crpix3_in = [x[0].header["CRPIX3"] for x in test_list]
        crpix3_corr = reduction.align_crpix3(test_list)
        test_res = np.all([crpix3_in[i] == c for i, c in enumerate(crpix3_corr)])
        for x in test_list: x.close()
        self.assertTrue(test_res)

    #Measure the center of QSO (SDSS1112+1521) from fit and compare to WCS
    def test_get_crpix12(self):
        test_fits = get_test_fits()
        test_ra, test_dec = 168.218550543, 015.356529219
        crpix1, crpix2 = reduction.get_crpix12(test_fits, test_ra, test_dec)
        wcs = WCS(test_fits[0].header)
        x0, y0, w0 = wcs.all_world2pix(test_ra, test_dec, 5200, 0)
        dist = np.sqrt((crpix1 - x0)**2 + (crpix2 - y0)**2)
        test_fits.close()
        self.assertTrue(dist <= 5)

    #Bin the data 2x2x2 and check that new shape is 1/2 each axis
    def test_rebin(self):
        test_fits = get_test_fits()
        w0, y0, x0 = test_fits[0].data.shape
        test_fits_rebinned = reduction.rebin(test_fits, xybin=2, zbin=2)
        w1, y1, x1 = test_fits_rebinned[0].data.shape
        test_fits.close()
        self.assertTrue((w0 // w1 == 2) & (y0 // y1 == 2) & (x0 // x1 == 2))

    #Crop the data and check (1) shape and (2) header
    def test_crop(self):
        test_fits = get_test_fits()

        #Get cube shape and position of main RA/DEC before cropping
        h_in = test_fits[0].header
        ra0, dec0, wav0 = h_in["CRVAL1"], h_in["CRVAL2"], h_in["CRVAL3"]
        shape_in = test_fits[0].data.shape
        wav_pix2 = h_in["CRVAL3"] + (2 - h_in["CRPIX3"]) * h_in["CD3_3"]
        wav_pixN = h_in["CRVAL3"] + (shape_in[0] - h_in["CRPIX3"]) * h_in["CD3_3"]

        #Bin the data
        test_fits_cropped = reduction.crop(test_fits,
            xcrop = (2, shape_in[2]),
            ycrop = (2, shape_in[1]),
            wcrop = (wav_pix2, wav_pixN)
        )

        shape_out = test_fits_cropped[0].data.shape
        h_out = test_fits_cropped[0].header

        shape_test = (shape_out[0] == shape_in[0] - 2)\
                     & (shape_out[1] == shape_in[1] - 2)\
                     & (shape_out[2] == shape_in[2] - 2)

        wcs_test = (h_in["CRPIX3"] - h_out["CRPIX3"] == 2)\
                   & (h_in["CRPIX2"] - h_out["CRPIX2"] == 2)\
                   & (h_in["CRPIX1"] - h_out["CRPIX1"] == 2)
        test_fits.close()
        self.assertTrue(shape_test & wcs_test)

    #For this test, just assert that the coadd call successfully completed,
    #As there is no easy automatic validation of the output
    def test_coadd(self):

        #Load fits
        test_fits = get_test_fits()

        #Get same fits but shift coordinate system a bit
        test_fits_2 = test_fits.copy()
        test_fits_2[0].header["CRPIX1"] + 4
        test_fits_2[0].header["CRPIX2"] + 4

        #Coadd the two fits images
        coadd_fits = reduction.coadd([test_fits, test_fits_2])

        self.assertTrue(type(test_fits) == type(coadd_fits))

if __name__ == '__main__':

    unittest.main()
