from astropy.io import fits
from cwitools import modeling

import numpy as np
import os
import unittest

def test_gauss1d(p, x):
    return p[0]*np.exp(-0.5 * np.power((x - p[1]) / p[2], 2))

def get_testXY(min=-10, max=10, mean=0, std=2, amp=1, N=100):
    x = np.linspace(min, max, N)
    y = amp*np.exp(-0.5 * np.power((x - mean) / std, 2))
    return x, y

class ModelingTestCases(unittest.TestCase):

    def test_rss_func(self):
        min, max, mean, std, amp = -10, 10, 0, 2, 1
        x, y = get_testXY(min=min, max=max, mean=mean, std=std, amp=amp)
        gauss_model = test_gauss1d([2*amp, mean, std], x)
        rss_known = np.sum(np.power(y - gauss_model, 2))
        rss_meas = modeling.rss_func([2*amp, mean, std], x, y, test_gauss1d)
        self.assertEqual(rss_meas, rss_known)

    def test_fit_de(self):
        min, max, mean, std, amp = -10, 10, 0, 2, 1
        x, y = get_testXY(min=min, max=max, mean=mean, std=std, amp=amp)
        bounds = [(min, max), (-3*mean, 2*mean), (std/5, std*5)]
        de_fit = modeling.fit_de(test_gauss1d, bounds, x, y)
        p_fit = de_fit.x
        self.assertEqual(round(p_fit[2], 0), std)

    def test_gauss1d(self):
        xdata = np.arange(-20, 20, step=0.1)
        ydata = modeling.gauss1d([2.0, 3.0, 4.0], xdata)
        amp = np.max(ydata)
        avg = np.sum(ydata * xdata) / np.sum(ydata)
        std = np.sqrt(np.sum(ydata * (xdata - avg)**2) / np.sum(ydata))
        amp = round(amp, 0)
        avg = round(avg, 0)
        std = round(std, 0)
        self.assertTrue((std == 4) & (avg == 3) & (amp == 2))

    #Numerical test TBD - just testing function call for now
    def test_moffat1d(self):
        xdata = np.arange(-20, 20, step=0.1)
        ydata = modeling.moffat1d([2.0, 3.0, 1.0, 1.0], xdata)
        self.assertTrue(len(ydata) == len(xdata))

    def test_bic_weights(self):
        bic_vals = [400, 440, 500]
        bic_weights = modeling.bic_weights(bic_vals)
        self.assertTrue(np.nanargmax(bic_weights) == np.nanargmin(bic_vals))

    def test_rss(self):
        min, max, mean, std, amp = -10, 10, 0, 2, 1
        x, y = get_testXY(min=min, max=max, mean=mean, std=std, amp=amp)
        gauss_model = test_gauss1d([2*amp, mean, std], x)
        rss_known = np.sum(np.power(y - gauss_model, 2))
        rss_meas = modeling.rss(y, gauss_model)
        self.assertEqual(rss_meas, rss_known)

    def test_sigma2fwhm(self):
        sig = 2.0
        fwhm = 2*np.sqrt(2*np.log(2))*sig
        self.assertTrue(modeling.sigma2fwhm(sig) == fwhm)

    def test_fwhm2sigma(self):
        sig = 2.0
        fwhm = 2*np.sqrt(2*np.log(2))*sig
        self.assertTrue(modeling.fwhm2sigma(fwhm) == sig)

    def test_bic(self):
        min, max, mean, std, amp = -10, 10, 0, 2, 1
        model_params = [2*amp, mean, std]
        x, y = get_testXY(min=min, max=max, mean=mean, std=std, amp=amp)
        gauss_model = test_gauss1d(model_params, x)
        k = len(model_params)
        n = len(y)
        rss = np.sum(np.power(y - gauss_model, 2))
        bic = n * np.log(rss / n) + k * np.log(n)
        self.assertEqual(bic, modeling.bic(gauss_model, y, k))

    def test_aic(self):
        min, max, mean, std, amp = -10, 10, 0, 2, 1
        model_params = [2*amp, mean, std]
        x, y = get_testXY(min=min, max=max, mean=mean, std=std, amp=amp)
        gauss_model = test_gauss1d(model_params, x)
        k = len(model_params)
        n = len(y)
        rss = np.sum(np.power(y - gauss_model, 2))
        aic = 2 * k + n * np.log(rss)
        aic += (2 * k * k + 2 * k) / (n - k - 1)

        self.assertEqual(aic, modeling.aic(gauss_model, y, k))


if __name__ == '__main__':

    unittest.main()

    test_fits.close()
