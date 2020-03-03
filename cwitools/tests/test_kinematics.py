from cwitools import kinematics

import numpy as np
import unittest

class KinematicsTestCases(unittest.TestCase):

    def test_first_moment(self):
        wav_lo, wav_hi, wav_size = 4300, 4400, 200
        wav_mean, wav_sig = 4350, 3
        x = np.linspace(wav_lo, wav_hi, wav_size)
        y = np.exp(-0.5 * np.power((x - wav_mean) / wav_sig, 2))
        m1 = kinematics.first_moment(x, y)
        self.assertTrue(np.abs(m1 - wav_mean) < 1)

    def test_second_moment(self):
        wav_lo, wav_hi, wav_size = 4300, 4400, 200
        wav_mean, wav_sig = 4350, 3
        x = np.linspace(wav_lo, wav_hi, wav_size)
        y = np.exp(-0.5 * np.power((x - wav_mean) / wav_sig, 2))
        m2 = kinematics.second_moment(x, y)
        self.assertTrue(np.abs(m2 - wav_sig) < 1)

    #No numerical validation is included here, for lack of a separate
    #and independent error calculation to test against.
    def test_first_moment_err(self):
        wav_lo, wav_hi, wav_size = 4300, 4400, 200
        wav_mean, wav_sig = 4350, 3
        x = np.linspace(wav_lo, wav_hi, wav_size)
        y = np.exp(-0.5 * np.power((x - wav_mean) / wav_sig, 2))
        m1_err = kinematics.first_moment_err(x, y)
        self.assertTrue(1)

    #As above. Just checking function call executes successfully
    def test_second_moment_err(self):
        wav_lo, wav_hi, wav_size = 4300, 4400, 200
        wav_mean, wav_sig = 4350, 3
        x = np.linspace(wav_lo, wav_hi, wav_size)
        y = np.exp(-0.5 * np.power((x - wav_mean) / wav_sig, 2))
        m2_err = kinematics.second_moment_err(x, y)
        self.assertTrue(1)

    #As above. Just checking function call executes successfully
    def test_closing_window_moments(self):
        wav_lo, wav_hi, wav_size = 4300, 4400, 200
        wav_mean, wav_sig = 4350, 3

        x = np.linspace(wav_lo, wav_hi, wav_size)
        y = np.exp(-0.5 * np.power((x - wav_mean) / wav_sig, 2))

        m1, m2 = kinematics.closing_window_moments(x, y, get_err=False)
        m1, m2, m1_err, m2_err = kinematics.closing_window_moments(x, y,
            get_err=True
        )

        self.assertTrue(1)

if __name__ == '__main__':

    unittest.main()
