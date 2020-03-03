from cwitools import parameters
from cwitools.tests import test_data

import os
import unittest


class ParametersTestCases(unittest.TestCase):

    def test_get_pkpc_per_px(self):
        params = parameters.load_params(test_data.param_path)
        self.assertEqual(params["TARGET_NAME"], "TEST_TARGET")

    def test_write_params(self):
        params = {
            "TARGET_NAME" : "TEST",
            "TARGET_RA" : 0,
            "TARGET_DEC": 0,
            "ALIGN_RA": None,
            "ALIGN_DEC": None,
            "INPUT_DIRECTORY" : "/path/to/files/",
            "SEARCH_DEPTH": 2,
            "OUTPUT_DIRECTORY": "/path/to/save/",
            "ID_LIST": [123, 456, 789]
        }
        parameters.write_params(params, "test.param")
        params_reloaded = parameters.load_params("test.param")
        os.remove("test.param")
        self.assertEqual(params_reloaded["TARGET_NAME"], "TEST")

    def test_find_files(self):

        #Make fake parameter file to search for cwitools python scripts
        cwidir = __file__.replace("tests/test_parameters.py", "")
        files = parameters.find_files(
            ["reduction", "coordinates"],
            cwidir,
            ".py",
            depth=0
        )
        #Assert that both scripts are found
        self.assertEqual(len(files), 2)

if __name__ == '__main__':
    unittest.main()
