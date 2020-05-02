"""Tools for using CWITools parameter files."""
from astropy.io import fits

import copy
import numpy as np
import os
import sys
import warnings

parameter_fields = {
"TARGET_NAME":str,
"TARGET_RA":float,
"TARGET_DEC":float,
"INPUT_DIRECTORY":str,
"OUTPUT_DIRECTORY":str,
"SEARCH_DEPTH":int,
"ID_LIST":list
}

parameter_defaults = {
"TARGET_NAME":"TARGNAME",
"TARGET_RA":0.0,
"TARGET_DEC":0.0,
"INPUT_DIRECTORY":".",
"OUTPUT_DIRECTORY":".",
"SEARCH_DEPTH":2,
"ID_LIST":['test_id1', 'test_id2']
}

def init_params():
    """Get an initial params dictionary with default values.

    Args:
        None

    Returns:
        dict: A params dictionary with default values.

    """
    global parameter_defaults
    return copy.deepcopy(parameter_defaults)

def load_params(path):
    """Load a CWITools parameter file into a dictionary structure.

    Args:
        path (str): Path to CWITools parameter file.

    Returns:
        dict: Python dictionary containing CWITools parameters

    """
    global parameter_fields

    params = {x:None for x in parameter_fields.keys()}
    params["ID_LIST"] = []

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    
    parfile = open(path, 'r')
    for line in parfile:
        line = line[:-1]
        if line == "": continue
        elif line[0] == '>': params["ID_LIST"].append(line.replace('>', ''))
        elif '=' in line:
            line = line.replace(' ', '')     #Remove white spaces
            line = line.replace('\n', '')    #Remove line ending
            line = line.split('#')[0]        #Remove any comments

            key, val = line.split('=')

            if val.upper() == 'NONE' or val == '':
                params[key] = None

            elif parameter_fields[key] == float:
                params[key] = float(val)

            elif parameter_fields[key] == int:
                params[key] = int(val)

            else:
                params[key] = val

    parfile.close()

    for p in parameter_fields.keys():
        if p not in params:
            warnings.warn("Parameter %s missing from %s."%(p, path))
            params[p] = None

    return params

def write_params(params, path):

    paramfile_string = f"""# CWITOOLS PARAMETER FILE
TARGET_NAME = {params["TARGET_NAME"]}
TARGET_RA   = {params["TARGET_RA"]}
TARGET_DEC  = {params["TARGET_DEC"]}

# INPUT/OUTPUT SETTINGS
INPUT_DIRECTORY = {params["INPUT_DIRECTORY"]}
SEARCH_DEPTH =  {params["SEARCH_DEPTH"]}
OUTPUT_DIRECTORY = {params["OUTPUT_DIRECTORY"]}

# LIST OF UNIQUE IMAGE IDS FOR INPUT FRAMES (ONE PER LINE, STARTING WITH '>')
"""

    for imgnum in params["ID_LIST"]:
        paramfile_string += ">%s\n"%imgnum

    param_file = open(path, 'w')
    param_file.write(paramfile_string)
    param_file.close()
