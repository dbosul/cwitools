"""Tools for using CWITools parameter files."""
from astropy.io import fits

import numpy as np
import os
import sys
import warnings

parameter_fields = {
"TARGET_NAME":str,
"TARGET_RA":float,
"TARGET_DEC":float,
"ALIGN_RA":str,
"ALIGN_DEC":str,
"ALIGN_WAV":str,
"INPUT_DIRECTORY":str,
"OUTPUT_DIRECTORY":str,
"SEARCH_DEPTH":int,
"ID_LIST":list
}

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

    # RA/DEC OF SOURCE USED TO ALIGN CUBES (if None, TARGET RA/DEC will be used)
    ALIGN_RA  = {params["ALIGN_RA"]}
    ALIGN_DEC = {params["ALIGN_DEC"]}

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

def find_files(id_list, datadir, cubetype, depth=3):
    """Finds the input files given a CWITools parameter file and cube type.

    Args:
        params (dict): CWITools parameters dictionary.
        cubetype (str): Type of cube (e.g. icubes.fits) to load.

    Returns:
        list(string): List of file paths of input cubes.

    Raises:
        NotADirectoryError: If the input directory does not exist.

    """

    #Check data directory exists
    if not os.path.isdir(datadir):
        raise NotADirectoryError("Data directory (%s) does not exist. Please correct and try again." % datadir)

    #Load target cubes
    N_files = len(id_list)
    target_files = []
    typeLen = len(cubetype)

    for root, dirs, files in os.walk(datadir):

        if root[-1] != '/': root += '/'
        rec = root.replace(datadir, '').count("/")

        if rec > depth: continue
        else:
            for f in files:
                if f[-typeLen:] == cubetype:
                    for i,ID in enumerate(id_list):
                        if ID in f:
                            target_files.append(root + f)

    #Print file paths or file not found errors
    if len(target_files) < len(id_list):
        warnings.warn("Some files were not found:")
        for id in id_list:
            is_in = np.array([ id in x for x in target_files])
            if not np.any(is_in):
                warnings.warn("Image with ID %s and type %s not found." % (id, cubetype))


    return sorted(target_files)
