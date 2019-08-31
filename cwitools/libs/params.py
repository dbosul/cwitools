"""CWITools library for handling of parameter (.param) files.

This module contains functions for loading, creating and saving CWITools
parameter files, which are needed to make use of the CWITools.reduction module.
"""

from astropy.io import fits
import numpy as np
import os
import sys
import warnings

parameterTypes = {  "TARGET_NAME":str,
                    "TARGET_RA":float,
                    "TARGET_DEC":float,
                    "ALIGN_RA":float,
                    "ALIGN_DEC":float,
                    "INPUT_DIRECTORY":str,
                    "OUTPUT_DIRECTORY":str,
                    "SEARCH_DEPTH":int,
                    "ID_LIST":list
                 }

def loadparams(path):
    """Load a CWITools parameter file into a dictionary structure.

    Args:
        path (str): Path to CWITools parameter file.

    Returns:
        dict: Python dictionary containing CWITools parameters

    """
    global parameterTypes

    params = { x:None for x in parameterTypes.keys() }
    params["ID_LIST"] = []

    for line in open(path,'r'):
        line = line[:-1]
        if line=="": continue
        elif line[0]=='>': params["ID_LIST"].append(line.replace('>',''))
        elif '=' in line:
            line = line.replace(' ','')     #Remove white spaces
            line = line.replace('\n','')    #Remove line ending
            line = line.split('#')[0]       #Remove any comments

            key,val = line.split('=')

            if val.upper()=='NONE' or val=='': params[key]=None
            elif parameterTypes[key]==float: params[key]=float(val)
            elif parameterTypes[key]==int: params[key]=int(val)
            else: params[key]=val

    for p in parameterTypes.keys():
        if p not in params:
            warnings.warn("Parameter %s missing from %s."%(p,path))
            params[p] = None

    return params

def findfiles(params,cubetype):
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
    if not os.path.isdir(params["INPUT_DIRECTORY"]):
        for x in params["INPUT_DIRECTORY"]:
            print(x)
        raise NotADirectoryError("Data directory (%s) does not exist. Please correct and try again." % params["INPUT_DIRECTORY"])

    #Load target cubes
    datadir = params["INPUT_DIRECTORY"]
    depth   = params["SEARCH_DEPTH"]
    id_list = params["ID_LIST"]
    N_files = len(id_list)

    target_files = ["" for i in range(N_files) ]
    for root, dirs, files in os.walk(datadir):
        rec = root.replace(datadir,'').count("/")
        if rec > depth: continue
        else:
            for f in files:
                if cubetype in f:
                    for i,ID in enumerate(id_list):
                        if ID in f:
                            target_files[i] = os.path.join(root,f)

    #Print file paths or file not found errors
    incomplete = False
    for i,f in enumerate(target_files):
        if f=="":
            incomplete = True
            warnings.warn("File Not Found (ID:%s, Type:%s)" % (id_list[i],cubetype))

    #Current mode - exit if incomplete
    if incomplete:
        warnings.warn("Some files are missing.")

    return target_files
