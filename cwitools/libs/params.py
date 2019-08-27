"""CWITools library for handling of parameter (.param) files.

This module contains functions for loading, creating and saving CWITools
parameter files, which are needed to make use of the CWITools.reduction module.


"""
from astropy.io import fits
import numpy as np
import sys

parameterTypes = {  "TARGET_NAME":str,
                    "RA_ALIGN":float,
                    "DEC_ALIGN":float,
                    "INPUT_DIRECTORY":str,
                    "OUTPUT_DIRECTORY":str,
                    "SEARCH_DEPTH":int,
                    "ID_LIST":list
                 }

#######################################################################
# Check for incomplete parameter data
def verify(params):
    """Verify that all keys are present in a CWITools parameters dictionary"""
    global parameterNames
    for p in parameterNames:
        if not params.has_key(p):
            return False
    return True

def loadparams(paramPath):
    """Load a CWITools parameter file into a dictionary structure.

    Args:
        paramPath (str): Path to CWITools parameter file.

    Returns:
        dict: Python dictionary containing CWITools parameters

    """
    global parameterTypes

    params = { x:None for x in parameterTypes.keys() }
    params["ID_LIST"] = []

    for line in open(paramPath,'r'):

        if line[0]=='>': params["ID_LIST"].append(line.replace('>',''))
        elif '=' in line:
            line = line.replace(' ','')     #Remove white spaces
            line = line.replace('\n','')    #Remove line ending
            line = line.split('#')[0]       #Remove any comments

            key,val = line.split('=')

            if parameterTypes[key]==float: params[key]=float(val)
            elif parameterTypes[key]==int: params[key]=int(val)
            else: params[key]=val

    return params

def findfiles(params,cubeType):
    """Finds the input files given a CWITools parameter file and cube type.

    Args:
        params (dict): CWITools parameters dictionary.
        cubeType (str): Type of cube (e.g. icubes.fits) to load.

    Returns:
        string list: List of file paths of input cubes.

    """
    print(("Locating %s files:" % cubeType))

    #Check data directory exists
    if not os.path.isdir(params["DATA_DIR"]):
        print(("Data directory (%s) does not exist. Please correct and try again." % params["DATA_DIR"]))
        sys.exit()

    #Load target cubes
    target_files = ["" for i in range(len(params["IMG_ID"]))]
    for root, dirs, files in os.walk(params["DATA_DIR"]):
        rec = root.replace(params["DATA_DIR"],'').count("/")
        if rec > int(params["DATA_DEPTH"]): continue
        else:
            for f in files:
                if cubeType in f:
                    for i,ID in enumerate(params["IMG_ID"]):
                        if ID in f: target_files[i] = os.path.join(root,f)

    #Print file paths or file not found errors
    incomplete = False
    for i,f in enumerate(target_files):
        if f!="": print(f)
        else:
            incomplete = True
            print(("File not found: ID:%s Type:%s" % (params["IMG_ID"][i],cubeType)))

    #Current mode - exit if incomplete
    if incomplete:
        print("Some input files are missing. Please make sure files exist or comment out the relevant lines paramfile with '#'")
        sys.exit()

    print("")

    return target_files
