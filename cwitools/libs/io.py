"""CWITools library for file input/output.

This module contains functions related to loading/saving files.

"""
from astropy.io import fits as fitsIO
import numpy as np
import os
import sys



##################################################################################################
# Find FITS files using params and given cube type
def findfiles(params,cubetype,getSky=False,verbose=3):

    if verbose>1: print(("Locating %s files:" % cubetype))

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
                if cubetype in f:
                    for i,ID in enumerate(params["IMG_ID"]):
                        if ID in f: target_files[i] = os.path.join(root,f)

    #Print file paths or file not found errors
    incomplete = False
    for i,f in enumerate(target_files):
        if f!="":
            if verbose>1: print(f)
        else:
            incomplete = True
            print(("File not found: ID:%s Type:%s" % (params["IMG_ID"][i],cubetype)))

    #Current mode - exit if incomplete
    if incomplete:
        print("Some input files are missing. Please make sure files exist or comment out the relevant lines paramfile with '#'")
        sys.exit()
    if verbose>1: print("")

    return target_files



#######################################################################
#Output image as fits
def saveFits(data,path,header):
        print(("""Saving %s""" % path))
        hdu = fitsIO.PrimaryHDU(data)
        hdulist = fitsIO.HDUList([hdu])
        hdulist[0].header = header
        hdulist.writeto(path,clobber=True)
