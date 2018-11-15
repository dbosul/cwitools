#!/usr/bin/env python

from astropy.io import fits as fitsIO

import os
import sys
import time
import libs


def getInt(prompt,default=None): 
    done=False
    while not done:
        val = raw_input("\n%s (Default:%s) \n> " % (prompt,str(default)))
        if val=="" and default!=None: return default
        else:
            try:
                val = int(val)
                return val
            except: print("Enter integer value.")
                
def getString(prompt,default=None):
    done=False
    while not done:
        val = raw_input("\n%s (Default:%s) \n> " % (prompt,str(default)))
        if val=="" and default!=None: return default
        else: return val

def getDir(prompt,default=None):
    done=False
    while not done:
        val = raw_input("\n%s (Default:%s) \n> " % (prompt,str(default)))
        if val=="" and default!=None: return default
        else:
            if os.path.isdir(val): return os.path.abspath(val)
            else: print("%s is not a directory.")

def getFile(prompt,default=None):
    done=False
    while not done:
        val = raw_input("\n%s (Default:%s) \n> " % (prompt,str(default)))
        if val=="" and default!=None: return default
        else:
            if os.path.isfile(val): return os.path.abspath(val)
            else: print("File does not exist: %s."%val)  

def getString(prompt,default=None):
    done=False
    while not done:
        val = raw_input("\n%s (Default:%s) \n> " % (prompt,str(default)))
        if val=="" and default!=None: return default
        else: return val  
        
def getFloat(prompt,default=None): 
    done=False
    while not done:
        val = raw_input("\n%s (Default:%s) \n> " % (prompt,str(default)))
        if val=="" and default!=None: return default
        else:
            try:
                val = float(val)
                return val
            except: print("Enter decimal value.")
                
def getList(prompt,default=None): 
    done=False
    while not done:
        val = raw_input("\n%s (Default:%s) \n> " % (prompt,str(default)))
        vals = val.split(',')             
        if len(vals)==0 or val=="": print("Enter comma-separated list. At least one image ID is needed for a valid param file.") 
        else: return vals  
                     
#Timer start
tStart = time.time()
curDir = os.getcwd()

#Get required parameters
params = {}
params["NAME"]          = getString("Target Name")
params["RA"]            = getFloat("RA  (dd.ddd)",default=0.0)
params["DEC"]           = getFloat("DEC (dd.ddd)",default=0.0)
params["Z"]             = getFloat("Redshift",default=0.0)
params["ZLA"]           = getFloat("Lyman-Alpha Redshift",default=params["Z"])
params["DATA_DIR"]      = getDir("Data Directory (top level directory for input files)",default=curDir)
params["DATA_DEPTH"]    = getInt("Data Depth (how many levels below 'data directory' to search)",default=2)
params["REG_FILE"]      = getFile("Region File (DS9 .reg file for continuum sources)",default="None")
params["PRODUCT_DIR"]   = getDir("Product directory (where coadded frames will be saved.)",default=curDir)
params["IMG_ID"]        = getList("Image IDs (comma-separated list of unique identifier strings for each input frame)")

try:
    
    cubeType = getString("What cube type do you want to load to auto-fill the rest of the params from headers?")
    if not ".fits" in cubeType: cubeType+=".fits"
    
    #Get filenames     
    files = libs.io.findfiles(params,cubeType)

    #Open custom FITS-3D objects
    fits = [fitsIO.open(f) for f in files] 

    #1 - verify the parameter files
    libs.params.verify(params)

    #2 - re-initialize param values from FITS headers
    params = libs.params.parseHeaders(params,fits)

except:

    print("Something went wrong. Will use default values for now. You can run 'fillParams' to auto-populate the params later.")
    
    #Initialize defaults for the rest of the params
    params["SKY_ID"] = [-1 for imgID in params["IMG_ID"]]
    params["INST"]   = ['?' for imgID in params["IMG_ID"]]
    params["XCROP"]  = ["0:-1" for imgID in params["IMG_ID"]]
    params["YCROP"]  = ["0:-1" for imgID in params["IMG_ID"]]
    params["WCROP"]  = ["0:-1" for imgID in params["IMG_ID"]]

#Get info for saving paramfile
paramName = getString("Parameter file name",default="%s.param"%params["NAME"])
if not ".param" in paramName: paramName+=".param"
paramDir = getString("Where to save %s" % paramName,default=curDir)

paramPath = "%s/%s" % (paramDir,paramName)

libs.params.writeparams(params,paramPath)

print("Wrote to %s"%paramPath)
