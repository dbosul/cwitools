from astropy.io import fits as fitsIO

import argparse
import numpy as np
import os
import sys
import time
import libs

def run(paramPath,cubeType,fileExt=".c.fits"):

    # Add file extension if omitted
    if not ".fits" in cubeType: cubeType += ".fits"

    # Check if any parameter values are missing (set to set-up mode if so)
    params = libs.params.loadparams(paramPath)

    # Check if parameters are complete
    libs.params.verify(params)

    # Get filenames
    files = libs.io.findfiles(params,cubeType)

    # If there are non-NAS cubes - add sky cubes to list to be cropped (same crop params)
    for i in range(len(params["IMG_ID"])):
        img,sky = params["IMG_ID"][i],params["SKY_ID"][i]
        if img!=sky and sky!=-1:
            skyFile = files[i].replace(img,sky)
            if os.path.isfile(skyFile):
                files.append(skyFile)
                params["IMG_ID"].append(sky)
                params["XCROP"].append(params["XCROP"][i])
                params["YCROP"].append(params["YCROP"][i])
                params["WCROP"].append(params["WCROP"][i])
            else:
                print("Warning: File not found - %s"%files[i])
                continue

        elif sky==-1:
            print("Warning: No sky image associated with %s."%img)

    # Open  FITS objects
    fits = [fitsIO.open(f) for f in files]

    # If all input cubes are icubes - try to update variance cubes as well
    propVar  = np.all([ "icube" in fileName for fileName in files])
    if propVar:
        varFiles = [ f.replace("icube","vcube") for f in files ]
        try: varFits = [ fitsIO.open(v) for v in varFiles ]
        except:
            print("Could not load variance input cubes from data directory. Error will not be propagated throughout coadd.")
            propVar=False

    # Crop FITS and make sure units are flux-like before coadding
    print("Cropping cubes...\n"),
    fits = libs.cubes.cropFITS(fits,params)
    for i,f in enumerate(fits):
        cropName = files[i].replace('.fits',fileExt)
        f.writeto(cropName,overwrite=True)
        print("Saved %s"%cropName)

    # Crop the variance cubes too
    if propVar:
        print("\nCropping corresponding variance cubes...\n"),
        var = libs.cubes.cropFITS(varFits,params)
        for i,v in enumerate(varFits):
            cropName = varFiles[i].replace('.fits',fileExt)
            v.writeto(cropName,overwrite=True)
            print("Saved %s"%cropName)

if __name__=="__main__":

    #Handle input with argparse
    parser = argparse.ArgumentParser(description='Crop input cubes according to a CWITools parameter file.')
    parser.add_argument('paramPath',
                        type=str,
                        help='Path to CWITools parameter file.'
    )
    parser.add_argument('cubeType',
                        type=str,
                        help='The PCWI/KCWI cube type to load for each image ID. (e.g. "icubes.fits" or "ocubes.fits") Must end in .fits file extension.'
    )
    parser.add_argument('-ext',
                        type=str,
                        help='The filename extension to add to cropped cubes. Default: .c.fits'
    )
    args = parser.parse_args()

    run(args.paramPath,args.cubeType,fileExt=args.ext)
