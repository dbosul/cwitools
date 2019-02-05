#!/usr/bin/env python
#
# coadd - Crop, Scale, Rotate (90deg only), Align and Coadd individual CWI observations.
# 
# syntax: python coadd.py <parameterFile> <cubeType>
#
# Optional parameters:
# - expThreshold (float) (0-1) - Minimum required overlap to crop to. (e.g. 0 = do not crop, 1=take only maximal overlap of input frames)
# - propVar (bool) - Whether to propagate the variance into a 'coadded' variance cube
# - pixelThreshold (float) (0-1) - For a pixel in the coadd frame, minimum coverage by input pixels as fraction of total pixel area.
#

from astropy.io import fits as fitsIO
import argparse
import numpy as np
import sys
import time
import libs

#Timer start
tStart = time.time()

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Coadd data cubes.')

mainGroup = parser.add_argument_group(title="Main",description="Basic input")
mainGroup.add_argument('paramFile', 
                    type=str, 
                    metavar='Parameter File',             
                    help='CWITools Parameter file (used to load cube list etc.)'
)
mainGroup.add_argument('cubeType', 
                    type=str, 
                    metavar='Cube Type',             
                    help='The type of cube (i.e. file extension such as \'icubed.fits\') to coadd'
)

methodGroup = parser.add_argument_group(title="Methods",description="Parameters related to coadd methods.")
methodGroup.add_argument('-pxThresh',
                    type=float,
                    metavar='Pixel Threshold',
                    help='Fraction of a coadd-frame pixel that must be covered by an input frame to be included (0-1)',
                    default=0.9
)
methodGroup.add_argument('-expThresh',
                    type=float,
                    metavar='Exposure Threshold',
                    help='Crop cube to include only spaxels with this fraction of the maximum overlap (0-1)',
                    default=0.75
)

fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")
fileIOGroup.add_argument('-propVar',
                    type=bool,
                    metavar='Propagate Error',
                    help='Propagate error through coadd process (i.e. coadd the variance cubes also.)',
                    default=False
)
args = parser.parse_args()
             
#Try to load the param file
try: params = libs.params.loadparams(args.paramFile)
except: print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()

#Check if parameters are complete
libs.params.verify(params)

  
#Add file extension of omitted
if not ".fits" in args.cubeType: args.cubeType += ".fits"

#Get filenames     
files = libs.io.findfiles(params,args.cubeType)

#Stack cubes and trim
stackedFITS,varFITS = libs.cubes.coadd(files,params,expThresh=args.expThresh,pxThresh=args.pxThresh,propVar=args.propVar)  

#Add redshift info to header
stackedFITS[0].header["Z"] = params["Z"]
stackedFITS[0].header["ZLA"] = params["ZLA"]

#Save stacked cube
stackedpath = '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],args.cubeType)
stackedFITS[0].writeto(stackedpath,overwrite=True)
print "\nSaved %s" % stackedpath

#Save variance cube if one was returned
if varFITS!=None:
    varpath = '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],args.cubeType.replace("icube","vcube"))
    varFITS[0].writeto(varpath,overwrite=True)
    print "Saved %s" % varpath    
    
#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))
