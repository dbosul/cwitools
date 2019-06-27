
from astropy.io import fits
from astropy.wcs import WCS

import argparse
import astropy.convolution as astConvolve
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

import libs #CWITools import



#Timer start
tStart = time.time()

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Make products from data cubes and object masks.')
mainGroup = parser.add_argument_group(title="Main",description="Basic input")
mainGroup.add_argument('cube',
                    type=str,
                    metavar='cube',
                    help='The input data cube.'
)
mainGroup.add_argument('var',
                    type=str,
                    metavar='cube',
                    help='Corresponding variance cube, used for weighting..'
)

mainGroup.add_argument('-obj',
                    type=str,
                    metavar='path',
                    help='Object Mask cube.',
)
mainGroup.add_argument('-objID',
                    type=str,
                    metavar='str',
                    help='The ID of the object to use. Use -1 for all objects. Can also provide multiple as comma-separated list.',
                    default='-1'
)
mainGroup.add_argument('-par',
                    type=str,
                    metavar='path',
                    help='Center the image on a target using CWITools parameter file.',
                    default=None
)
args = parser.parse_args()

## Replace <=0 values with infinities
def zer2inf(a):
    a2 = a.copy()
    a2[a==0]=np.inf
    return a2

#Try to load the fits file
try: cube,h3D = fits.getdata(args.cube,header=True)
except: print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()

#Try to load variance
try: varcube = fits.getdata(args.var); varIn = True
except: varcube = None; varIn = False; print("Failed to load variance.")

#Get inverse-variance weighted cubes
libs.science.nonpos2inf(varcube)
xvar = 1/varcube.copy()
xint = cube.copy()/xvar.copy()

#Extract useful stuff and create useful data structures
w,y,x = cube.shape
h2D   = libs.cubes.get2DHeader(h3D)
wav   = libs.cubes.getWavAxis(h3D)

#Load object info
if args.obj==None: print("Must provide object mask (-obj) if you want to make a pseudo-NB or velocity map of an object.");sys.exit()
try: O = fits.open(args.obj)
except: print("Error opening object mask: %s"%args.obj);sys.exit()

try: objIDs = list( int(x) for x in args.objID.split(',') )
except: print("Could not parse -objID list. Should be int or comma-separated list of ints.");sys.exit()


#If object info is loaded - now turn object mask into binary mask using objIDs
idCube = O[0].data
if objIDs==[-1]: idCube[idCube>0] = 1
elif objIDs==[-2]: idCube[idCube>0] = 0
else:
    for obj_id in objIDs: idCube[idCube==obj_id] = -99
    idCube[idCube>0] = 0
    idCube[idCube==-99] = 1

#Create copy of input cube with non-object voxels set to zero
xint[idCube==0] = 0
xvar[idCube==0] = 0

#Get 2D mask of useable spaxels
msk2D = np.max(idCube,axis=0)

#Calculate first moment
m1_num = np.zeros((y,x))
m1_den = m1_num.copy()
for i in range(len(wav)):
    m1_num += wav[i]*xint[i]
    m1_den += xint[i]
libs.science.nonpos2inf(m1_den)
m1map = m1_num/m1_den
m1path = args.cube.replace('.fits','.V1.fits')
libs.cubes.saveFITS(m1map,h2D,m1path)

#Calculate error on first moment
m1_err = np.zeros((y,x))
for i in range(len(wav)):
    m1_err += ( np.power(wav[i]/zer2inf(m1_num),2) )*xvar[i]
m1_err = np.sqrt(m1_err)
m1_err *= np.abs( m1map )
m1errpath = args.cube.replace('.fits','.V1err.fits')
libs.cubes.saveFITS(m1_err,h2D,m1errpath)
