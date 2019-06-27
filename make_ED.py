
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from photutils import DAOStarFinder
from scipy.stats import sigmaclip

import astropy.convolution as astConvolve
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pyregion
import sys
import time

import libs



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
                    metavar='variance',
                    help='The associated variance cube.'
)
objGroup = parser.add_argument_group(title="Objects")
objGroup.add_argument('-obj',
                    type=str,
                    metavar='path',
                    help='Object Mask cube.',
)
objGroup.add_argument('-objID',
                    type=str,
                    metavar='str',
                    help='The ID of the object to use. Use -1 for all objects. Can also provide multiple as comma-separated list.',
                    default='-1'
)
imgGroup = parser.add_argument_group(title="Image Settings")
imgGroup.add_argument('-type',
                    type=str,
                    metavar='str',
                    help='Type of image to be made: wl=white light, nb=pseudo-NB, vel=velocity (0th and 1st moment, tri=nb+vel+spc). Default is white-light image.',
                    default='wl',
                    choices=['wl','nb','vel','spc','tri']
)
args = parser.parse_args()

## PARSE PARAMETERS AND LOAD DATA

#Try to load the fits file
try: F = fits.open(args.cube)
except: print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()
try: V = fits.open(args.var)
except: print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()


#Extract useful stuff and create useful data structures
cube  = F[0].data
xvar  = V[0].data

w,y,x = cube.shape
h3D   = F[0].header
h2D   = libs.cubes.get2DHeader(h3D)
wcs   = WCS(h2D)
wav   = libs.cubes.getWavAxis(h3D)

pxScales = proj_plane_pixel_scales(wcs)
xScale,yScale = (pxScales[0]*u.deg).to(u.arcsec), (pxScales[1]*u.degree).to(u.arcsec)
pxArea   = ( xScale*yScale ).value

#Convert cube to units of surface brightness (per arcsec2)
cube /= pxArea

#Convert cube to units of integrated flux (e.g. F_lambda*delta_lambda)
cube *= h3D["CD3_3"]

#If user wants to make pseudo NB image or velocity map   # only this option left (ED)
if args.type in ['nb','vel','spc','tri']:

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

    #Get 2D mask of useable spaxels
    msk2D = np.max(idCube,axis=0)

    #Create copy of input cube with non-object voxels set to zero
    objCube = cube.copy()
    objCube[idCube==0] = 0

    #Now use binary mask to generate the requested data product
    if args.type=='nb' or args.type=='tri':

        objNB = np.sum(objCube,axis=0)

        xvarCube = xvar.copy()
        xvarCube[idCube==0] = np.inf
        unc_objNB = np.sqrt(np.sum(1/xvarCube,axis=0))


        nbFITS = fits.HDUList([fits.PrimaryHDU(objNB)])
        nbFITS[0].header = h2D
        nbFITS.writeto(args.cube.replace('.fits','.NB.fits'),overwrite=True)
        print("Saved %s"%args.cube.replace('.fits','.NB.fits'))

        nbFITS = fits.HDUList([fits.PrimaryHDU(unc_objNB)])
        nbFITS[0].header = h2D
        nbFITS.writeto(args.cube.replace('.fits','.NB_sigma.fits'),overwrite=True)
        print("Saved %s"%args.cube.replace('.fits','.NB_sigma.fits'))

        nbFITS = fits.HDUList([fits.PrimaryHDU(msk2D)])
        nbFITS[0].header = h2D
        nbFITS.writeto(args.cube.replace('.fits','.NB_mask.fits'),overwrite=True)
        print("Saved %s"%args.cube.replace('.fits','.NB_mask.fits'))

    if (args.type=='vel' or args.type=='tri') and np.count_nonzero(idCube)>0:

	xvar[xvar<=0] = np.inf
	xvar = 1/xvar
	ncube = cube*xvar
    	xvarCube = xvar.copy()
    	xvarCube[idCube==0] = 0
    	nvarCube = ncube.copy()
    	nvarCube[idCube==0] = 0


        #Create canvas for both first and zeroth (f,z) moments
        m0map = np.zeros_like(msk2D,dtype=float)
        m1map = np.zeros_like(m0map)

        #Get zeroth moment map and error
	num = np.zeros_like(m0map)
	den = np.zeros_like(m0map)
	dm0 = np.zeros_like(m0map)
	m1 = np.zeros_like(m0map)
	dm1 = np.zeros_like(m0map)

	for i in range(len(wav)):
		num = num+nvarCube[i,:,:]*wav[i]
		den = den+nvarCube[i,:,:]

	tmpden = den.copy()
	tmpden[tmpden==0] = np.inf
	m0 = num/tmpden

	for i in range(len(wav)):
		dm0 = dm0 + np.power((wav[i]*den-num), 2)*xvarCube[i,:,:]

	dm0 = np.sqrt(dm0)

	dm0 = dm0/np.power(tmpden, 2)

        #Get first moment map and error
	for i in range(len(wav)):
		m1 = m1 + nvarCube[i,:,:]*np.power((wav[i]-m0), 2)

	for i in range(len(wav)):
		dm1 = dm1 + np.power(np.power(wav[i]-m0, 2)*den-m1, 2)*xvarCube[i,:,:]

	dm1 = dm1 + np.power(2*den*m1*dm0, 2)
	dm1 = np.sqrt(dm1)
	dm1 = dm1/np.power(tmpden,2)

	m1 = m1/tmpden
	m1 = np.sqrt(m1)

	tmpm1 = m1.copy()
	tmpm1[m1<=0] = np.inf
	dm1 = dm1/(2*tmpm1)

	# change units to velocity
	m0ref = np.sum(m0*objNB)/np.sum(objNB)
	print('Reference lambda (flux weighted average of the velocity field)', m0ref)
	m0 = (m0-m0ref)/m0ref*3e5
	dm0 = dm0/m0ref*3e5
	m0[m0<-5000] = -5000
	m1 = m1/m0ref*3e5
	dm1 = dm1/m0ref*3e5
	m1[m1<=0] = -1


        #Save FITS images
        m0FITS = fits.HDUList([fits.PrimaryHDU(m0)])
        m0FITS[0].header = h2D
        m0FITS[0].header["BUNIT"] = "km/s"
        m0FITS.writeto(args.cube.replace('.fits','.V0.fits'),overwrite=True)
        print("Saved %s"%args.cube.replace('.fits','.V0.fits'))

        m0FITS = fits.HDUList([fits.PrimaryHDU(dm0)])
        m0FITS[0].header = h2D
        m0FITS[0].header["BUNIT"] = "km/s"
        m0FITS.writeto(args.cube.replace('.fits','.V0_error.fits'),overwrite=True)
        print("Saved %s"%args.cube.replace('.fits','.V0_error.fits'))


        m1FITS = fits.HDUList([fits.PrimaryHDU(m1)])
        m1FITS[0].header = h2D
        m1FITS[0].header["BUNIT"] = "km/s"
        m1FITS.writeto(args.cube.replace('.fits','.V1.fits'),overwrite=True)
        print("Saved %s"%args.cube.replace('.fits','.V1.fits'))

        m1FITS = fits.HDUList([fits.PrimaryHDU(dm1)])
        m1FITS[0].header = h2D
        m1FITS[0].header["BUNIT"] = "km/s"
        m1FITS.writeto(args.cube.replace('.fits','.V1_error.fits'),overwrite=True)
        print("Saved %s"%args.cube.replace('.fits','.V1_error.fits'))
