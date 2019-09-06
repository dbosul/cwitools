
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

from cwitools import libs

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
imgGroup.add_argument('-weighted',
                    type=str,
                    metavar='bool',
                    help='Use noise weighting to compute moments (True/False)',
                    choices=["True","False"],
                    default="False"
)
args = parser.parse_args()

## PARSE PARAMETERS AND LOAD DATA

#Try to load the fits file
try: F = fits.open(args.cube)
except: sys.exit()#print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()
try: V = fits.open(args.var)
except: sys.exit()#print("Error: could not open '%s'\nExiting."%args.var);sys.exit()


#Extract useful stuff and create useful data structures
cube  = F[0].data
xvar  = V[0].data
print(cube.shape)
cube = libs.science.smooth3d(cube,2.0,axes=(1,2))
xvar = libs.science.smooth3d(xvar,2.0,axes=(1,2),var=True)
cube = libs.science.smooth3d(cube,2.0,axes=[0])
xvar = libs.science.smooth3d(xvar,2.0,axes=[0],var=True)

print(xvar.shape,cube.shape)
xvar[cube<0] = np.inf
cube[cube<0] = 0

w,y,x = cube.shape
h3D   = F[0].header
h2D   = libs.cubes.get_header2d(h3D)
wcs   = WCS(h2D)
wav   = libs.cubes.get_wavaxis(h3D)

pxScales = proj_plane_pixel_scales(wcs)
xScale,yScale = (pxScales[0]*u.deg).to(u.arcsec), (pxScales[1]*u.degree).to(u.arcsec)
pxArea   = ( xScale*yScale ).value

#Convert cube to units of surface brightness (per arcsec2)
cube /= pxArea
xvar /= pxArea
xvar /= pxArea

#Convert cube to units of integrated flux (e.g. F_lambda*delta_lambda)
cube *= h3D["CD3_3"]
xvar *= h3D["CD3_3"]
xvar *= h3D["CD3_3"]

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

    xvarCube[idCube==0] = 0
    var_objNB = np.sum(xvarCube,axis=0)
    unc_objNB = np.sqrt(var_objNB)

    total_fluxNB = np.sum(objNB)*pxArea
    total_fluxNB_error = np.sqrt(np.sum(var_objNB))*pxArea

    cumSB = np.sort(-np.ravel(objNB[objNB>0]*1e-16)); cumSB = -cumSB
    xArea = (np.arange(len(cumSB))+1)*pxArea  # arcsec^2

    #Save Area vs SB cumulative distribution
    table= np.vstack((xArea, cumSB)) ; table=table.T

    spec1d = np.sum(objCube, axis=(1,2))*pxArea
    spec1d_var = np.sum(xvarCube, axis=(1,2))
    spec1d_error = np.sqrt(spec1d_var)*pxArea

    #print('Integrated flux on component and its error (10-16 erg/s/cm2)',total_fluxNB,total_fluxNB_error)

    nbFITS = fits.HDUList([fits.PrimaryHDU(objCube)])
    nbFITS[0].header = h3D
    #nbFITS.writeto(args.cube.replace('.fits','.obj3D.fits'),overwrite=True)
    #print("Saved %s"%args.cube.replace('.fits','.obj3D.fits'))

    nbFITS = fits.HDUList([fits.PrimaryHDU(objNB*1e-16)])
    nbFITS[0].header = h2D
    #nbFITS.writeto(args.cube.replace('.fits','.NB.fits'),overwrite=True)
    #print("Saved %s"%args.cube.replace('.fits','.NB.fits'))

    nbFITS = fits.HDUList([fits.PrimaryHDU(unc_objNB*1e-16)])
    nbFITS[0].header = h2D
    #nbFITS.writeto(args.cube.replace('.fits','.NB_error.fits'),overwrite=True)
    #print("Saved %s"%args.cube.replace('.fits','.NB_error.fits'))

    nbFITS = fits.HDUList([fits.PrimaryHDU(msk2D)])
    nbFITS[0].header = h2D
    #nbFITS.writeto(args.cube.replace('.fits','.NB_mask.fits'),overwrite=True)
    #print("Saved %s"%args.cube.replace('.fits','.NB_mask.fits'))

    if (args.type=='vel' or args.type=='tri'):

        xvar[xvar<=0] = np.inf
        xvar = 1/xvar   # xvar is weight
        if args.weighted=="True": ncube = cube*xvar
        else: ncube = cube
        xvarCube = xvar.copy()
        xvarCube[idCube==0] = 0
        nvarCube = ncube.copy()
        nvarCube[idCube==0] = 0

        #Create canvas for both first and zeroth (f,z) moments
        m0map = np.zeros_like(msk2D,dtype=float)
        m1map = np.zeros_like(m0map)
        num = np.zeros_like(m0map)
        den = np.zeros_like(m0map)
        dm0 = np.zeros_like(m0map)
        m1 = np.zeros_like(m0map)
        dm1 = np.zeros_like(m0map)

        if np.count_nonzero(idCube)==0:
            #Save FITS images
            m0FITS = fits.HDUList([fits.PrimaryHDU(m0map-5000)])
            m0FITS[0].header = h2D
            m0FITS[0].header["M0REF"] = 0
            m0FITS[0].header["BUNIT"] = "km/s"
            #m0FITS.writeto(args.cube.replace('.fits','.V0.fits'),overwrite=True)
            #print("Saved %s"%args.cube.replace('.fits','.V0.fits'))

            m0FITS = fits.HDUList([fits.PrimaryHDU(dm0)])
            m0FITS[0].header = h2D
            m0FITS[0].header["BUNIT"] = "km/s"
            #m0FITS.writeto(args.cube.replace('.fits','.V0_error.fits'),overwrite=True)
            #print("Saved %s"%args.cube.replace('.fits','.V0_error.fits'))

            m1FITS = fits.HDUList([fits.PrimaryHDU(m1map-5000)])
            m1FITS[0].header = h2D
            m1FITS[0].header["BUNIT"] = "km/s"
            #m1FITS.writeto(args.cube.replace('.fits','.V1.fits'),overwrite=True)
            #print("Saved %s"%args.cube.replace('.fits','.V1.fits'))

            m1FITS = fits.HDUList([fits.PrimaryHDU(dm1)])
            m1FITS[0].header = h2D
            m1FITS[0].header["BUNIT"] = "km/s"
            #m1FITS.writeto(args.cube.replace('.fits','.V1_error.fits'),overwrite=True)
            #print("Saved %s"%args.cube.replace('.fits','.V1_error.fits'))

        else:
            #Get zeroth moment map and error

            for i in range(len(wav)):
            	num = num+nvarCube[i,:,:]*wav[i]
            	den = den+nvarCube[i,:,:]

            tmpden = den.copy()
            tmpden[tmpden==0] = np.inf
            m0 = num/tmpden

            tmpxvarCube = xvarCube.copy(); tmpxvarCube[xvarCube<=0] = np.inf
            if args.weighted=="True":
                for i in range(len(wav)): dm0 = dm0 + np.power((wav[i]*den-num), 2)*xvarCube[i,:,:]
            else:
                for i in range(len(wav)): dm0 = dm0 + np.power((wav[i]*den-num), 2)/tmpxvarCube[i,:,:]
            dm0 = np.sqrt(dm0)

            dm0 = dm0/np.power(tmpden, 2)

            #Get first moment map and error
            for i in range(len(wav)): m1 = m1 + nvarCube[i,:,:]*np.power((wav[i]-m0), 2)

            if args.weighted=="True":
            	for i in range(len(wav)): dm1 = dm1 + np.power(np.power(wav[i]-m0, 2)*den-m1, 2)*tmpxvarCube[i,:,:]
            else:
            	for i in range(len(wav)): dm1 = dm1 + np.power(np.power(wav[i]-m0, 2)*den-m1, 2)/tmpxvarCube[i,:,:]

            # dm1 = dm1 + np.power(2*den*m1*dm0, 2)  # error propagation on m0, which enters m1 definition -- do we need this ?
            dm1 = dm1/np.power(tmpden,4)
            dm1 = np.power(dm1, 0.5)

            m1 = m1/tmpden
            m1 = np.sqrt(m1)

            tmpm1 = m1.copy()
            tmpm1[m1<=0] = np.inf
            dm1 = dm1/(2*tmpm1)

            # change units to velocity
            #m0ref = np.sum(m0*objNB)/np.sum(objNB)
            m0ref = np.sum(wav*spec1d)/np.sum(spec1d) # the above is giving identical results, as expected
            dm0ref = np.sum(spec1d_error**2*np.power((wav*np.sum(spec1d)-np.sum(wav*spec1d))/np.power(np.sum(spec1d), 2), 2)); dm0ref = np.sqrt(dm0ref)
            #print('Reference lambda (flux weighted average of the velocity field)', m0ref)
            #print('Redshift', m0ref/1215.8-1, dm0ref/1215.8)
            sigma0 = np.sum(spec1d*(wav-m0ref)**2)/np.sum(spec1d); sigma0 = np.sqrt(sigma0)
            dsigma0 = 1/np.power(np.sum(spec1d), 2)*np.sum(spec1d_error**2*np.power((wav-m0ref)*(wav-m0ref)-sigma0*sigma0,2)); dsigma0 = np.sqrt(dsigma0)
            dsigma0 = dsigma0/2.0/sigma0
            #print('Global dispersion (km/s)', sigma0/m0ref*3e5, dsigma0/m0ref*3e5)

            #print('%25s%15s%15s%15s%15sf'%('target','m0ref','dm0ref','sigma0','dsigma0'))
            print('%25s%15.4f%15.4f%15.4f%15.4f'%(args.cube.split('/')[0],m0ref,dm0ref,sigma0,dsigma0))

            m0 = (m0-m0ref)/m0ref*3e5
            dm0 = dm0/m0ref*3e5
            m0[m0<-5000] = -5000
            m1 = m1/m0ref*3e5
            dm1 = dm1/m0ref*3e5
            m1[m1<=0] = -1
            absvel = (wav-m0ref)/m0ref*3e5

            trg=args.cube.split('/')[0]

            #Save FITS images
            m0FITS = fits.HDUList([fits.PrimaryHDU(m0)])
            m0FITS[0].header = h2D
            m0FITS[0].header["M0REF"] = m0ref
            m0FITS[0].header["BUNIT"] = "km/s"
            #m0FITS.writeto(args.cube.replace('.fits','.V0.fits'),overwrite=True)
            #print("Saved %s"%args.cube.replace('.fits','.V0.fits'))

            m0FITS = fits.HDUList([fits.PrimaryHDU(dm0)])
            m0FITS[0].header = h2D
            m0FITS[0].header["BUNIT"] = "km/s"
            #m0FITS.writeto(args.cube.replace('.fits','.V0_error.fits'),overwrite=True)
            #print("Saved %s"%args.cube.replace('.fits','.V0_error.fits'))

            m1FITS = fits.HDUList([fits.PrimaryHDU(m1)])
            m1FITS[0].header = h2D
            m1FITS[0].header["BUNIT"] = "km/s"
            #m1FITS.writeto(args.cube.replace('.fits','.V1.fits'),overwrite=True)
            #print("Saved %s"%args.cube.replace('.fits','.V1.fits'))

            m1FITS = fits.HDUList([fits.PrimaryHDU(dm1)])
            m1FITS[0].header = h2D
            m1FITS[0].header["BUNIT"] = "km/s"
            m1FITS.writeto(args.cube.replace('.fits','.V1_error.fits'),overwrite=True)
            #print("Saved %s"%args.cube.replace('.fits','.V1_error.fits'))

            #print("")
