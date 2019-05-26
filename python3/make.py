
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from astropy.modeling import models,fitting
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from photutils import DAOStarFinder
from scipy.stats import sigmaclip
from scipy.ndimage.measurements import center_of_mass as CoM

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
objGroup.add_argument('-zWing',
                    type=int,
                    metavar='int',
                    help='Additional z-pixels to use (blue/red of object mask) when calculating moments.\
                    This is useful because the object mask is thresholded, meaning you often get the peak but not the full line shape. (Default 5 px)',
                    default=10
)
objGroup.add_argument('-zSNR',
                    type=float,
                    metavar='float',
                    help='Minimum integrated SNR of a spaxel spectrum before using to calculate velocity (Default 3)',
                    default=3
)
objGroup.add_argument('-nl',
                    type=int,
                    metavar='int',
                    help='The wavelength layer to sample noise from for pseudoNB.',
                    default=None
)
imgGroup = parser.add_argument_group(title="Image Settings")
imgGroup.add_argument('-type',
                    type=str,
                    metavar='str',
                    help='Type of image to be made: wl=white light, nb=pseudo-NB, vel=velocity (0th and 1st moment, tri=nb+vel+spc). Default is white-light image.',
                    default='wl',
                    choices=['wl','nb','vel','spc','tri']
)
imgGroup.add_argument('-wav0',
                    type=float,
                    metavar='float',
                    help='If making velocity map - you can set the central wavelength with this.'
)
imgGroup.add_argument('-centerOn',
                    type=str,
                    metavar='path',
                    help='Center the image on a target using CWITools parameter file.',
                    default=None
)
imgGroup.add_argument('-boxSize',
                    type=float,
                    metavar='float',
                    help='If using -centerOn, this determines the box size around the target in pkpc or pixels. Set unit with -boxUnit.'
)
imgGroup.add_argument('-boxUnit',
                    type=str,
                    metavar='str',
                    help='Unit for -boxSize setting [px/arcsec/pkpc]. Defaults to px.',
                    choices=['px','pkpc','arcsec'],
                    default='px'
)
imgGroup.add_argument('-zmask',
                    type=str,
                    metavar='int tuple',
                    help='Z-indices to mask when making WL image (e.g. \'21,32\')',
                    default='0,0'
)
imgGroup.add_argument('-zunit',
                    type=str,
                    metavar='str',
                    help='Unit of input for zmask. Can be Angstrom (A) or Pixels (px) (Default: A)',
                    default='A',
                    choices=['A','px']
)
imgGroup.add_argument('-wSmooth',
                    type=float,
                    metavar='str',
                    help='Wavelength smoothing kernel radius (pixels). Default: None.',
)
imgGroup.add_argument('-wkernel',
                    type=str,
                    metavar='str',
                    help='Type of kernel to use for wavelength smoothing',
                    default='box',
                    choices=['box','gaussian']
)
imgGroup.add_argument('-xySmooth',
                    type=float,
                    metavar='str',
                    help='Wavelength smoothing kernel radius (pixels). Default: None'
)
imgGroup.add_argument('-xykernel',
                    type=str,
                    metavar='str',
                    help='Type of kernel to use for wavelength smoothing',
                    default='box',
                    choices=['box','gaussian']
)
args = parser.parse_args()

## PARSE PARAMETERS AND LOAD DATA

#Try to load the fits file
try: F = fits.open(args.cube)
except: print(("Error: could not open '%s'\nExiting."%args.cube));sys.exit()


#Try to parse the wavelength mask tuple
try: z0,z1 = tuple(int(x) for x in args.zmask.split(','))
except: print("Could not parse zmask argument. Should be two comma-separated integers (e.g. 21,32)");sys.exit()

#Prepare smoothing kernels if needed

#Spatial
if args.xySmooth!=None:
    if args.xykernel=="box": Kxy = astConvolve.Box2DKernel(2*args.xySmooth)
    else: Kxy = astConvolve.Gaussian2DKernel(x_stddev=fwhm2sig(2*args.xySmooth),y_stddev=fwhm2sig(2*args.xySmooth))
    
#Wavelength
if args.wSmooth!=None:
    if args.wkernel=="box": Kw = astConvolve.Box1DKernel(2*args.wSmooth)
    else: Kw = astConvolve.Gaussian1DKernel(stddev=fwhm2sig(2*args.wSmooth))
                        
      
#Extract useful stuff and create useful data structures
cube  = F[0].data
w,y,x = cube.shape
h3D   = F[0].header
h2D   = libs.cubes.get2DHeader(h3D) 
wcs   = WCS(h2D)
wav   = libs.cubes.getWavAxis(h3D)

pxScales = proj_plane_pixel_scales(wcs)
xScale,yScale = (pxScales[0]*u.deg).to(u.arcsec), (pxScales[1]*u.degree).to(u.arcsec) 
pxArea   = ( xScale*yScale ).value

#If -centerOn flag is given - make a new cube/header with the given size/center
if args.centerOn!=None: 
    try: params = libs.params.loadparams(args.centerOn)
    except: print("Could not open parameter file (-centerOn flag). Please check path and try again.");sys.exit()
    
    xC,yC = wcs.all_world2pix(params["RA"],params["DEC"],0)  
    
    #If user did not give boxSize, take largest spatial axis
    if args.boxSize==None:
        print("No -boxSize given with -centerOn flag. Using maximum dimension.")
        args.boxSize = max(y,x)
    
    #Otherwise...  
    else:
    
        #If unit for boxsize is arcseconds - convert to pixels
        if args.boxUnit=='arcsec': args.boxSize /= xScale.value
        
        #If unit for boxsize is proper kpc - convert to pixels
        elif args.boxUnit=='pkpc':      
            pkpc_per_arcsec = cosmo.kpc_proper_per_arcmin(params["Z"])/60.0        
            pkpc_per_pixel  = xScale*pkpc_per_arcsec       
            args.boxSize /= pkpc_per_pixel.value

        #If the unit is already in pixels - do nothing ( code is just for structural clarity)
        else: pass
    
    #Crop cube to this 2D region  
    bSize = int(round(args.boxSize))
    cubeC = np.zeros((w,bSize,bSize))
    for wi in range(w):
        cutout = Cutout2D(cube[wi],(xC,yC),bSize,wcs,mode='partial',fill_value=0)
        cubeC[wi] = cutout.data
        if wi==0: hdrNew = cutout.wcs.to_header()
    
    #Replace cube and update wcs/dim values
    cube  = cubeC
    w,y,x = cubeC.shape
    wcs   = cutout.wcs
    
    #Update 3D header with new WCS values
    for i in [1,2]:
        h3D["CRVAL%i"%i] = hdrNew["CRVAL%i"%i]
        h3D["CRPIX%i"%i] = hdrNew["CRPIX%i"%i]
    
    #Update 2D header
    h2D = libs.cubes.get2DHeader(h3D)
    
#General Prep
def fwhm2sig(fwhm): return fwhm/(2*np.sqrt(2*np.log(2)))

#Convert cube to units of surface brightness (per arcsec2)
cube /= pxArea

#Convert cube to units of integrated flux (e.g. F_lambda*delta_lambda)
cube *= h3D["CD3_3"]

#Convert cube from FLAM16 to FLAM18
cube *= 100
        
                
#Now to make the product
if args.type=='wl':
    
    #Convert zmask to pixels if given in angstrom
    if args.zunit=='A': z0,z1 = libs.cubes.getband(z0,z1,h3D)

    #Mask cube
    cube[z0:z1] = 0
    
    #Sum along wavelength axis
    img = np.sum(cube,axis=0)

    useX = np.sum(img,axis=0)!=0
    useY = np.sum(img,axis=1)!=0
    
    for yi in range(y):
        if useY[yi]:      
            med = np.median(sigmaclip(img[yi,useX],high=2.5)[0])
            if np.isnan(med): continue
            img[yi,useX] -= med
            
    for xi in range(x):
        if useX[xi]:
            med = np.median(sigmaclip(img[useY,xi],high=2.5)[0])
            if np.isnan(med): continue
            img[useY,xi] -= med

    #Apply spatial smoothing to Narrow-Band Image if option is set
    if args.xySmooth!=None: img = astConvolve.convolve(img,Kxy)
            
    #Adjust values take into account in 
    F[0].data = img
    F[0].header = libs.cubes.get2DHeader(h3D)
    F.writeto(args.cube.replace('.fits','.WL.fits'),overwrite=True)
    print(("Saved %s"%args.cube.replace('.fits','.WL.fits')))
    
#If user wants to make pseudo NB image or velocity map
elif args.type in ['nb','vel','spc','tri']:
    
    #Load object info   
    if args.obj==None: print("Must provide object mask (-obj) if you want to make a pseudo-NB or velocity map of an object.");sys.exit() 
    try: O = fits.open(args.obj)
    
    except: print(("Error opening object mask: %s"%args.obj));sys.exit()    
    try: objIDs = list( int(x) for x in args.objID.split(',') )
    except: print("Could not parse -objID list. Should be int or comma-separated list of ints.");sys.exit()


    #If object info is loaded - now turn object mask into binary mask using objIDs
    idCube = O[0].data
    if objIDs==[-1]: idCube[idCube>0] = 1
    elif objIDs==[-2]: idCube[idCube>0] = 0
    else:
        for obj_id in objIDs:
            idCube[idCube==obj_id] = -99
        idCube[idCube>0] = 0
        idCube[idCube==-99] = 1
    
    #Crop idCube if cropping was performed on input cube
    if args.centerOn!=None:
        idCubeC = np.zeros((w,bSize,bSize))
        for wi in range(w):
            cutout = Cutout2D(idCube[wi],(xC,yC),bSize,wcs,mode='partial',fill_value=0)
            idCubeC[wi] = cutout.data
        idCube = idCubeC
           
    #Create copy of input cube with non-object voxels set to zero
    objCube = cube.copy()
    objCube[idCube==0] = 0

    #Get 2D mask of useable spaxels        
    msk2D = np.max(idCube,axis=0)
    objNB = np.sum(objCube,axis=0)
            
    #Get 1D wavelength mask
    msk1D = np.max(idCube,axis=(1,2))
    comZ  = CoM(msk1D)[0]
    try: comZ = int(round(comZ))
    except: comZ = w/2
    
    #Now use binary mask to generate the requested data product
    if args.type=='nb' or args.type=='tri':

        #Set noiselayer (-nl) if not set
        if args.nl==-1: args.nl=comZ
  
        stddev = np.std(cube[args.nl])
        
 
        if args.nl!=None: objNB[msk2D==0] = ( cube[args.nl][msk2D==0] )- np.median(cube[args.nl][msk2D==0]) #np.random.normal(0,stddev,objNB[msk2D==0].size) # 

        #Apply spatial smoothing to Narrow-Band Image if option is set
        if args.xySmooth!=None: objNB = astConvolve.convolve(objNB,Kxy)
                               
        nbFITS = fits.HDUList([fits.PrimaryHDU(objNB)])
        nbFITS[0].header = h2D
        nbFITS.writeto(args.cube.replace('.fits','.NB.fits'),overwrite=True)
        print(("Saved %s"%args.cube.replace('.fits','.NB.fits')))

        nbFITS = fits.HDUList([fits.PrimaryHDU(msk2D)])
        nbFITS[0].header = h2D
        nbFITS.writeto(args.cube.replace('.fits','.M2D.fits'),overwrite=True)
        print(("Saved %s"%args.cube.replace('.fits','.M2D.fits')))     
           
    if args.type=='vel' or args.type=='tri':

        #Create canvas for both first and zeroth (f,z) moments
        m0map = np.zeros_like(msk2D,dtype=float)
        m1map = np.zeros_like(m0map)
        
        lineFitter = fitting.SimplexLSQFitter()
        
        #Only make real velmaps if there is an object given
        if np.count_nonzero(idCube)>0:
        
            #Apply spatial smoothing to cube if option is set
            if args.xySmooth!=None:
                for wi in range(w):
                    cube[wi] = astConvolve.convolve(cube[wi],Kxy)

            #Apply wavelength smoothing if option is set
            if args.wSmooth!=None:
                for xi in range(x):
                    for yi in range(y):
                        cube[:,yi,xi] = astConvolve.convolve(cube[:,yi,xi],Kw)
                        
            #Get spaxels that are in 2D mask
            useY,useX = np.where(msk2D==1)
            
            #Run through each
            for j in range(len(useX)):
                
                #Get x,y position
                y,x = useY[j],useX[j]
                
                #Extract object wav-mask
                mskZj = idCube[:,y,x]  

                #Get upper and lower z-indices of this object spectrum  
                zA = np.argmax(mskZj==1)
                zB = zA + np.argmax(mskZj[zA:]==0)
                
                #Expand object mask in z-axis to account for the clipping/thresholding       
                z0 = max(0,zA-args.zWing)
                z1 = min(w-1,zB+args.zWing)

                #Now get spectrum and wavelength axis
                specj = cube[z0:z1,y,x] - np.median(cube[:,y,x])
                wavj  = wav[z0:z1]
                wavjSmooth = np.linspace(wavj[0],wavj[-1],len(wavj)*10)
                
                #Get integrated SNR of spectrum
                specSTD = np.std(cube[:,y,x])
                if specSTD==0: continue
                else:
            
                    if np.sum( specj ) >0:   
                                           
                        #Get zeroth moment
                        
                        #Force positive points only
                        pos = specj>0                       
                        specj2 = specj[pos==1]
                        wavj2 = wavj[pos==1]
                        


                        #Try Gaussian fitting
                        gaussian1 = models.Gaussian1D(amplitude=np.max(specj),mean=4237,stddev=3)
                        gaussian2 = models.Gaussian1D(amplitude=np.max(specj),mean=wavj[np.nanargmax(specj)]+2,stddev=3)
                        lineModel0 = gaussian1 #+ gaussian2
                        
                        #lineModel0.mean_0.min = wavj[1]                        
                        #lineModel0.mean_0.max = lineModel0.mean_1.min
                        #lineModel0.mean_1.min = wavj[1]                    
                        #lineModel0.mean_1.max = wavj[-1]
                        #lineModel0.amplitude_0.min = 0
                        #lineModel0.amplitude_1.min = 0
                        #lineModel0.stddev_0.min = 1
                        #lineModel0.stddev_0.max = 10
                        #lineModel0.stddev_1.min = 1
                        #lineModel0.stddev_1.max = 10
                        lineModel0.mean.min = wav[zA]                    
                        lineModel0.mean.max = wav[zB]
                        lineModel0.amplitude.min = 0
                        lineModel0.stddev.min = 1
                        lineModel0.stddev.max = 10

      
                        lineModel1 = lineFitter(lineModel0,wavj,specj)
                        m0 = lineModel1.mean.value #np.average(wavj2,weights=specj2)
                                                
                        rss = np.sum( (specj-lineModel1(specj))**2 )/(np.std(specj)**2)
                        print(("%5.1f %f"% (m0,rss)))

                        m0map[y,x] = rss
                        
                        
                        m1 = lineModel1.stddev.value #np.sqrt( np.sum( specj*(wavj-m0)**2 )/np.sum(specj) )
                        m1map[y,x] = m1
                                                
                        if 1:
                            fig = plt.figure()
                            ax = fig.add_subplot(111)
                            ax.plot(wavj,specj,'kx-')
                            ax.plot(wavjSmooth,lineModel0(wavjSmooth),'r-',alpha=0.5)
                            ax.plot(wavjSmooth,lineModel1(wavjSmooth),'b-')
                            ax.plot([m0,m0],[np.min(specj),np.max(specj)],'r-')
                            fig.show()
                            time.sleep(0.5)
                            plt.close()

            useM0  = m0map!=0
            useM1  = m1map!=0
            
            if len(m0map[useM0])==0: m0ref = 0
            else:
                m0ref  = np.average(m0map[useM0],weights=objNB[useM0])              
                #m0map[useM0] -= m0ref
                #m0map[useM0] *= (3e5/m0ref)
                
                m1map[useM1] *= (3e5/m0ref)      
                      
            m0map[~useM0] = -5000
            m1map[~useM1] = -5000
            
            if "ZLA" in list(h2D.keys()) and m0ref!=0:
                lyAP = (1+h2D["ZLA"])*1215.7
                lyAQ = (1+h2D["Z"])*1215.7
                m0ref_VelP = 3e5*(m0ref-lyAP)/lyAP
                m0ref_VelQ = 3e5*(m0ref-lyAQ)/lyAQ        
            else:
                m0ref_VelP = None
                m0ref_VelQ = None
                
        #If no objects in input
        else:

            m0map -= 5000
            m1map -= 5000
            #Set blank values for velocity offsets
            m0ref_VelP = None
            m0ref_VelQ = None
            
            
        #Save FITS images
        m0FITS = fits.HDUList([fits.PrimaryHDU(m0map)])
        m0FITS[0].header = h2D
        m0FITS[0].header["BUNIT"] = "km/s"

        m0FITS[0].header["VOFF_P"] = m0ref_VelP
        m0FITS[0].header["VOFF_Q"] = m0ref_VelQ

        m0FITS.writeto(args.cube.replace('.fits','.V0.fits'),overwrite=True)            
        print(("Saved %s"%args.cube.replace('.fits','.V0.fits')))
        
        m1FITS = fits.HDUList([fits.PrimaryHDU(m1map)])
        m1FITS[0].header = h2D
        m1FITS[0].header["BUNIT"] = "km/s"
        m1FITS.writeto(args.cube.replace('.fits','.V1.fits'),overwrite=True)
        print(("Saved %s"%args.cube.replace('.fits','.V1.fits')))
    
    if args.type=='spc' or args.type=='tri':

        #Revert cube to flux/px
        cube *= pxArea

        #Revert to units of 'per Angstrom'
        cube /= h3D["CD3_3"]
   
        #Create cube where non-object spaxels are zeroed out
        cubeT = cube.T.copy()
        cubeT[msk2D.T==0] = 0
        
        #Get object spectrum summed over 2D mask
        objSpc = np.sum(cubeT,axis=(0,1))
        
        #Get upper and lower bounds of 3D mask projected to 1D for this spectrum
        if np.sum(msk1D)>0:
            wavMsk = wav[msk1D>0]       
            mskW0,mskW1 = wavMsk[0],wavMsk[-1]
        else:
            mskW0,mskW1 = 0,0
            
        #Save to FITS and write 1D header
        spcFITS = fits.HDUList([fits.PrimaryHDU(objSpc)])
        spcFITS[0].header = libs.cubes.get1DHeader(h3D)
        spcFITS[0].header["MSKW0"] = mskW0
        spcFITS[0].header["MSKW1"] = mskW1
        spcFITS.writeto(args.cube.replace('.fits','.SPC.fits'),overwrite=True)
        print(("Saved %s"%args.cube.replace('.fits','.SPC.fits')))
        
    
        
