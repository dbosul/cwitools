from astropy.convolution import Gaussian1DKernel
from astropy.cosmology import WMAP9 as cosmo
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales as getPxScales
from scipy.ndimage import gaussian_filter as GaussND
from scipy.optimize import differential_evolution

import astropy.units as u
import cubes
import matplotlib.pyplot as plt
import numpy as np

#Replace non-positive values with infinity
def nonpos2inf(cube): cube[cube<=0] = np.inf

def fwhm2sigma(fwhm): return fwhm/(2*np.sqrt(2*np.log(2)))

#Return a pseudo-Narrowband image (either SB units or SNR)
def pseudoNB(cube,hdr,wav0,var=[],snrMode=False,window=15,wing=20,pos=None,fitRad=2,subRad=None,smoothR=None,wlsub=True,maskPSF=True):

    #Figure out whether var has been provided or not
    if np.size(var)==0: varIn = False
    else:
        varIn = True
        nonpos2inf(var)
        cube = cube/var
        var = 1/var

    #Prep: get some useful structures numbers
    wcs2D = WCS(cubes.get2DHeader(hdr)) #Astropy world-coord sys
    pxScls = getPxScales(wcs2D)*3600 #Pixel scales in degrees (x3600 to arcsec)
    pxArea = pxScls[0]*pxScls[1] #Pixel size in arcsec2
    dwav  = hdr["CD3_3"] #Spectral plate scale in angstrom/px

    #Create plain narrow-band
    A,B = cubes.getband(wav0-window,wav0+window,hdr)
    NB = np.sum(cube[A:B],axis=0)

    #Create correpsonding variance map if cube provided
    if varIn: NB_Var = np.sum(var[A:B],axis=0)


    #If requested NB is not in range of cube, return none type
    if B<=0 or A>=cube.shape[0]-1:
        print("Warning: Requested NB range (%.1f+/-%.1fA) is outside cube bandpass."%(wav0,window))
        return np.zeros_like(cube[0])
    elif A<0:
        print("Warning: Requested NB range (%.1f+/-%.1fA) is clipped by cube bandpass."%(wav0,window))
        A=0

    #fMask is the 2D mask of pixels used for fitting
    fMask = np.zeros_like(cube[0],dtype=bool)
    sMask = np.ones_like(cube[1],dtype=bool)

    #Create white-light image and subtract if requested
    if wlsub:

        #Calculate wing indices and get WL image
        a,b = cubes.getband(wav0-window-wing,wav0+window+wing,hdr)
        WL = np.sum(cube[a:A],axis=0) + np.sum(cube[B:b],axis=0)

        #If source position given, get mask
        if pos!=None:
            yy,xx = np.indices(cube[0].shape)
            rr = np.sqrt( (yy-pos[1])**2 + (xx-pos[0])**2 )
            fMask = rr<=fitRad

            #If subtraction radius given, update sMask
            if subRad!=None: sMask = rr<=subRad

        #Define objective function for WL subtraction optimization
        def chiSquare(params,x,y): return np.sum( (y-params[0]*x)**2 )


        #Define bounds and perform fit
        scaleBounds = [(0,100*np.max(NB,axis=None)/np.max(WL,axis=None))]
        minimized = differential_evolution(chiSquare,bounds=scaleBounds,args=(WL[fMask],NB[fMask]),seed=2)

        #Subtract WL image
        NB[sMask] -=  minimized.x[0]*WL[sMask]

        #Update variance if working with variance data
        if varIn:

            #Get variance of WL image
            WL_Var = np.sum(var[a:A],axis=0) + np.sum(var[B:b],axis=0)

            #Propagate error accordingly (coefficient squared)
            NB_Var += (minimized.x[0]**2)*WL_Var

    #If smoothing requested
    if smoothR!=None:

        #Get Gaussian Kernel
        K = Gaussian1DKernel(fwhm2sigma(smoothR)).array

        #Apply as two 1D convolutions to NB data
        NBS = smooth3D(NB,smoothR,axes=(0,1))

        #Apply (with kernel squared) to variance data if provided
        if varIn:
            NB_VarS1 = smooth3D(NB_Var,smoothR,axes=(0,1),var=False)
            NB_VarS2 = smooth3D(NB_Var,smoothR,axes=(0,1),var=True)

    else:

        NBS = NB
        NB_VarS1 = NB_Var
        NB_VarS2 = NB_Var

    #If no variance provided, just use estimate scalar variance from the data now
    if not varIn: NB_Var = np.var(NB)

    #Mask pixels used for fitting if requested
    if maskPSF: NB[fMask]=0

    #If SNR map requested, convert to SNR values and return
    if snrMode:

        #Calculate SNR
        SNR = NBS/np.sqrt(NB_VarS2)

        #Zero-out any infinite variance pixels
        SNR[NB_VarS2==0]=0

        #Return SNR map
        return SNR

    #Otherwise, return NB in SB units (mult. by angstrom/px, div. by arcsec2/px)
    else:

        #Re-divide out the variance to recover intensity units
        if varIn: NB_Final = NBS/NB_VarS1
        else: NB_Final = NBS
        #Convert to SB units
        NB_Final *= dwav/pxArea

        #Return SB map
        return NB_Final

def getPhysicalScalePx(wcs2D,redshift):

    #Get platescale in arcsec/px (assumed to be 1:1 aspect)
    pxScale = getPxScales(wcs2D)[0]*3600

    #Get pkpc/arcsec from cosmology
    pkpcScale = cosmo.kpc_proper_per_arcmin(redshift)/60.0

    #Get pkpc/pixel by combining
    pkpc_per_px = (pkpcScale*pxScale).value

    return pkpc_per_px

#Function to smooth along wavelength axis
def smooth3D(cube,scale,axes=(0,1,2),ktype='gaussian',var=False):

    #Make copy
    cubeFilt = cube.copy()

    #Set kernel type
    if ktype=='box': kernel = Box1DKernel(scale)
    elif ktype=='gaussian': kernel = Gaussian1DKernel(fwhm2sigma(scale))
    else: output("# Mode not found\n");exit()

    #Square kernel if needed
    kernel = np.array(kernel)
    if var==True: kernel = np.power(kernel,2)

    #Apply kernel
    for a in axes: cubeFilt = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=a, arr=cubeFilt)


    #Return
    return cubeFilt
