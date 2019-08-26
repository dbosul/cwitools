from astropy.nddata import Cutout2D
from astropy.constants import G
from astropy.convolution import Gaussian1DKernel,Box1DKernel
from astropy.cosmology import WMAP9 as cosmo
from astropy.modeling import models,fitting
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales as getPxScales
from scipy.ndimage import gaussian_filter as GaussND
from scipy.stats import sigmaclip
from scipy.optimize import differential_evolution

import astropy.units as u
from . import cubes
import matplotlib.pyplot as plt
import numpy as np


#Replace non-positive values with infinity
def nonpos2inf(cube,level=0): cube[cube<=level] = np.inf

def fwhm2sigma(fwhm): return fwhm/(2*np.sqrt(2*np.log(2)))

#Return a pseudo-Narrowband image (either SB units or SNR)
def pseudoNB(cube,hdr,wav0,var=[],snrMode=False,window=15,wing=20,pos=None,fitRad=2,subRad=None,smoothR=None,wlsub=True,maskPSF=True,weighted=True,kernel='gaussian'):

    #DEBUG
    weighted=False

    #Figure out whether var has been provided or not
    if np.size(var)==0: varIn = False
    else: varIn = True; #nonpos2inf(var)

    #Prep: get some useful structures numbers
    wcs2D = WCS(cubes.get2DHeader(hdr)) #Astropy world-coord sys
    pxScls = getPxScales(wcs2D)*3600 #Pixel scales in degrees (x3600 to arcsec)
    pxArea = pxScls[0]*pxScls[1] #Pixel size in arcsec2
    dwav  = hdr["CD3_3"] #Spectral plate scale in angstrom/px

    #Create plain narrow-band
    A,B = cubes.getband(wav0-window,wav0+window,hdr)
    NB = np.sum(cube[A:B],axis=0)

    #Create correpsonding variance map if cube provided
    if varIn:

        NB_Var = np.sum(var[A:B],axis=0)
        #NB/=np.std(NB/np.sqrt(NB_Var))
        #NB-=np.median(NB)
        #Apply inverse variance weighting
        #if weighted:
        #     NB = NB.copy()/NB_Var.copy()
        #     NB_Var = 1/NB_Var.copy()




    #If requested NB is not in range of cube, return none type
    if B<=0 or A>=cube.shape[0]-1:
        print(("Warning: Requested NB range (%.1f+/-%.1fA) is outside cube bandpass."%(wav0,window)))
        return np.zeros_like(cube[0])
    elif A<0:
        print(("Warning: Requested NB range (%.1f+/-%.1fA) is clipped by cube bandpass."%(wav0,window)))
        A=0

    #fMask is the 2D mask of pixels used for fitting
    fMask = np.zeros_like(cube[0],dtype=bool)
    sMask = np.ones_like(cube[1],dtype=bool)


    #NB -= np.median(NB)
    for yi in range(NB.shape[0]): NB[yi,:] -= np.median(sigmaclip(NB[yi,:],high=2,low=2)[0])
    #for xi in range(NB.shape[1]): NB[:,xi] -= np.median(sigmaclip(NB[:,xi],high=2,low=2)[0])

    #Create white-light image and subtract if requested
    if wlsub:

        #Calculate wing indices and get WL image
        a,b = cubes.getband(wav0-window-wing,wav0+window+wing,hdr)
        WL = np.sum(cube[a:A],axis=0) + np.sum(cube[B:b],axis=0)
        for yi in range(NB.shape[0]): WL[yi,:] -= np.median(WL[yi,:])

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
        #scaleBounds = [(0,1e9)]
        #minimized = differential_evolution(chiSquare,bounds=scaleBounds,args=(WL[fMask],NB[fMask]),seed=2)

        scalingFactors = NB[fMask]/WL[fMask]
        scalingFactors_Clipped = sigmaclip(scalingFactors,high=2.5,low=2.5)
        scalingFactors_Mean = np.median(scalingFactors)
        S = scalingFactors_Mean
        print(scalingFactors)
        #plt.figure();plt.hist(scalingFactors,bins=25);plt.plot([S,S],[0,10]);plt.show()
        sMask[WL<=0] = 0

        #Subtract WL image
        NB[sMask] -=  S*WL[sMask]
        print(np.count_nonzero(fMask))

        if maskPSF: NB[fMask] = 10000
        NB -= np.median(NB)

        #Update variance if working with variance data
        if varIn:

            #Get variance of WL image
            WL_Var = np.sum(var[a:A],axis=0) + np.sum(var[B:b],axis=0)

            #Propagate error accordingly (coefficient squared)
            NB_Var[sMask] += (S**2)*WL_Var[sMask]


    #If smoothing requested
    if smoothR!=None:

        #Apply as two 1D convolutions to NB data
        NBS = smooth3D(NB,smoothR,axes=(0,1),ktype=kernel)

        #Apply (with kernel squared) to variance data if provided
        if varIn:
            NB_VarS1 = smooth3D(NB_Var,smoothR,axes=(0,1),var=False,ktype=kernel)
            NB_VarS2 = smooth3D(NB_Var,smoothR,axes=(0,1),var=True,ktype=kernel)

    else:

        NBS = NB
        if varIn:
            NB_VarS1 = NB_Var.copy()
            NB_VarS2 = NB_Var.copy()



    #If SNR map requested, convert to SNR values and return
    if snrMode:

        #NBS -= np.median(NBS)
        #Calculate SNR
        if varIn and 0:

            nonpos2inf(NB_VarS2,level=1e-6)

            SNR = NBS/np.sqrt(NB_VarS2)

            #Zero-out any infinite variance pixels
            SNR[NB_VarS2==0]=0

        else:
            stdv = np.std(sigmaclip(NBS,high=2.5,low=2.5)[0])
            SNR=NBS/stdv

        #Return SNR map
        return SNR

    #Otherwise, return NB in SB units (mult. by angstrom/px, div. by arcsec2/px)
    else:

        #Re-divide out the variance to recover intensity units
        if varIn and weighted: NB_Final = NBS/NB_VarS1
        else: NB_Final = NBS

        #NB_Final -= np.median(NB_Final)
        #Convert to SB units
        NB_Final *= dwav/pxArea

        #Return SB map
        return NB_Final

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In:
    cmap, name
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(list(zip(k,reverse)))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r

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

    #Make copy - do not modify input cube directly
    cubeFilt = cube.copy()

    #Set kernel type
    if ktype=='box': kernel = Box1DKernel(scale)
    elif ktype=='gaussian': kernel = Gaussian1DKernel(fwhm2sigma(scale))
    else: output("# Mode not found\n");exit()

    #Square kernel if needed
    kernel = np.array(kernel)

    if var==True: kernel = np.power(kernel,2)

    #Apply kernel
    for a in axes: cubeFilt = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=a, arr=cubeFilt.copy())


    #Return
    return cubeFilt

def getBox(fitsIN,paramsIN,boxSize,fill=0):

    wcs2D = WCS(cubes.get2DHeader(fitsIN[0].header))

    xC,yC = wcs2D.all_world2pix(paramsIN["RA"],paramsIN["DEC"],0)

    pkpc_per_px = getPhysicalScalePx(wcs2D,paramsIN["Z"])
    boxSize_px = boxSize/pkpc_per_px

    NBCutout = Cutout2D(fitsIN[0].data,(xC,yC),boxSize_px,wcs2D,mode='partial',fill_value=fill)

    return NBCutout.data

def AIC(data,model,k):
    n = len(data)
    rss = np.sum( (data-model)**2 )
    return n*np.log(rss/n) + 2*k
