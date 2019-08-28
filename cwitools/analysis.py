"""CWITools Data Analysis Functions"""

from .. imports libs

from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.modeling import models,fitting
from scipy.signal import medfilt
from scipy.stats import tstd
from scipy.ndimage.filters import generic_filter


import argparse
import numpy as np
import sys
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import ndimage

import argparse
import numpy as np
import sys

from .. imports libs

from astropy import units as u
from astropy.io import fits
from astropy.modeling import models,fitting
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import SqrtStretch
from photutils import DAOStarFinder
from scipy.stats import sigmaclip
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import argparse
import numpy as np
import pyregion
import sys


def psf_subtract(inpFits, rMin=1.5,rMax=5.0,reg=None,pos=None,
                 auto=None, wlWindow=200,localWindow=0,scaleMask=1.0,
                 zMask=(0,0),zUnit='A'):
    """Models and subtracts point-sources in a 3D data cube.

    Args:
        inpFits (astrop FITS object): Input data cube/FITS.
        rMin (float): Inner radius, used for fitting PSF.
        rMax (float): Outer radius, used to subtract PSF.
        reg (str): Path to a DS9 region file containing sources to subtract.
        pos (float tuple): X,Y position of the source to subtract.
        auto (float): SNR above which to automatically detect/subtract sources.
            Note: One of the parameters reg, pos, or auto must be provided.
        wlWindow (int): Size of white-light window (in Angstrom) to use.
            This is the window used to form a white-light image centered
            on each wavelength layer. Default: 200A.
        localWindow (int): Size of local window (in Angstrom) to use.
            This is the window used around each wavelength layer to form the
            local narrowband image. Default: 0 (i.e. single layer only)
        scaleMask (float): Scaling factor for output PSF mask (Default: 1)
        zMask (int tuple): Wavelength region to exclude from white-light images.
        zUnit (str): Unit of argument zMask.
            Can be Angstrom ('A') or pixels ('px'). Default: 'A'.

    """



    #Open fits image and extract info
    cube  = inpFits[0].data
    w,y,x = cube.shape
    W,Y,X = np.arange(w),np.arange(y),np.arange(x)
    header = inpFits[0].header

    #Create cube for model of psf
    psf_cube = np.zeros_like(cube)
    sub_cube = cube.copy()
    wl_cube = cube.copy()
    wl_cube[z0:z1] = 0
    wlImg   = np.sum(wl_cube,axis=0)

    wcs = WCS(header)
    pxScales = proj_plane_pixel_scales(wcs)





    mask  = np.zeros((y,x))
    zeroMask = np.sum(cube,axis=0)==0
    wav = libs.cubes.getWavAxis(hdr)


    #Convert plate scale to arcseconds
    xScale,yScale = (pxScales[:2]*u.deg).to(u.arcsecond)
    zScale = (pxScales[2]*u.meter).to(u.angstrom)

    #Convert fitting & subtracting radii to pixel values
    rMin_px = rMin/xScale.value
    rMax_px = rMax/xScale.value
    delZ_px = int(round(0.5*wlWindow/zScale.value))

    #Get fitter for PSF fit
    fitter = fitting.LevMarLSQFitter()

    boxSize = 3*int(round(rMax_px))
    yy,xx   = np.mgrid[:boxSize, :boxSize]

    #Get default PSF model for re-centering
    psfModel = models.Gaussian2D(amplitude=1,x_mean=boxSize/2,y_mean=boxSize/2)

    #Get fitter for PSF re-centering
    fitter   = fitting.LevMarLSQFitter()

    #Convert zmask to pixels if given in angstrom
    if zUnit='A': z0,z1 = libs.cubes.getband(z0,z1,hdr)

    print("""
    CWITools PSF Subtraction
    --------------------------------------
    Input Cube: {0}""".format(cubePath))

    #Create main WL image for PSF re-centering


    #Get sources from region file or position input
    sources = []
    if reg!=None:
        print("Region File: %s:"%reg)
        try: regFile = pyregion.open(reg)
        except: print("Error opening region file! Double-check path and try again.");sys.exit()
        for src in regFile:
            ra,dec,pa = src.coord_list
            xP,yP,wP = wcs.all_world2pix(ra,dec,hdr["CRVAL3"],0)
            sources.append((xP,yP))
    elif pos!=None:
        try: pos = tuple(float(x) for x in pos.split(','))
        except: print("Could not parse position argument. Should be two comma-separated floats (e.g. 45.2,33.6)");sys.exit()
        print("Source Position: %.1f,%.1f"%(pos[0],pos[1]))
        sources = [ pos ]
    else:
        print("Automatic Source Finding (python-photutils)")

        auto= float(auto)

        stddev = np.std(wlImg[wlImg<=10*np.std(wlImg)])

        #Run source finder
        daofind  = DAOStarFinder(fwhm=8.0, threshold=auto*stddev)
        autoSrcs = daofind(wlImg)

        #Get list of peak values
        peaks   = list(autoSrcs['peak'])

        #Make list of sources
        sources = []
        for i in range(len(autoSrcs['xcentroid'])): sources.append( (autoSrcs['xcentroid'][i], autoSrcs['ycentroid'][i]) )

        #Sort according to peak value (this will be ascending)
        peaks,sources = zip(*sorted(zip(peaks, sources)))

        #Cast back to list
        sources = list(sources)

        #Reverse to get descending order (brightest sources first)
        sources.reverse()

        print("%i sources detected above SNR threshold of %.1f"%(len(sources),auto))

    print("Zmask (%s): %i,%i"%(zUnit,z0,z1))
    print("--------------------------------------")



    #Define objective function for 2D PSF subtraction optimization
    def psfSub_ObjectiveFunction(params,x,y): return np.sum( (y-params[0]*x)**2 )

    #Run through sources
    for (xP,yP) in sources:

        #Get meshgrid of distance from P
        YY,XX = np.meshgrid(X-xP,Y-yP)
        RR    = np.sqrt(XX**2 + YY**2)

        if np.min(RR)>rMin_px: continue
        else:

            #Get cut-out around source
            psfBox = Cutout2D(wlImg,(xP,yP),(boxSize,boxSize),mode='partial',fill_value=-99).data

            #Get useable spaxels
            fitXY = np.array( psfBox!=-99, dtype=int)

            #Run fit
            psfFit = fitter(psfModel,yy,xx,psfBox,weights=fitXY)

            #Get sigma/fwhm
            xfwhm,yfwhm = 2.355*psfFit.x_stddev.value, 2.355*psfFit.y_stddev.value

            #We take larger of the two for our purposes
            fwhm = max(xfwhm,yfwhm)

            #Only continue with well-fit, high-snr sources
            if 1 or (fitter.fit_info['nfev']<100 and fwhm<10):

                #Update position with fitted center, if user has set recenter to True
                #Note - X and Y are reversed here in the convention that cube shape is W,Y,X
                if recenter=='True': yP, xP = psfFit.x_mean.value+yP-boxSize/2, psfFit.y_mean.value+xP-boxSize/2

                #Update meshgrid of distance from P
                YY,XX = np.meshgrid(X-xP,Y-yP)
                RR    = np.sqrt(XX**2 + YY**2)

                #Get half-width-half-max
                hwhm = fwhm/2.0

                #Add source to mask
                mask[RR<=scaleMask*hwhm] = 1

                #Get boolean masks for
                fitPx = RR<=rMin_px
                subPx = (RR<=rMax_px) & (zeroMask==0)

                meanRs = []
                #Run through wavelength layers
                for wi in range(w):

                    #Get this wavelength layer and subtract any median residual
                    wl1,wl2 = max(0,wi-localWindow), min(w,wi+localWindow)+1
                    layer = np.mean(cube[wl1:wl2],axis=0)

                    #Get upper and lower-bounds for creating WL image
                    a = max(0,wi-delZ_px)
                    b = min(w,a+delZ_px)

                    #Create PSF image
                    psfImg = np.sum(wl_cube[a:b],axis=0)
                    psfImg[psfImg<0]=0

                    scalingFactors = layer[fitPx]/psfImg[fitPx]
                    scalingFactors_Clipped = sigmaclip(scalingFactors,high=3.5,low=3.5)
                    scalingFactors_Mean = np.mean(scalingFactors)
                    A = scalingFactors_Mean
                    if A<0: A=0


                    #Subtract fit from data
                    F[0].data[wi][subPx] -= A*psfImg[subPx]
                    meanR = np.mean(F[0].data[wi][fitPx])

                    #Add to PSF model
                    model[wi][subPx] += A*psfImg[subPx]


                meanRsSmooth = gaussian_filter1d(meanRs,sigma=3)
                #for wi,meanR in enumerate(meanRsSmooth):
                #        F[0].data[wi][subPx] -= (meanR/np.max(psfImg[subPx]))*psfImg[subPx]

                #Update WL cube and image after subtracting this source
                wl_cube = F[0].data.copy()
                wl_cube[z0:z1] = 0
                wlImg   = np.mean(wl_cube,axis=0)

    #Try to load the fits file
    return



def estimate_variance(inpFits,zwindow=10,rescale=True,sigmaclip=4,zmask=(0,0),fmin=0.9,fmax=10):
    """Estimates the 3D variance cube of an input cube.

    Args:
        cubePath (str): Path to the input cube.
        zWindow (int): Size of z-axis bins to use for 2D variance estimation. Default: 10.
        zMask (int tuple): Wavelength layers to exclude while estimating variance.
        rescale (bool): Set to TRUE to perform layer-by-layer rescaling of 2D variance.
        fMin (float): The minimum rescaling factor (Default 0.9)
        fMax (float): The maximum rescaling factor (Default: 10)
        fileExt (str): The extension to use for the output cube (Default .var.fits)

    Returns:
        NumPy ndarray: Estimated variance cube

    """

    cube = inpFits[0].data
    z0,z1 = zmask
    dz = zwindow

    #Output warning
    if z1-z0 >= dz: print("WARNING: Your z-mask is large relative to your zWindow size - this means your variance estimate near the mask may be unreliable. There must be enough non-masked layers in each bin to get a reliable variance estimate.")

    #Parse boolean input
    rescale = True if rescale=="True" else False

    #Run sigma-clip if set
    if sigmaclip>0: cube = sigma_clip(cube,sigma=sigmaclip).data

    #Make first estimate by binning data
    varcube = np.zeros_like(cube)
    i   = 0
    a,b = (i*dz), (i+1)*dz
    while b < cube.shape[0]:
        varcube[a:b] = np.var(cube[a:b],axis=0)
        i+=1
        a,b = (i*dz), (i+1)*dz
    varcube[a:] = np.var(cube[a:],axis=0)

    #Adjust first estimate by rescaling, if set to do so
    if rescale:
        for wi in range(len(varcube)):

            sig = np.sqrt(varcube[wi])

            useXY = sig>0

            varNorm = np.var(cube[wi][useXY]/sig[useXY])

            #Normalize so that variance of layer as a whole is ~1
            #
            # Note: this assumes most of the 3D field is empty of real signal.
            # Z and XY Masks should be supplied if that is not the case
            #

            rsFactor = (1/varNorm)

            rsFactor = max(rsFactor,fmin)
            rsFactor = min(rsFactor,fmax)

            varcube[wi] *= rsFactor

    return varcube


def bg_subtract(inpFits,method='polyfit',polyK=1,medfiltWindow=31,zMask=(0,0),zUnit='A'):
    """
    Subtracts extended continuum emission / scattered light from a cube

    Args:
        cubePath (str): Path to the data cube to be subtracted.
        method (str): Which method to use to model background
            'polyfit': Fits polynomial to the spectrum in each spaxel (default.)
            'median': Subtract the spatial median of each wavelength layer.
            'medfilt': Model spectrum in each spaxel by median filtering it.
            'noiseFit': Model noise in each z-layer and subtract mean.
        polyK (int): The degree of polynomial to use for background modeling.
        medfiltWindow (int): The filter window size to use if median filtering.
        zMask (int tuple): Wavelength region to mask, given as tuple of indices.
        zUnit (str): If using zmask, indices are given in these units.
            'A': Angstrom (default)
            'px': pixels
        saveModel (bool): Set to TRUE to save background model cube.
        fileExt (str): File extension to use for output (Default: .bs.fits)

    """

    #Load header and data
    header = inpFits.header.copy()
    cube   = inpFits.data.copy()
    W      = libs.cubes.getWavAxis(header)
    z,y,x  = cube.shape
    xySize = cube[0].size
    useZ   = np.ones_like(W,dtype=bool)
    maskZ  = False
    modelC = np.zeros_like(cube)

    #Get empty regions mask
    mask2D = np.sum(cube,axis=0)==0

    #Convert zmask to pixels if given in angstrom
    z0,z1 = zMask
    if zUnit=='A': z0,z1 = libs.cubes.getband(z0,z1,header)

    #Subtract background by fitting a low-order polynomial
    if method=='polyfit':

        useZ[z0:z1] = 0
        fitter  = fitting.LinearLSQFitter()
        pModel0 = models.Polynomial1D(degree=polyK)

        #Track progress % using n
        n = 0

        #Run through spaxels and subtract low-order polynomial
        for yi in range(y):
            for xi in range(x):

                n+=1
                p = 100*float(n)/xySize
                sys.stdout.write('%5.2f percent complete\r'%p)
                sys.stdout.flush()

                #Extract spectrum at this location
                spectrum = cube[:,yi,xi].copy()

                #Fit polynomial to data, ignoring masked pixels
                pModel1 = fitter(pModel0,W[useZ],spectrum[useZ])

                #Get background model
                bgModel = pModel1(W)

                if mask2D[yi,xi]==0:

                    cube[:,yi,xi] -= bgModel

                    #Add to model
                    modelC[:,yi,xi] += bgModel

    #Subtract background by estimating it with a median filter
    elif method=='medfilt':

        #Get +/- 5px windows around masked region, if mask is set
        if z1>0:

            #Get size of window region used to interpolate (minimum 5 to get median)
            nw = max(5,(z1-z0))

            #Get left and right index of window regions
            a = max(0,z0-nw)
            b = min(z,z1+nw)

            #Get two z mid-points which we will use for calculating line slope/intercept
            ZA = (a+z0)/2.0
            ZB = (b+z1)/2.0

            maskZ = True

        #Track progress % using n
        n = 0

        for yi in range(y):
            for xi in range(x):
                n+=1
                p = 100*float(n)/xySize
                sys.stdout.write('%5.2f percent complete\r'%p)
                sys.stdout.flush()

                #Extract spectrum at this location
                spectrum = cube[:,yi,xi].copy()

                #Fill in masked region with smooth linear interpolation
                if maskZ:

                    #Calculate slope and intercept
                    YA = np.mean(spectrum[a:z0]) if (z0-a)<5 else np.median(spectrum[a:z0])
                    YB = np.mean(spectrum[z1:b]) if (b-z1)<5 else np.median(spectrum[z1:b])
                    m  = (YB-YA)/(ZB-ZA)
                    c  = YA - m*ZA

                    #Get domain for masked pixels
                    ZZ = np.arange(z0,z1+1)

                    #Apply mask
                    spectrum[z0:z1+1] = m*ZZ + c

                #Get median filtered spectrum as background model
                bgModel = generic_filter(spectrum,np.median,size=medfiltWindow,mode='reflect')

                if mask2D[yi,xi]==0:

                    #Subtract from data
                    cube[:,yi,xi] -= bgModel

                    #Add to model
                    modelC[:,yi,xi] += bgModel

    #Subtract layer-by-layer by fitting noise profile
    elif method=='noiseFit':
        fitter = fitting.SimplexLSQFitter()
        medians = []
        for zi in range(z):

            #Extract layer
            layer = cube[zi]
            layerNonZ = layer[~mask2D]

            #Get median
            median = np.median(layerNonZ)
            stddev = np.std(layerNonZ)
            trimmed_stddev = tstd(layerNonZ,limits=(-3*stddev,3*stddev))
            trimmed_median = np.median(layerNonZ[np.abs(layerNonZ-median)<3*trimmed_stddev])

            medians.append(trimmed_median)
        medians = np.array(medians)
        bgModel0 = models.Polynomial1D(degree=2)
        useZ[z0:z1] = 0
        bgModel1 = fitter(bgModel0,W[useZ],medians[useZ])
        for i,wi in enumerate(W):
            cube[i][~mask2D] -= bgModel1(wi)
            modelC[i][~mask2D] = bgModel1(wi)

    #Subtract using simple layer-by-layer median value
    elif method=="median":

        sigclip = SigmaClip(sigma=2)
        dataclipped = sigclip(cube)
        for zi in range(z):
            medianModel[zi][mask2D==0] = np.median(dataclipped[zi][mask2D==0])
        medianModel = medfilt(medianModel,kernel_size=(3,1,1))
        cube -= medianModel
        modelC = medianModel

    return cube,modelC
