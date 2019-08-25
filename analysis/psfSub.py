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


def run(cubePath,varPath=None,rMin=1.5,rMax=5.0,reg=None,pos=None,auto=None,
        wlWindow=200,localWindow=0,scaleMask=1.0,zMask=(0,0),zUnit='A',
        fileExt=".ps.fits",savePSF=True,saveMask=True ):


    #Try to load the fits file
    try: F = fits.open(cubePath)
    except: print("Error: could not open '%s'\nExiting."%cubePath);sys.exit()

    #Try to parse the wavelength mask tuple
    try: z0,z1 = tuple(int(x) for x in zMask.split(','))
    except: print("Could not parse zmask argument. Should be two comma-separated integers (e.g. 21,32)");sys.exit()

    #Try loading variance cube
    propVar=False
    if varPath!=None:
        try: vFits = fits.open(varPath)
        except: print("Error opening varcube ('%s')" % settings["var"]); sys.exit()
        varcube = vFits[0].data
        propVar=True

    #Open fits image and extract info
    hdr  = F[0].header
    wcs = WCS(hdr)
    pxScales = proj_plane_pixel_scales(wcs)
    in_cube = F[0].data.copy()
    wl_cube = in_cube.copy()
    wl_cube[z0:z1] = 0

    #Convert zmask to pixels if given in angstrom
    if zUnit='A': z0,z1 = libs.cubes.getband(z0,z1,hdr)

    print("""
    CWITools PSF Subtraction
    --------------------------------------
    Input Cube: {0}""".format(cubePath))

    #Create main WL image for PSF re-centering
    wlImg   = np.sum(wl_cube,axis=0)*hdr["CD3_3"]

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


    #Create cube for psfModel
    model = np.zeros_like(in_cube)
    w,y,x = in_cube.shape
    W,Y,X = np.arange(w),np.arange(y),np.arange(x)
    mask  = np.zeros((y,x))
    zeroMask = np.sum(in_cube,axis=0)==0
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
                    layer = np.mean(in_cube[wl1:wl2],axis=0)

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

                    #Propagate error if requested
                    if propVar:
                        varImg = np.sum(varcube[a:b],axis=0)
                        varcube[wi][subPx] += (A**2)*varImg[subPx]


                meanRsSmooth = gaussian_filter1d(meanRs,sigma=3)
                #for wi,meanR in enumerate(meanRsSmooth):
                #        F[0].data[wi][subPx] -= (meanR/np.max(psfImg[subPx]))*psfImg[subPx]

                #Update WL cube and image after subtracting this source
                wl_cube = F[0].data.copy()
                wl_cube[z0:z1] = 0
                wlImg   = np.mean(wl_cube,axis=0)

    outFileName = cubePath.replace('.fits',fileExt)
    F.writeto(outFileName,overwrite=True)
    print("Saved {0}".format(outFileName))

    if savePSF:
        psfOut  = outFileName.replace('.fits','.psfModel.fits')
        psfFits = fits.HDUList([fits.PrimaryHDU(model)])
        psfFits[0].header = hdr
        psfFits.writeto(psfOut,overwrite=True)
        print("Saved {0}.".format(psfOut))

    if saveMask:
        mskOut  = outFileName.replace('.fits','.psfMask.fits')
        psfMask = fits.HDUList([fits.PrimaryHDU(mask)])
        psfMask[0].header = libs.cubes.get2DHeader(hdr)
        psfMask.writeto(mskOut,overwrite=True)
        print("Saved {0}.".format(mskOut))

    if propVar:
        varOut = outFileName.replace('.fits','.var.fits')
        vFits[0].data = varcube
        vFits.writeto(varOut,overwrite=True)
        print("Saved {0}.".format(varOut))


if __name__=="__main__":

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Perform PSF subtraction on a data cube.')
    mainGroup = parser.add_argument_group(title="Main",description="Basic input")
    mainGroup.add_argument('cube',
                        type=str,
                        metavar='cube',
                        help='The cube to be PSF subtracted.'
    )
    srcGroup = parser.add_mutually_exclusive_group(required=True)
    srcGroup.add_argument('-reg',
                        type=str,
                        metavar='path',
                        help='Region file of sources to subtract.',
                        default=None
    )
    srcGroup.add_argument('-pos',
                        type=str,
                        metavar='float tuple',
                        help='Position of source (x,y) to subtract.',
                        default=None
    )
    srcGroup.add_argument('-auto',
                        type=str,
                        metavar='float',
                        help='Automatically detect and subtract sources above this SNR (default: 5).',
                        default=None
    )
    methodGroup = parser.add_argument_group(title="Method",description="Parameters related to PSF subtraction methods.")
    methodGroup.add_argument('-rmin',
                        type=float,
                        metavar='Fit Radius',
                        help='Radius (arcsec) used to FIT the PSF model (default 1)',
                        default=1
    )
    methodGroup.add_argument('-rmax',
                        type=float,
                        metavar='Sub Radius',
                        help='Radius (arcsec) of subtraction area (default 3).',
                        default=1
    )
    methodGroup.add_argument('-scaleMask',
                        type=float,
                        metavar='float',
                        help='Scaling factor for PSF mask (mask radius=S*HWHM).',
                        default=1.0
    )
    methodGroup.add_argument('-wlWindow',
                        type=int,
                        metavar='PSF Window',
                        help='Window (angstrom) used to create WL image of PSF (default 150).',
                        default=150
    )
    methodGroup.add_argument('-localWindow',
                        type=int,
                        metavar='Local PSF Window',
                        help='Use this many extra layers around each wavelength layer to construct local PSF for fitting (default 0 - i.e. only fit to current layer)',
                        default=0
    )
    methodGroup.add_argument('-zMask',
                        type=str,
                        metavar='Wav Mask',
                        help='Z-indices to mask when fitting or median filtering (e.g. \'21,32\')',
                        default='0,0'
    )
    methodGroup.add_argument('-zunit',
                        type=str,
                        metavar='Wav Mask',
                        help='Unit of input for zmask. Can be Angstrom (A) or Pixels (px) (Default: A)',
                        default='A',
                        choices=['A','px']
    )
    methodGroup.add_argument('-recenter',
                        type=str,
                        metavar='Recenter',
                        help='Auto-recenter the input positions using PSF centroid',
                        choices=["True","False"],
                        default="True"
    )
    fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")
    fileIOGroup.add_argument('-var',
                        type=str,
                        metavar='varCube',
                        help='The variance cube associated with input cube - used to propagate error.',
                        default=None
    )
    fileIOGroup.add_argument('-ext',
                        type=str,
                        metavar='File Extension',
                        help='Extension to append to subtracted cube (.ps.fits)',
                        default='.ps.fits'
    )
    fileIOGroup.add_argument('-savePSF',
                        type=str,
                        metavar='Save PSFCube',
                        help='Set to True to output PSF Cube)',
                        choices=["True","False"],
                        default="False"
    )
    fileIOGroup.add_argument('-saveMask',
                        type=str,
                        metavar='Save PSFCube',
                        help='Set to True to output 2D Source Mask',
                        choices=["True","False"],
                        default="True"
    )
    args = parser.parse_args()

    #Convert boolean-like strings to actual booleans
    for x in [args.saveMask,args.savePSF]: x=(x.upper()=="TRUE")

    run(args.cube,
        var=args.var,
        reg=args.reg,
        pos=args.pos,
        auto=args.auto,
        recenter=args.recenter,
        zMask=args.zMask,
        zUnit=args.zUnit,
        wlWindow=args.wlWindow,
        localwindow=args.localWindow,
        savePSF=args.savePSF,
        saveMask=args.saveMask,
        fileExt=args.ext
    )
