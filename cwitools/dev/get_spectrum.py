from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales as getPxScales
from scipy.ndimage.filters import gaussian_filter1d as Gauss1D
from scipy.optimize import differential_evolution

import libs #CWITools import

import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Get summed spectrum within a certain radius of a source.')
mainGroup = parser.add_argument_group(title="Main",description="Basic input")
mainGroup.add_argument('cube',
                    type=str,
                    metavar='cube',
                    help='The input data cube.'
)
mainGroup.add_argument('par',
                    type=str,
                    metavar='params',
                    help='CWITools parameter file for target (used for target position).'
)
mainGroup.add_argument('-r',
                    type=float,
                    metavar='radius',
                    help='Radius within which to sum spectrum.',
                    default=15

)
mainGroup.add_argument('-smooth',
                    type=float,
                    metavar='float',
                    help='Standard deviation of 2D Gaussian spatial smoothing kernel. Default: 1.5px',
                    default=None

)
mainGroup.add_argument('-fitEm',
                    type=str,
                    metavar='wavRange',
                    help='Provide a wavelength range specified as an upper/lower bounds in order to fit an emission feature.',
                    default=None

)
mainGroup.add_argument('-ext',
                    type=str,
                    metavar='str',
                    help='File extension to use for saving spectrum. Default: .rSPC.fits',
                    default='.rSPC.fits'

)
args = parser.parse_args()


#Constants
c = 3e5 #Speed of light in km/s

#Load data
cube,hdr = fits.getdata(args.cube,header=True)

#Get WCS structures
hdr2D = libs.cubes.get2DHeader(hdr)
wcs2D = WCS(hdr2D)
wavAxis = libs.cubes.getWavAxis(hdr)

#Open CWITools params
p = libs.params.loadparams(args.par)

#Get position if not provided as a separate argument
qsoPos = wcs2D.all_world2pix(p["RA"],p["DEC"],0)

#Get meshgrid of distance from source
yy,xx = np.indices(cube[0].shape)
rr = np.sqrt( (yy-qsoPos[1])**2 + (xx-qsoPos[0])**2 )
rMask = rr<=args.r

#Zero-out any spaxels not being summed
cubeT = cube.T.copy()
cubeT[~rMask.T] = 0
spectrum = np.sum(cubeT,axis=(0,1))

#Get number of spaxels
Nspax = np.count_nonzero(rMask)

#Get area of pixel in arcsec 2
pxScls = getPxScales(wcs2D)*3600 #Pixel scales in degrees (x3600 to arcsec)
pxArea = pxScls[0]*pxScls[1] #Pixel size in arcsec2

#Convert spectrum from erg/s/cm2/A to units of erg/s/cm2/arcsec2/A
spectrum /= (Nspax*pxArea)

if args.smooth!=None: spectrum = Gauss1D(spectrum,sigma=libs.science.fwhm2sigma(args.smooth))

#spectrum -= np.median(spectrum)

#fig,ax =plt.subplots(1,1,figsize=(12,6))
#ax.plot(wavAxis,spectrum,'kx-')

if args.fitEm!=None:

    def AIC(data,model,k):
        n = len(data)
        rss = np.sum( (data-model)**2 )
        return n*np.log(rss/n) + 2*k


    w0,w1 = ( float(x) for x in args.fitEm.split(','))
    def line(x,pars): return pars[0]*x + pars[1]
    def quad(x,pars): return pars[0]*(x**2) + pars[1]*x + pars[2]
    def gaussian(x,pars): return pars[0]*np.exp( -((x-pars[1])**2)/(2*pars[2]**2))
    def gaussSumofSquares(params,x,y): return np.sum( (y - gaussian(x,params))**2 )
    def lineSumofSquares(params,x,y): return np.sum( (y - line(x,params))**2 )
    def quadSumofSquares(params,x,y): return np.sum( (y - quad(x,params))**2 )

    gaussianBounds = [(0,np.max(spectrum)*10), (w0,w1), (1,20)]
    gminimized = differential_evolution(gaussSumofSquares,bounds=gaussianBounds,args=(wavAxis,spectrum),seed=2)
    gfitModel = gaussian(wavAxis,gminimized.x)
    gmean,gsig = gminimized.x[1:]

    lineBounds = [(-1,1),(-1,1)]
    lminimized = differential_evolution(lineSumofSquares,bounds=lineBounds,args=(wavAxis,spectrum),seed=2)
    lfitModel = line(wavAxis,lminimized.x)

    quadBounds = [(-10,10),(-10,10),(-10,10)]
    qminimized = differential_evolution(quadSumofSquares,bounds=quadBounds,args=(wavAxis,spectrum),seed=2)
    qfitModel = quad(wavAxis,qminimized.x)

    gAIC = AIC(spectrum,gfitModel,2)
    qAIC = AIC(spectrum,qfitModel,3)

    qVg = qAIC-gAIC

    if qVg>=100: label="Strongly Gaussian"
    elif qVg>=50: label="Moderately Gaussian"
    elif qVg>=20: label="Weak Gaussian"
    else: label="Line"
    ax.plot(wavAxis,gfitModel,'r-')
    ax.plot(wavAxis,qfitModel,'g-')
    ax.plot([gmean,gmean],[0,np.max(gfitModel)],'r-')
    ax.plot([gmean+2*gsig,gmean+2*gsig],[0,np.max(gfitModel)],'r--')
    ax.plot([gmean-2*gsig,gmean-2*gsig],[0,np.max(gfitModel)],'r--')
    #print("Wavelength Center: %8.2f"%mean)
    #print("Full 2-Sigma Size: %8.2f"%(4*gsig))
    print("%s\t%6.2f\t%6.2f\t%10.1f\t%20s"%(p["NAME"].split('_')[0],gmean,gsig*4,qVg,label))
#fig.show()
#raw_input("")
#plt.waitforbuttonpress()

outpath = args.cube.replace('.fits','.SPC.fits')
libs.cubes.saveFITS(spectrum,libs.cubes.get1DHeader(hdr),outpath)
print(outpath)
