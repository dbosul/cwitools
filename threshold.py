from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib  import cm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.measurements import center_of_mass
from statsmodels.stats.weightstats import DescrStatsW

import matplotlib.pyplot as plt
import numpy as np
import sys

import CWITools.libs as libs

plt.style.use('ggplot')

R = 2500 #Medrez spectral res.
lyA = 1215.7

inCube = sys.argv[1]
idCube = sys.argv[2]
z = float(sys.argv[3])

w0 = (z+1)*lyA
c  = 3e10
dv = 2e8

dw = (dv/c)*w0
print w0-dw,w0+dw
targname = inCube.split('.')[0]
objFileOut = open(inCube.replace('.fits','.DET.tab'),'w')
objFileString = "%10s,%10s,%10s,%10s,%10s,%10s\n" % ("ID","NVoxels","Area","WavExtent","SNR","SUM")

inFITS = fits.open(inCube)
idFITS = fits.open(idCube)

inC = inFITS[0].data
idC = idFITS[0].data
outC = np.zeros_like(idC)

W = libs.cubes.getWavAxis(inFITS[0].header) #Wavelength axis

wcs = WCS(inFITS[0].header)
pxScales = proj_plane_pixel_scales(wcs)

# Get pixel scales in Angstrom & Arcseconds
pxDLam = pxScales[2]*1e10
pxArea = (pxScales[0]*pxScales[1])*(3600**2)

#Area (px) and wavelenght extend (px) below which to reject objects
areaThresh = np.pi
wavThresh  = inFITS[0].header["CRVAL3"]/R

IDS = np.unique(idC[idC>0])

inC *= 1e2

n = 1
Areas,WavExts = [],[]
SBs = []
for Id in IDS:

    sys.stdout.write('{0}/{1}\r'.format(Id,len(IDS)))
    sys.stdout.flush()
    inC2 = inC.T.copy()
    inC2[idC.T!=Id] = 0

    idImg = np.sum(inC2,axis=2)
    
    idSpc = np.sum(inC2,axis=(0,1))
    idSpc = gaussian_filter1d(idSpc,1.0)
    
    V = np.count_nonzero(inC2)
    A = np.count_nonzero(idImg)*pxArea
    L = np.count_nonzero(idSpc)*pxDLam

    Lcenter = int(round(center_of_mass(idSpc)))
    Wcenter = W[Lcenter]
    
    Areas.append( A )
    WavExts.append( L  )

    weighted_stats = DescrStatsW(W, weights=idSpc, ddof=0)
    wMean = weighted_stats.mean
    wDisp = weighted_stats.std
    I = np.sum(inC2)

    #Add this to new ID cube if it meets thresholds and is in correct wavelength range
    if A>areaThresh and L>wavThresh and abs(wCenter-w0)<dw:
        outC[idC==Id] = n
        objFileString+="%10i,%10i,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f\n" % (n,V,A,L,wMean,wDisp,I)
        n+=1
        
print(objFileString),
objFileOut.write(objFileString)

fig = plt.figure(figsize=(12,8))
ax  = fig.add_subplot(111)
ax.scatter( WavExts, Areas, s=100, c=SBs, marker='o', cmap = cm.jet, vmin=-2, vmax=2)
ax.plot( [wavThresh,wavThresh], [1e-2,1e5], 'b--')
ax.plot( [0,100], [areaThresh,areaThresh], 'b--', label="Resolution Element")
ax.set_ylim( [0.99,1000] )
ax.set_xlim( [0,6] )
ax.set_xlabel(r"Wavelength Extent ($\AA$)",fontsize=18)
ax.set_ylabel(r"Surface Area (arcsec$^2$)",fontsize=18)
ax.set_title(r"Object Detections in %s" % targname,fontsize=18)
ax.set_yscale('log')
plt.legend(loc=4,fontsize=16)
plt.tight_layout()
plt.savefig(sys.argv[1].replace('.fits','.TH.png'))
#fig.show()
#aw_input("")
idFITS[0].data = outC
idFITS.writeto(idCube.replace('.fits','.T.fits'),overwrite=True)
