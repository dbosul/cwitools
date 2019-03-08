# 
# MEASURE - take in .OBJ.fits, an intensity input cube (.AKS.fits or other) and measure object properties
#
# Rough algorithm: use the parameter file, glob the other input files, 
# loop through OBJ_ID and do basic measurements of ea
#

#Timer start
import time
tStart = time.time()

import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

from astropy import units as u
from astropy.convolution import convolve, Box1DKernel
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits 
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from CWITools import libs
from scipy.ndimage.measurements import center_of_mass as CoM
from skimage import measure

parFile = sys.argv[1]
targPar = libs.params.loadparams(parFile)

#Get product directory for this target
prodDir = targPar["PRODUCT_DIR"]
tarName = prodDir.split('/')[-2]

#Get redshifts
qso_zla = targPar["ZLA"]
qso_z   = targPar["Z"]

#Get physical distance/arcsec at this redshift
pkpc_arcmin = cosmo.kpc_proper_per_arcmin(qso_zla)

#Get QSO RA/DEC
qsoRA  = targPar["RA"]
qsoDEC = targPar["DEC"]
qsoCoord = SkyCoord(qsoRA*u.deg,qsoDEC*u.deg)

#Get Object ID (OBJ) input file
objFile = glob.glob("{0}*ps.bs.M.AKS.OBJ.fits".format(prodDir))[0]
objFITS = fits.open(objFile)
print("OBJ File: {0}".format(objFile))

#Get Adaptively Smoothed (AKS) input file
aksFile = glob.glob("{0}*ps.bs.M.AKS.fits".format(prodDir))[0]
aksFITS = fits.open(aksFile)
print("AKS File: {0}".format(aksFile))

#Get Intensity (INT) input file
intFile = glob.glob("{0}*.ps.bs.M.fits".format(prodDir))[0]
intFITS = fits.open(intFile)
print("INT File: {0}".format(intFile))

#Get 2D and 3D headers
aksHead3D = aksFITS[0].header
aksHead2D = libs.cubes.get2DHeader(aksHead3D)

#Get Wavelength axis for these cubes
wav = libs.cubes.getWavAxis(aksHead3D)

#Get Astropy WCS objects
wcs3D = WCS(aksHead3D)
wcs2D = WCS(aksHead2D)

qsoY,qsoX = wcs2D.all_world2pix(qsoRA,qsoDEC,0)

#Get projected pixel scales
xScale,yScale,zScale = proj_plane_pixel_scales(wcs3D)
xScale = (xScale*u.deg).to(u.arcsec)
yScale = (yScale*u.deg).to(u.arcsec)
zScale = (zScale*u.meter).to(u.angstrom)
pxArea = xScale*yScale

#Get list of OBJ_IDS
OBJ = objFITS[0].data
IDS = np.unique(OBJ)
IDS = IDS[IDS>0]

#Get short reference for AKS data
AKS = aksFITS[0].data
INT = intFITS[0].data
INT_T = INT.T.copy() #Transpose for indexing later

#Iterate over objects and make measurements
Area  = np.zeros_like(IDS,dtype=float)
dWav  = np.zeros_like(Area)
wavCR = np.zeros_like(Area)
R_QSO = np.zeros_like(Area)
I_rel = np.zeros_like(Area)
I_int = np.zeros_like(Area)
Nvoxel = np.zeros_like(Area)

log = open(objFile.replace('.fits','.tab'),'w')
def output(s):
    global log
    print s,
    log.write(s)
    
output("#%19s %8s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n"%("Target","objID","NVox","Area","dWav","lam0_r","R_QSO","I_int","I_rel","M","Disp"))

Nz = OBJ.shape[0]
ZKern  = Box1DKernel(3)
    
for i,_id in enumerate(IDS):

    sys.stdout.write('{0}/{1}\r'.format(i+1,len(IDS)))
    sys.stdout.flush()   
     
     
    #Isolate object in mask cube
    OBJ2 = OBJ.copy()
    OBJ2[OBJ!=_id] = 0
    
    #Get Number of Voxels
    Nvoxel[i] = np.count_nonzero(OBJ2)
    
    #Get XY Mask and Npix/Area
    msk_img = np.sum(OBJ2,axis=0)
    nArea = np.count_nonzero(msk_img)
    Area[i] = nArea*pxArea.value    

    #Get Z Mask and Npix/Wav
    msk_spc = np.max(OBJ2,axis=(1,2))
    nWav    = np.count_nonzero(msk_spc)
    dWav[i] = nWav*zScale.value
    
    #Isolate object in AKS cube
    AKS2 = AKS.copy()
    AKS2[OBJ!=_id] = 0
    
    #Get XY image and XY Center-of-Mass
    aks_img = np.sum(AKS2,axis=0)
    aks_cXY = CoM(aks_img)
    
    #Convert XY CoM into RA/DEC
    cX,cY = aks_cXY
    cRA,cDEC = wcs2D.all_pix2world(cY,cX,0)
    
    #Get SkyCoord of object and calculate distance to QSO
    cCoord = SkyCoord( cRA*u.deg,cDEC*u.deg)
    dstQSO = cCoord.separation(qsoCoord).arcmin*u.arcmin*pkpc_arcmin

    R_QSO[i] = dstQSO.value
    
    #Get Z spectrum and Z Center-of-Mass
    aks_spc = np.sum(AKS2,axis=(1,2))
    aks_cZ  = CoM(aks_spc)[0]
    
    #Convert Z-com into central wavelength in observer frame
    cZ_obs = wav[int(round(aks_cZ))]
    
    #Convert cZ_observer frame into restframe using QSO LyA Peak
    cZ_res = cZ_obs/(1+qso_zla)
    wavCR[i] = cZ_res
    
    #Get optimally extracted spectrum for this object
    INT_T2 = INT_T.copy()
    INT_T2[aks_img.T<=0] = 0
    int_spc = np.sum(INT_T2,axis=(0,1))
    int_spc = convolve(int_spc,ZKern)
    
    #Calculate relative integrated SNR
    w0s,I0s = [],[]
    disps = []
    _spcCopy = int_spc.copy()
    _spcCopy[_spcCopy>0] = 1
    _spcCopy[_spcCopy<0] = 0
    labels = measure.label(_spcCopy,background=0,connectivity=1)
    labUnique = np.unique(labels[labels>0])
    for l in labUnique:
    
        #Get mask of this 1d object
        lmsk = labels==l
        inds = np.where(lmsk)
        
        #Get shortened obj spec and wav
        ospc = int_spc[lmsk]
        owav = wav[lmsk]
        
        
        if np.sum(ospc)==0: continue
        

        #Get flux-weighted centroid
        w0l = np.average(owav,weights=ospc)
        
        #Calc summed Intensity
        I0l = np.sum( ospc )

        #Get dispersion
        v0l = np.sqrt( np.sum( ospc*(owav-w0l)**2 )/np.sum(ospc))
        
        #Convert to km/s
        v0l *= (3e5/w0l)
        
        w0s.append( w0l )
        I0s.append( I0l )            
        disps.append( v0l )

    I0s = np.array(I0s)

    I0s_rel = I0s/np.std(sigma_clip(I0s,sigma=5))
    

    M = np.count_nonzero(I0s_rel>5)       

    w0s_old = np.array(w0s).copy()
    I0s_rold = I0s_rel.copy()
    
    close = np.abs(w0s-cZ_obs)<5
    #I0s = I0s[ close ]
    #I0s_rel = I0s_rel[ close ]
    #w0s = np.array(w0s)[ close ]    
    #disps = np.array(disps)[ close ]

    if len(I0s_rel)>0:
        arg = np.argmax(I0s_rel)
        
        I_rel[i] = I0s_rel[arg]
        I_int[i] = I0s[arg]
        disp     = disps[arg]
    else:
        I_rel[i] = 0
        I_int[i] = 0
        disp     = 0 
            
    output("%20s %8i %10i %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10i %10.2f\n"%(tarName,_id,Nvoxel[i],Area[i],dWav[i],wavCR[i],R_QSO[i],I_int[i],I_rel[i],M,disp))

    if 0 and _id==307:
        plt.figure()
        plt.subplot(211)
        plt.plot(wav,int_spc,'k-')
        plt.plot(wav[msk_spc>0],int_spc[msk_spc>0],'r-')
        plt.xlim([wav[0],wav[-1]])
        plt.subplot(212)
        plt.plot(w0s_old,I0s_rold,'ko')
        plt.xlim([wav[0],wav[-1]])
        plt.show()  
#End timer  
tEnd = time.time()
print("Time Elapsed: %.2f"%(tEnd-tStart))
