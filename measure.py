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
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits 
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from CWITools import libs
from scipy.ndimage.measurements import center_of_mass as CoM

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
objFile = glob.glob("{0}*ps.M.OBJ.fits".format(prodDir))[0]
objFITS = fits.open(objFile)
print("OBJ File: {0}".format(objFile))

#Get Adaptively Smoothed (AKS) input file
aksFile = glob.glob("{0}*ps.bs.M.fits".format(prodDir))[0]
aksFITS = fits.open(aksFile)
print("AKS File: {0}".format(aksFile))

#Get Intensity (INT) input file
intFile = aksFile#glob.glob("{0}*.cs.M.fits".format(prodDir))[0]
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
I_tot = np.zeros_like(Area)
I_peak = np.zeros_like(Area)
Nvoxel = np.zeros_like(Area)

log = open(objFile.replace('.fits','.tab'),'w')
def output(s):
    global log
    print s,
    log.write(s)
    
output("#%19s %8s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n"%("Target","objID","NVox","Area","dWav","lam0_r","R_QSO","Ipeak","Itot","xCoM","yCoM"))

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
    msk_spc = np.sum(OBJ2,axis=(1,2))
    nWav    = np.count_nonzero(msk_spc)
    dWav[i] = nWav*zScale.value
    
    #Isolate object in AKS cube
    AKS2 = AKS.copy()
    AKS2[OBJ!=_id] = 0
    
    #Get XY image and XY Center-of-Mass
    aks_img = np.sum(AKS2,axis=0)
    aks_cXY = CoM(aks_img)
    
    #Add total intensity to array
    I_tot[i] = np.sum(aks_img)
    
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


    #Sum non-zero, box-bound pixels and divide by std deviation of spec
    stddev = np.std(int_spc)
    if stddev==0 or np.isnan(stddev): stddev=np.inf

    #Normalize spectrum to sigma=1
    int_spc /= stddev
        
    #Get bounding box for this object
    where = np.where(msk_spc)
    a,b   = np.min(where), np.max(where)

    #Expand box to allow some flexibility (AKS only picks out brightest part)
    a = max(0,a-3)
    b = min(len(int_spc),b+3)
    
    #Get wings of object 
    a0 = max(0,a-5)
    b1 = min(len(int_spc),b+5)
    
    #Get median from wings
    usePix = np.zeros_like(int_spc,dtype=bool)
    usePix[a0:a] = 1
    usePix[b:b1] = 1
    wingMed = np.median(int_spc[usePix==1])


    #Get pixels to use in summing
    int_spc2 = (int_spc[a:b].copy() - wingMed)
    int_spc2[int_spc2<0] = 0

    I_tot[i] = np.sum(int_spc2) 
    I_peak[i] = np.max(int_spc2)
    
    output("%20s %8i %10i %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n"%(tarName,_id,Nvoxel[i],Area[i],dWav[i],wavCR[i],R_QSO[i],I_peak[i],I_tot[i],cX,cY))

#End timer  
tEnd = time.time()
print("Time Elapsed: %.2f"%(tEnd-tStart))
