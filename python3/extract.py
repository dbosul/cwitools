### EXTRACT - Pull 3D objects out of the input cube
#
# Algorithm:
# 1. Adaptive Smoothing: Smooth the data with increasingly large box kernels. Extract & subtract detections at each stage
# 2. Segmentation: Threshold the smoothed data (>0) and divide into 3D regions
# 3. Thresholding: Reject objects below a certain size in voxels
# 4. Repair: Smooth the 3D mask with a (small) kernel to reconnect noisier parts of the mask
#
# SYNTAX:
# > python extract.py <cube> [varCube=..., kMin=..., kMax=... etc.]
# 
# OPTIONAL ARGUMENTS:
# varCube - point the algorithm to a 


from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from scipy.signal import fftconvolve
from skimage import measure

import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings

#Take any additional input params, if provided
settings = {"repair":True,"repairRadius":3,"snrMin":3,"nVoxMin":1000,"outPath":None,"var":None,"boxFilt":1,"adaptive":False}
if len(sys.argv)>2:
    for item in sys.argv[2:]:      
        key,val = item.split('=')
        if key in settings:
            if key in ["kMax","kMin","kStep","repairRadius","nVoxMin","boxFilt"]: val=int(val)
            elif key in ["snrMin"]: val=float(val)
            elif key in ["repair","adaptive"]:
                if val=="False" or val==0: val=False
                else: val=True
            settings[key]=val
        else:
            print("Input argument not recognized: %s" % key)
            sys.exit()

#Take input
fitsPath = sys.argv[1]
snrMin   = settings["snrMin"]
nVoxMin  = settings["nVoxMin"]
R        = settings["repairRadius"]
doRepair = settings["repair"]
outPath  = settings["outPath"]
adaptive = settings["adaptive"]

print("\nCWITools Extraction")
print("------------------------------")
print(("Input cube: %s"%fitsPath))
print(("Minimum SNR/vox: %.1f"%snrMin))
print(("Mask repair radius: %i"%R))
print("------------------------------")

#Load data
F = fits.open(fitsPath)
I = F[0].data
Iw = I.copy()
D = np.zeros_like(I,dtype=float)
I0 = np.ones_like(I,dtype=float)

#Create Kernels
repairKernel = np.ones((R,R,R))/R**3

k = settings["boxFilt"]
K = np.ones((k,k,k))/k**3

if k>1: I = fftconvolve(I,K,mode='same')

#Get or estimate noise
if settings["var"]==None: sigma=np.std(I)
else:
    sigma = fits.open(settings["var"])[0].data
    if k>1 and (not adaptive): sigma = fftconvolve(sigma,K**2,mode='same')

## STEP 1: EXTRACTION OF VOXELS>sigMin
print(("Extracting voxels > %2.1f sigma..."%snrMin), end=' ')

if not adaptive:
    detections = I/sigma > snrMin
    D[detections] = I[detections]
else:
    
    Iw = I.copy()
    Vw = sigma.copy()
    
    for k in [2,3,4]:
        K = np.ones((k,k,k))/k**3

        Ib = fftconvolve(Iw,K,mode='same')
        Vb = fftconvolve(Vw,K,mode='same')
        
        #print np.mean(Ib), np.mean(Vb)
        detections = np.sqrt(K.size)*Ib/np.sqrt(Vb) > snrMin

        D[detections] += Ib[detections]
        Iw[detections] -= Ib[detections]
    
    
## STEP 2: SEGMENTATION
M = np.zeros_like(D)
M[D>0] = 1
L = measure.label(M)
U = np.unique(L[L>0])
try: print(("%i objects found."%np.max(U)))
except:
    print("No objects found.")
    sys.exit()

## STEP 3: THRESHOLD BY VOXEL COUNT
print(("Thresholding at nVoxels>%i ..."%nVoxMin), end=' ')
sys.stdout.flush()
n=0
for label in U:
    nVox = np.count_nonzero(L==label)
    if nVox<=nVoxMin:
        D[L==label] = 0
        L[L==label] = 0
    else:
        n+=1
        L[L==label] = n
print(("%i objects remaining."%n))

## STEP 4: REPAIR 3D MASKS
if doRepair:
    print(("Repairing masks..."), end=' ')
    L = np.array(L,dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L2 = fftconvolve(L,repairKernel,mode='same')
    L2[L2<0.1] = 0
    L2[L2>0.1] = 1
    LR = measure.label(L2)
    print(("%i objects after repairing."%np.max(LR)))
else: LR = L

print("------------------------------")
if outPath==None: outPath=fitsPath.replace('.fits','.OBJ.fits')
F[0].data = LR
F.writeto(outPath,overwrite=True)   
print(("Saved %s"%outPath))

