from astropy.io import fits
from astropy.convolution import Box1DKernel,Gaussian1DKernel,convolve_fft
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d,uniform_filter1d
from scipy.signal import boxcar,gaussian,medfilt

import argparse
import numpy as np
import scipy
import sys
import time

import libs

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Perform Adaptive-Kernel-Smoothing on a data cube (requires variance cube).')
mainGroup = parser.add_argument_group(title="Main",description="Basic input")
mainGroup.add_argument('cube', 
                    type=str, 
                    metavar='input cube',             
                    help='The cube to be smoothed.'
)
mainGroup.add_argument('var', 
                    type=str, 
                    metavar='variance',             
                    help='The associated variance cube.'
)
methodGroup = parser.add_argument_group(title="Method",description="Smoothing parameters.")
methodGroup.add_argument('-snr',
                    type=float,
                    metavar='float',
                    help='The objective signal-to-noise level (Default:3)',
                    default=3
)
methodGroup.add_argument('-xyMode',
                    type=str,
                    metavar='str',
                    help='Spatial moothing mode (box/gaussian) - Default: gaussian',
                    default='gaussian',
                    choices=['box','gaussian']
)
methodGroup.add_argument('-wMode',
                    type=str,
                    metavar='str',
                    help='Wavelength moothing mode (box/gaussian) - Default: box',
                    default='gaussian',
                    choices=['box','gaussian']
)
methodGroup.add_argument('-xyScale0',
                    type=float,
                    metavar='float (px)',
                    help='Minimum spatial smoothing scale (Default:3)',
                    default=3
)
methodGroup.add_argument('-wScale0',
                    type=float,
                    metavar='float (px)',
                    help='Minimum wavelength smoothing scale (Default:2)',
                    default=2
)
methodGroup.add_argument('-xyScale1',
                    type=float,
                    metavar='float (px)',
                    help='Maximum spatial smoothing scale (Default:10)',
                    default=10
)
methodGroup.add_argument('-wScale1',
                    type=float,
                    metavar='float (px)',
                    help='Maximum wavelength smoothing scale (Default:5)',
                    default=5
)
methodGroup.add_argument('-xyStep0',
                    type=float,
                    metavar='float (px)',
                    help='Minimum spatial scale step-size (Default:0.1px)',
                    default=0.1
)
methodGroup.add_argument('-wStep0',
                    type=float,
                    metavar='float (px)',
                    help='Minimum wavelength scale step-size (Default:0.5px)',
                    default=0.5
)

fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")

fileIOGroup.add_argument('-ext',
                    type=str,
                    metavar='str',
                    help='Extension to append to subtracted cube (.AKS.fits)',
                    default='.AKS.fits'
)
fileIOGroup.add_argument('-saveXYKer',
                    type=str,
                    metavar='bool',
                    help='Save XY kernel output cubes (True/False)',
                    choices=["True","False"],
                    default="False"
)
fileIOGroup.add_argument('-extXYKer',
                    type=str,
                    metavar='str',
                    help='Extension to append to Kernel cube (.AKS.kXY.fits)',
                    default='.AKS.kXY.fits'
)
fileIOGroup.add_argument('-saveWKer',
                    type=str,
                    metavar='bool',
                    help='Save W kernel output cubes (True/False)',
                    choices=["True","False"],
                    default="False"
)
fileIOGroup.add_argument('-extWKer',
                    type=str,
                    metavar='str',
                    help='Extension to append to Kernel cube (.AKS.kW.fits)',
                    default='.AKS.kW.fits'
)
fileIOGroup.add_argument('-saveSNR',
                    type=str,
                    metavar='bool',
                    help='Save SNR output cubes (True/False)',
                    choices=["True","False"],
                    default="False"
)
fileIOGroup.add_argument('-extSNR',
                    type=str,
                    metavar='str',
                    help='Extension to append to Kernel cube (.AKS.SNR.fits)',
                    default='.AKS.SNR.fits'
)
args = parser.parse_args()


logFile=None
def output(s):
    global logFile
    print(s),
    logFile.write(s)

def exit():
    global logFile
    logFile.close()
    sys.exit()

def fwhm2sigma(fwhm): return fwhm/(2*np.sqrt(2*np.log(2)))     

#Function to smooth along wavelength axis
def wavelengthSmooth(a,scale,var=False):
    global args
  
    if args.wMode=='box': K = Box1DKernel(scale)
    elif args.wMode=='gaussian': K = Gaussian1DKernel(scale/2.355)
    else: output("# Mode not found\n");exit()

    aFilt = np.apply_along_axis(lambda m: np.convolve(m, K, mode='same'), axis=0, arr=a)
    #aFilt /= np.max(K.array) #Undo kernel normalization
    return aFilt,K.array
    
#Function to smooth along two spatial axes
def spatialSmooth(a,scale,var=False):
    global args
    #if mode=='box': K = Box1DKernel(fwhm2sigma
    #elif mode=='gaussian': #aFilt = gaussian_filter1d(gaussian_filter1d(a,scale/2.355,axis=1,mode='constant'),scale/2.355,axis=2,mode='constant')
    #else: output("# Mode not found\n");exit() 
    K = Gaussian1DKernel(fwhm2sigma(scale)) 
  
    aFiltX = np.apply_along_axis(lambda m: np.convolve(m, K, mode='same'), axis=2, arr=a)
    aFiltXY = np.apply_along_axis(lambda m: np.convolve(m, K, mode='same'), axis=1, arr=aFiltX)

    return aFiltXY,K.array
              
#Load parameters

Ifile = args.cube
Vfile = args.var
logFile = open(Ifile.replace('.fits','.AKS.log'),'w')

output("# Input intensity data: %s\n" % Ifile)
output("# Input variance data: %s\n" % Vfile)
output("# XY Smoothing mode: %s\n" % args.xyMode)
output("# Wav Smoothing mode: %s\n" % args.wMode)

#Open input intensity cube
try: fI = fits.open(Ifile)
except: output("# Error opening file %s. Please check and try again.\n"%Ifile);exit()

#Open input variance cube
try: fV = fits.open(Vfile)
except: output("# Error opening file %s. Please check it exists and try again, or set variance input with 'varcube='.\n"%Vfile);exit()


## VARIABLES & DATA STRUCTURES

I = fI[0].data.copy()   #Original intensity cube

V = fV[0].data.copy()   #Original variance cube

D = np.zeros_like(I)    #Detection cube

M = np.zeros_like(I)    #For masking pixels after detection
 
kXY = np.zeros_like(I)    #Keep track of kernel sizes used
kW  = np.zeros_like(I)

T = np.zeros_like(I)    # SNR Cube

shape = I.shape         #Data shape

xyScale0 = args.xyScale0          #Establish minimum smoothing scales
wScale0 = args.wScale0

xyStep0 = args.xyStep0           #Establish default step sizes
wStep0 = args.wStep0

xyScale1 = args.xyScale1  #Establish maximum smoothing scales
wScale1 = args.wScale1

xyStepMin = args.xyStep0                #Establish minimum step sizes
wStepMin = args.wStep0

tau_min = float(args.snr) #Minimum signal-to-noise threshold
tau_max = tau_min*1.1
tau_mid = (tau_min+tau_max)/2.0

## PRE-PROCESSING FOR MAIN LOOP

#M[I==0] = 1   #Mask Zero values in cube

N0 = np.sum(M) #Get Npix that are masked by default (to distinguish from detections)

#Initialize w loop variables
wScale = wScale0  
wStep = wStep0  

#Keeping track of how many steps you have had no detections
n_under = 0

t1 = time.time()

## MAIN LOOP
output("# %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n" % ('wScale','wStep','xyScale','xyStep','Npix','% Done','minSNR','medSNR','maxSNR','mid/med')  )  
while wScale <= wScale1: #Run through wavelength bins

    #Initialize xy loop variables
    xyScale = xyScale0
    xyStep = xyStep0
    
    #Back up old scale and step size
    xyScale_old = xyScale 
    xyStep_old = xyStep 

    #Wavelength smooth intensity and variance data
    I_w,kw = wavelengthSmooth(I,wScale)
    V_w,kw = wavelengthSmooth(V,wScale,var=True)


    while xyScale <= xyScale1:
        
        #Output first half of diagnostic info
        output("%8.2f %8.3f %8.2f %8.3f" % (wScale,wStep,xyScale,xyStep))  
           
        #Start off boolean to flag detections
        detections=False

        #Spatially smooth intensity and variance data
        I_xy,kxy = spatialSmooth(I_w,xyScale)
        V_xy,kxy = spatialSmooth(V_w,xyScale,var=True)
        
        I_xy -= np.median(I_xy)
        
        #Prevent divison by zero and zero out any undefined SNRs 
        V_xy[V_xy<=0] = np.inf
        
        #Get SNR array
        SNR = (I_xy/np.sqrt(V_xy))
        

        # Since the smoothing kernels are normalized, we need to
        # artificially increase the SNR to reflect a hypothetical value
        # e.g. 'if we summed the data over this kernel, the SNR would be...'
        Nxy = np.sum(kxy/np.max(kxy))
        Nw  = np.sum(kw/np.max(kw))
        
        #Factor of pi/4 represents circular shape of kernel (not square)
        SNR *= np.sqrt(Nxy*Nxy*Nw)*(np.pi/4)
        
        #Get indices of detections
        indices = SNR >= tau_min
        indices[M==1] = 0

        #Get SNR values and total # of new detections
        SNRS = SNR[indices]
        Npix = len(SNRS)

        #If there are no detections
        if Npix==0:
        
            n_under+=1
            
            f = '-1' #Set fraction to null
            if n_under>0: xyStep += xyStepMin #Increase step size if we are lagging behind
            xyScale += xyStep #Increase scale          
            
              
        #If there are detections (Npix has to be >=0 as it is a len() call)
        else:

            #Reset non-detections counter
            n_under = 0
            
            #Raise detections flag
            detections=True
            
            #If a median is well defined
            if Npix>=5:
            
                #Get the fractional value
                f = round((tau_mid/np.median(SNRS) - 1),2)

                #If there is no oversmoothing OR we are at the smallest scales already
                if f>=0 or xyScale<=xyScale0 or xyScale<=xyScale_old+xyStepMin:


                    #Back-up old step size and scale
                    xyStep_old = xyStep
                    xyScale_old = xyScale
                    
                    if f>0:
                    
                        #Calculate new step size and scale using f
                        xyScale = (1+f)*xyScale_old
                        xyStep  = xyScale - xyScale_old
                        
                        #Floor this process at the minimum smoothing scale
                        if xyStep<=xyStepMin: xyStep=xyStepMin

                    elif f==0:
                    
                        #Update spatial scale and keep same step size               
                        xyScale += xyStep

                    elif xyScale<=xyScale0:
                    
                        #Back up old scale?
                        xyStep_old = xyStep
                        xyScale_old = xyScale

                        #Reduce step size and move forward
                        xyStep = xyStep/2.0
                        xyScale += xyStep    
                                            
                        #Floor this process at the minimum smoothing scale
                        if xyStep<=xyStepMin: xyStep=xyStepMin                                                      

                #If oversmoothing is evident (i.e if f<0 and we are above minimum spatial scale)
                else:

                    #Go back half a step in scale
                    xyScale = (xyScale_old + xyScale)/2.0
                    
                    #Back-up the step size to half the old value, unless minimum reached
                    xyStep = xyStep_old/2.0
                    
                    #Floor this process at the minimum smoothing scale
                    if xyStep<=xyStepMin: xyStep=xyStepMin
 
                    #Lower detections flag (i.e. skip detection phase)
                    detections=False
                
                                                                         
            #If a median is not well defined but there are detections    
            else:
            
                #Increase n_under counter
                n_under += 1
                
                #Set fraction to null value
                f = '-1'
                
                #Back up old step and scale
                xyScale_old = xyScale
                xyStep_old = xyStep
                
                #If we are und
                if n_under>0: xyStep *= 1.5    #Increase step size if we are lagging behind
                
                #Update spatial scale and keep same step size               
                xyScale += xyStep


            ## DETECTION PHASE
            if detections:
            
                kXY[indices] = xyScale
                kW[indices] = wScale
                T[indices] = SNR[indices]
                                
                #Fill in detections
                D[indices] = I_xy[indices]
                
                #Mask newly detected pixels
                M[indices] = 1
                
                #Subtract detected values from original cube
                I[indices] -= I_xy[indices]
 

                
                #Update wavelength smoothed intensity and variance data
                I_w,kw = wavelengthSmooth(I,wScale)               

        ## Output some diagnostics
        perc = 100*(np.sum(M)-N0)/M.size
        if Npix>0:
            maxS,minS = max(SNRS),min(SNRS)
            if Npix>5: medS = np.median(SNRS)
            else: medS = np.mean(SNRS)
        else: maxS,minS,medS = 0,0,0
        
        output("%8i %8.3f %8.4f %8.4f %8.4f %8s\n" % (Npix,perc,minS,medS,maxS,str(f)))     

        sys.stdout.flush()

                 
    #Update wavelength scale
    wScale += wStep

t2 = time.time()

output("# Time elapsed: %5.2f\n" % (t2-t1))
      

hdu = fits.PrimaryHDU(D)
hdulist = fits.HDUList([hdu])
hdulist[0].header = fI[0].header
hdulist.writeto(Ifile.replace('.fits',args.ext),overwrite=True)
output("# Wrote %s\n" % Ifile.replace('.fits',args.ext))


if args.saveXYKer=="True":
    hdu = fits.PrimaryHDU(kXY)
    hdulist = fits.HDUList([hdu])
    hdulist[0].header = fI[0].header
    hdulist.writeto(Ifile.replace('.fits',args.extXYKer),overwrite=True)
    output("# Wrote %s\n" % Ifile.replace('.fits',args.extXYKer))

if args.saveWKer=="True":
    hdu = fits.PrimaryHDU(kW)
    hdulist = fits.HDUList([hdu])
    hdulist[0].header = fI[0].header
    hdulist.writeto(Ifile.replace('.fits',args.extWKer),overwrite=True)
    output("# Wrote %s\n" % Ifile.replace('.fits',args.extWKer))

if args.saveSNR=="True":
    hdu = fits.PrimaryHDU(T)
    hdulist = fits.HDUList([hdu])
    hdulist[0].header = fI[0].header
    hdulist.writeto(Ifile.replace('.fits',args.extSNR),overwrite=True)
    output("# Wrote %s\n" % Ifile.replace('.fits',args.extSNR))
