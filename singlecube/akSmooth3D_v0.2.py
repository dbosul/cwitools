from astropy.io import fits
from astropy.convolution import Gaussian2DKernel,convolve_fft
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import gaussian_filter1d,uniform_filter1d
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys

import time

plt.style.use('ggplot')

#Settings for program
settings = {'wavmode':'box','xymode':'gaussian','snr':3}

#Function to smooth along wavelength axis
def wavelengthSmooth(a,scale):
    global settings
    mode = settings['wavmode']   
    if mode=='box': return uniform_filter1d(a,scale,axis=0,mode='constant')
    elif mode=='gaussian': return gaussian_filter1d(a,scale/2.355,axis=0,mode='constant')
    else: print "Mode not found";sys.exit()

#Function to smooth along two spatial axes
def spatialSmooth(a,scale):
    global settings
    mode = settings['xymode'] 
    if mode=='box': return uniform_filter(a,(1,scale,scale),mode='constant')
    elif mode=='gaussian': return gaussian_filter1d(gaussian_filter1d(a,scale/2.355,axis=1,mode='constant'),scale/2.355,axis=2,mode='constant')

    else: print "Mode not found";sys.exit()
     
### INPUT
Ifile = sys.argv[1]
Vfile = sys.argv[2]
if len(sys.argv)>3:
    for x in sys.argv[3:]:
        try:
            key,val = x.split('=')
            settings[key]=val
        except:
            print "Input not understood. Check arguments and try again."
            sys.exit()

print "Input intensity data: %s" % Ifile
print "Input variance data: %s" % Vfile
print "XY Smoothing mode: %s" % settings['xymode']
print "Wav Smoothing mode: %s" % settings['wavmode']

#Open input intensity cube
try: fI = fits.open(Ifile)
except: print "Error opening file %s. Please check and try again."%sys.argv[1];sys.exit()

#Open input variance cube
try: fV = fits.open(Vfile)
except: print "Error opening file %s. Please check and try again."%sys.argv[1];sys.exit()


## VARIABLES & DATA STRUCTURES

I = fI[0].data.copy()           #Original intensity cube

V = fV[0].data.copy()           #Original variance cube

D = np.zeros_like(I).flatten()  #Detection cube

M = np.zeros_like(I).flatten()  #For masking pixels after detection

shape = I.shape                 #Data shape

xyScale0 = 1.                   #Establish minimum smoothing scales
wScale0 = 2

xyStep0 = 1.                    #Establish default step sizes
wStep0 = 1

xyScale1 = 30.                  #Establish maximum smoothing scales
wScale1 = 12

xyStepMin = 0.1                #Establish minimum step sizes
wStepMin = 1

tau_min = float(settings["snr"])  #Minimum signal-to-noise threshold
tau_max = tau_min*1.1
tau_mid = (tau_min+tau_max)/2.0

## PRE-PROCESSING FOR MAIN LOOP

M[I.flatten()==0] = 1   #Mask Zero values in cube

N0 = np.sum(M) #Get Npix that are masked by default (to distinguish from detections)

#Initialize w loop variables
wScale = wScale0  
wStep = wStep0  

#Keeping track of how many steps you have had no detections
n_under = 0

t1 = time.time()

## MAIN LOOP
print "%8s %8s %8s %8s %8s %8s %8s %8s %8s %8s" % ('wScale','wStep','xyScale','xyStep','Npix','% Done','minSNR','medSNR','maxSNR','mid/med')     
while wScale <= wScale1: #Run through wavelength bins

    #Initialize xy loop variables
    xyScale = xyScale0
    xyStep = xyStep0
    
    #Back up old scale and step size
    xyScale_old = xyScale 
    xyStep_old = xyStep 
    
    #Back-up old scale and stepsize
    wScale_old = wScale 
    wStep_old = wStep
    
    #Wavelength smooth intensity and variance data
    I_w = wavelengthSmooth(I,wScale)
    V_w = wavelengthSmooth(V,wScale)

    while xyScale <= xyScale1:
        
        #Output first half of diagnostic info
        print "%8.2f %8.3f %8.2f %8.3f" % (wScale,wStep,xyScale,xyStep),  
           
        #Start off boolean to flag detections
        detections=False
        
        #Spatially smooth intensity and variance data
        I_xy = spatialSmooth(I_w,xyScale)
        V_xy = spatialSmooth(V_w,xyScale)

        #Flatten arrays
        I_xy_flat = I_xy.flatten()
        V_xy_flat = V_xy.flatten()
 
        #Prevent divison by zero and zero out any undefined SNRs
        V_xy_flat[V_xy_flat<=0] = np.inf
        
        #Get SNR array
        SNR = I_xy_flat/np.sqrt(V_xy_flat)
        
        # Since the smoothing kernels are normalized, we need to
        # artificially increase the SNR to reflect a hypothetical value
        # e.g. 'if we summed the data over this kernel, the SNR would be...'
        SNR *= (xyScale*np.sqrt(wScale)) 
        
        #Get indices of detections
        indices = SNR >= tau_min
        indices[M==1] = 0

        #Get SNR values and total # of new detections
        SNRS = SNR[indices]
        Npix = len(SNRS)
    
        #If there are no detections
        if Npix==0:
        
            n_under+=1
            
            f = '-'             #Set fraction to null
            if n_under>0: xyStep *= 1.1   #Increase step size if we are lagging behind
            xyScale += xyStep   #Increase scale            
                                 
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

                #If there is no oversmoothing or we are at the smallest scale already
                if f>=0 or xyScale<=xyScale0:

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

                #If oversmoothing is evident (i.e if f<0 and we are above minimum scale)
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
                f = '-'
                
                #Back up old step and scale
                xyScale_old = xyScale
                xyStep_old = xyStep
                
                #If we are und
                if n_under>0: xyStep *= 1.5    #Increase step size if we are lagging behind
                
                #Update spatial scale and keep same step size               
                xyScale += xyStep


            ## DETECTION PHASE
            if detections:
            
                #Fill in detections
                D[indices] = I_xy_flat[indices]
                
                #Mask newly detected pixels
                M[indices] = 1
                
                #Subtract detected values from original cube
                I_flat = I.flatten()
                I_flat[indices] -= I_xy_flat[indices]
                I = I_flat.reshape(shape)
                
                #Update wavelength smoothed intensity and variance data
                I_w = wavelengthSmooth(I,wScale)               

        ## Output some diagnostics
        perc = 100*(np.sum(M)-N0)/M.size
        if Npix>5: medS,maxS,minS = np.median(SNRS),max(SNRS),min(SNRS)
        else: medS,maxS,minS = 0,0,0
        
        print "%8i %8.3f %8.2f %8.2f %8.2f %8s" % (Npix,perc,minS,medS,maxS,str(f))     

        sys.stdout.flush()
                  
    #Update wavelength scale
    wScale += wStep

t2 = time.time()

print "Time elapsed: %5.2f" % (t2-t1)     
plt.show()        
D = D.reshape(shape)
hdu = fits.PrimaryHDU(D)
hdulist = fits.HDUList([hdu])
hdulist[0].header = fI[0].header
hdulist.writeto(sys.argv[1].replace('.fits','.AKS.fits'),overwrite=True)
print "Wrote %s" % sys.argv[1].replace('.fits','.AKS.fits')
