from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys


plt.style.use('ggplot')

#Settings for program
settings = {'wavmode':'box','xymode':'gaussian'}

#Function to smooth along wavelength axis
def wavelengthSmooth(a,scale):
    global settings
    mode = settings['wavmode']   
    if mode=='box': return uniform_filter(a,(scale,1,1),mode='constant')
    elif mode=='gaussian': return gaussian_filter(a,(scale/2.355,1,1),mode='constant')
    else: print "Mode not found";sys.exit()

#Function to smooth along two spatial axes
def spatialSmooth(a,scale):
    global settings
    mode = settings['xymode'] 
    if mode=='box': return uniform_filter(a,(1,scale,scale),mode='constant')
    elif mode=='gaussian': return gaussian_filter(a,(1,scale/2.355,scale/2.355),mode='constant')
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

xyScale0 = 1                    #Establish minimum smoothing scales
wScale0 = 1

xyStep0 = 1                   #Establish default step sizes
wStep0 = 1

xyScale1 = 6                    #Establish maximum smoothing scales
wScale1 = 4

T = 5                           #Minimum signal-to-noise threshold

## PRE-PROCESSING FOR MAIN LOOP

M[I.flatten()==0] = 1   #Mask Zero values in cube

wScale = wScale0        #Loop variables - smoothing scales
xyScale = xyScale0

wStep = wStep0          #Loop variables - smoothing scale step sizes
xyStep = xyStep0


## MAIN LOOP

while wScale <= wScale1: #Run through wavelength bins
   
    #Wavelength smooth intensity and variance data
    I_w = wavelengthSmooth(I,wScale)
    V_w = wavelengthSmooth(V,wScale)
    
    while xyScale <= xyScale1:
        
        #Spatially smooth intensity and variance data
        I_xy = spatialSmooth(I_w,xyScale)
        V_xy = spatialSmooth(V_w,xyScale)

        #Flatten arrays
        I_xy_flat = I_xy.flatten()
        V_xy_flat = V_xy.flatten()
 
        #Get SNR array
        SNR = I_xy_flat/np.sqrt(V_xy_flat)
        
        # Since the smoothing kernels are normalized, we need to
        # artificially increase the SNR to reflect a hypothetical value
        # e.g. 'if we summed the data over this kernel, the SNR would be...'
        SNR *= (xyScale*np.sqrt(wScale)) 
        
        #Get indices of detections
        indices = SNR >= T
        indices[M==1] = 0

        #Fill in detections
        D[indices] = I_xy_flat[indices]
        
        #Mask newly detected pixels
        M[indices] = 1
        
        #Subtract detected values from working cube
        I_flat = I.flatten()
        I_flat[indices] -= I_xy_flat[indices]
        I = I_flat.reshape(shape)

        #Update spatial scale
        xyScale += xyStep
                
        #Output some diagnostics
        snrs = SNR[indices]
        perc = 100*np.sum(M)/M.size
        if len(snrs)>5: medS,maxS,minS = np.median(snrs),max(snrs),min(snrs)
        else: medS,maxS,minS = 0,0,0
        if len(snrs)>5: frac = ((T+1.1*T)/2)/medS - 1
        else: frac = 0
        npix = np.sum(indices)
        print "%4.2f %4.2f %6i %5.2f %5.2f %5.2f %5.2f %6.2f" % (wScale,xyScale,npix,medS,maxS,minS,perc,frac)     
        sys.stdout.flush()
        
        
    #Update wavelength scale
    wScale += wStep
        
plt.show()        
D = D.reshape(shape)
hdu = fits.PrimaryHDU(D)
hdulist = fits.HDUList([hdu])
hdulist[0].header = fI[0].header
hdulist.writeto("aks-z.fits",overwrite=True)
        
