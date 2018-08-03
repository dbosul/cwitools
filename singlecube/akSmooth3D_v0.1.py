from astropy.convolution import Box1DKernel,Box2DKernel
from astropy.convolution import convolve
from astropy.io import fits


from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter

import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt

settings = {'wavmode':'box','xymode':'gaussian'}

plt.style.use('ggplot')

def wavelengthSmooth(a,scale):
    global settings
    mode = settings['wavmode']   
    if mode=='box': return uniform_filter(a,(scale,1,1),mode='constant')
    elif mode=='gaussian': return gaussian_filter(a,(scale/2.355,1,1),mode='constant')
    else: print "Mode not found";sys.exit()

def spatialSmooth(a,scale):
    global settings
    mode = settings['xymode'] 
    if mode=='box': return uniform_filter(a,(1,scale,scale),mode='constant')
    elif mode=='gaussian': return gaussian_filter(a,(1,scale/2.355,scale/2.355),mode='constant')
    else: print "Mode not found";sys.exit()
     
#Parse user input
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

#Establish bins for smoothing
xy_bins = np.linspace(1,10,13)
print xy_bins
w_bins = [2,4]
T = 5

#Extract data and create data structures
I = fI[0].data.copy()   #Original intensity cube
V = fV[0].data.copy()
D = np.zeros_like(I).flatten()    #Detection cube
M = np.zeros_like(I).flatten() #For masking pixels after detection
shape = I.shape
M[I.flatten()==0] = 1 #Mask Zero values in cube

#Smoothing process
plt.figure(figsize=(18,9))

iMax = len(xy_bins)*len(w_bins)
percs = []
medSNRs = []
maxSNRs = []
minSNRs = []

#Run through wavelength bins
for wb in w_bins:  

    sys.stdout.flush()
    
    #Wavelength smooth intensity and variance data
    I_wb = wavelengthSmooth(I,wb)
    V_wb = wavelengthSmooth(V,wb)
    
    for xyb in xy_bins:

        #Spatially smooth intensity and variance data
        I_xyb = spatialSmooth(I_wb,xyb)
        V_xyb = spatialSmooth(V_wb,xyb)

        #Flatten arrays
        I_xyb_flat = I_xyb.flatten()
        V_xyb_flat = V_xyb.flatten()
 
        #Get SNR array
        SNR = I_xyb_flat/np.sqrt(V_xyb_flat)
        SNR *= (xyb*np.sqrt(wb)) #Adjust SNR values to be 'as if' data was summed rather than averaged
        
        #Get detections
        indices = SNR >= T
        indices[M==1] = 0

        #Fill in detections
        D[indices] = I_xyb_flat[indices]
        
        #Mask newly detected pixels
        M[indices] = 1
        
        #Subtract detected values from working cube
        I_flat = I.flatten()
        I_flat[indices] -= I_xyb_flat[indices]
        I = I_flat.reshape(shape)
        
        snrs = SNR[indices]
        perc = 100*np.sum(M)/M.size
        if len(snrs)>0: medS,maxS,minS = np.median(snrs),max(snrs),min(snrs)
        else: medS,maxS,minS = 0,0,0
        npix = np.sum(indices)
        print "%4.2f %4.2f %6i %5.2f %5.2f %5.2f %5.2f" % (wb,xyb,npix,medS,maxS,minS,perc)
        
        sys.stdout.flush()

        percs.append(perc)
        medSNRs.append(medS)
        maxSNRs.append(maxS)
        minSNRs.append(minS)
        
        plt.subplot(311)
        plt.plot(medSNRs,'k-')
        plt.plot(maxSNRs,'b-',alpha=0.7)
        plt.plot(minSNRs,'b-',alpha=0.7)
        plt.xlim([0,iMax])
        plt.ylim([T,1.1*T])
        plt.subplot(312)
        plt.plot(percs,'k-')
        plt.xlim([min(percs)-1,max(percs)+1])
        plt.ylim([15,25])
        plt.pause(0.05)
        plt.subplot(313)
        
        
plt.show()        
D = D.reshape(shape)
hdu = fits.PrimaryHDU(D)
hdulist = fits.HDUList([hdu])
hdulist[0].header = fI[0].header
hdulist.writeto("aks-z.fits",overwrite=True)
        
