#!/usr/bin/env python

from astropy.io import fits as fitsIO
from astropy.modeling import models,fitting
from astropy.modeling.models import custom_model
from scipy import interpolate
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter1d

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import libs


script_path = os.path.abspath(sys.argv[0])

#TO-DO: Replace hard-coded sky lines with file/table input
sky_lines = [4358.328,4046.563,4839.794,4670.406,4916.068]
fit_window = 3

#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]
skycubetype = "scuber.fits"#cubetype.replace('icube','scube')

#Add file extension of omitted
if not ".fits" in cubetype: cubetype += ".fits"


#Load params from paramfile
params = libs.params.loadparams(parampath)

#Get regular and sky filenames   
files = libs.io.findfiles(params,cubetype)
skyfiles = libs.io.findfiles(params,skycubetype)

if "" in files or files==[]:
    print "Some files not found. Please correct paramfile (check data dir and image IDs) and try again.\n\n"
    sys.exit()
    
#Open custom FITS-3D objects
fits = [libs.fits3D.open(f) for f in files] 
skyfits = [libs.fits3D.open(f) for f in skyfiles] 


fitter = fitting.SimplexLSQFitter()
model = models.Gaussian1D()

for i,f in enumerate(fits):

    #Extract meta data
    h = f[0].header
    N = len(f[0].data)
    wg0,wg1 = h["WAVGOOD0"],h["WAVGOOD1"]
    w0,dw,w0px = h["CRVAL3"],h["CD3_3"],h["CRPIX3"]
    xc = int(h["CRPIX1"])
    yc = int(h["CRPIX2"])
    
    #Make wavelength array
    W = np.array([w0 + dw*(j - w0px) for j in range(N)])
    usewav = np.ones_like(W,dtype='bool')
    
    #Crop to good wavelengths (if there is a useable line in that range)
    if any([ wg0<=sl<=wg1 for sl in sky_lines]):
        usewav[W<wg0] = 0
        usewav[W>wg1] = 0
        
    W = W[usewav]

    #Get normalized sky spectrum
    sky_spectrum = np.sum(np.sum(skyfits[i][0].data[usewav],axis=1),axis=1)
    sky_spectrum/=np.max(sky_spectrum)

    #Get smooth wavelength array
    Wsmooth = np.linspace(W[0],W[-1],10*len(W))
    
    #Flag to see if line has been fit
    lineFit=False
    
    
    #Run through sky lines until one is useable
    for l in sky_lines:
                
        if W[0]<=l<=W[-1]:

            #Identify good fitting wavelengths 
            fitwav = np.ones_like(W,dtype='bool')
            fitwav[ W < l-fit_window ] = 0
            fitwav[ W > l+fit_window+1 ] = 0                    

            #Extract line from spectrum (1st try)
            linespec = sky_spectrum[fitwav] - np.median(sky_spectrum)
            linespec = gaussian_filter1d(linespec,1.0)
            linewav = W[fitwav]
            
            #Fit Gaussian to line (1st try)
            A0 = np.max(linespec)
            l0 = linewav[np.nanargmax(linespec)]
            
            modelguess = models.Gaussian1D(amplitude=A0,mean=l0,stddev=1.0)
            modelguess.mean.min = modelguess.mean.value-fit_window
            modelguess.mean.max = modelguess.mean.value+fit_window 
            modelguess.stddev.min = 0.5
            modelguess.stddev.max = 2.0                                           
            modelfit = fitter(modelguess,linewav,linespec)
            fit = modelfit(Wsmooth)


            #Identify good fitting wavelengths 
            l1 = modelfit.mean.value
            fitwav = np.ones_like(W,dtype='bool')
            fitwav[ W < l1-fit_window ] = 0
            fitwav[ W > l1+fit_window+1 ] = 0                    
                 

            #Extract line from spectrum (1st try)
            linespec = sky_spectrum[fitwav] - np.min(sky_spectrum[fitwav])
            linespec = gaussian_filter1d(linespec,1.0)
            linewav = W[fitwav]
            
            #Fit Gaussian to line (1st try)
            A1 = np.max(linespec)
            modelguess = models.Gaussian1D(amplitude=A1,mean=l1,stddev=1.0)
            modelguess.mean.min = modelguess.mean.value-fit_window
            modelguess.mean.max = modelguess.mean.value+fit_window 
            modelguess.stddev.min = 0.5
            modelguess.stddev.max = 2.0      
            modelfit = fitter(modelguess,linewav,linespec)
            fit = modelfit(Wsmooth)

            modelproper = models.Gaussian1D(amplitude=modelfit.amplitude.value,mean=l,stddev=modelfit.stddev.value)
            
            dp=(modelfit.mean.value-l)/h["CD3_3"]

            grid_width = 10
            grid_height = 1
            gs = gridspec.GridSpec(grid_height,grid_width)   
            fig = plt.figure(figsize=(16,4))
            
            skyPlot = fig.add_subplot(gs[ :, 0:8 ])
            skyPlot.set_title("%s - %.2f - %.2f" % (params["IMG_ID"][i],l,dp))
            skyPlot.plot(W,sky_spectrum,'kx-')
            
            linePlot = fig.add_subplot(gs[ :, 8: ])           
            linePlot.plot(linewav,linespec,'kx-')
            linePlot.plot(Wsmooth,fit,'r-')
            linePlot.plot(Wsmooth,modelproper(Wsmooth),'b-')
            linePlot.set_xlim([l-3*fit_window,l+3*fit_window])
            
            fig.show()
            plt.waitforbuttonpress()
            plt.savefig("%s_%s_skyfix.png" % (params["DATA_DIR"],params["IMG_ID"][i]))
            plt.close()
            
            lineFit=True
            
            #Shift data along wavelength axis
            f[0].data = shift(f[0].data,(-dp,0,0),order=0)
            
            #Save wavelength-corrected (wc) cube
            wcPath = files[i].replace(".fits",".wc.fits")
            f.save(wcPath)
            
            print "%20s %6.2f %5.2f" % (params["IMG_ID"][i],l,dp)
            break
        
        else: continue
        
    if not lineFit: print r"No good sky line for wavelength range %i-%i\AA"%(int(wg0),int(wg1))
        
