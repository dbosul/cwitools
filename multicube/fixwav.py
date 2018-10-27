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
cubeType = sys.argv[2]

#Figure out proper cube naming for variance data (limits this script to accept icube/ocube input)
if "icube" in cubeType: varType = cubeType.replace("icube","vcube")
elif "ocube" in cubeType: varType = cubeType.replace("ocube","vcube")
    
#Add file extension of omitted
if not ".fits" in cubeType: cubeType += ".fits"

#Load params from paramfile
params = libs.params.loadparams(parampath)

#Get regular and sky filenames   
ifiles = libs.io.findfiles(params,cubeType)

#Get sky files, if specified as separate to int files
snums  = [ s for s in set(params["SKY_ID"]) if s not in params["IMG_ID"] ]
sfiles = [ ifiles[0].replace(params["IMG_ID"][0],s) for s in snums ]


print "Corresponding sky files to be fixed:"
for s in sfiles: print s
print ""

#Add sky files to list of files to be corrected
for s in sfiles: ifiles.append(s)

#Create list of corresponding variance cubes
vfiles = [ f.replace(cubeType,varType) for f in ifiles ]

if "" in ifiles:
    print "Some files not found. Please correct paramfile (check data dir and image IDs) and try again.\n\n"
    sys.exit()

#Open fits files
ifits = [fitsIO.open(f) for f in ifiles] 

#Create fitter and line profile model
fitter = fitting.SimplexLSQFitter()
model = models.Gaussian1D()

#Run through input files + skies (whatever is to be corrected)
for i,f in enumerate(ifits):

    #Extract meta data
    h = f[0].header
    N = len(f[0].data)
    wg0,wg1 = h["WAVGOOD0"],h["WAVGOOD1"]
    w0,dw,w0px = h["CRVAL3"],h["CD3_3"],h["CRPIX3"]
    xc = int(h["CRPIX1"])
    yc = int(h["CRPIX2"])
    
    ## Figure out which sky to use for fitting this
    
    #If this is not N&S data - we can use the "icube" product of this exposure to fit sky lines
    if h["NASMASK"]==False: skyfits = fitsIO.open(ifiles[i].replace(cubeType,"icube.fits"))
    
    #If it IS N&S data, we can use the scube product from the same exposure
    elif h["NASMASK"]==True: skyfits = fitsIO.open(ifiles[i].replace(cubeType,"scube.fits"))
    
    else:
        print("Error reading NASMASK header for %i" % ifiles[i])
        sys.exit()
           
    #Make wavelength array
    W = np.array([w0 + dw*(j - w0px) for j in range(N)])
    
    #Crop to good wavelengths (if there is a useable line in that range)
    usewav = np.ones_like(W,dtype='bool')
    if any([ wg0<=sl<=wg1 for sl in sky_lines]):
        usewav[W<wg0] = 0
        usewav[W>wg1] = 0       
    W = W[usewav]

    #Get normalized sky spectrum
    sky_spectrum = np.sum(np.sum(skyfits[0].data[usewav],axis=1),axis=1)
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
            skyPlot.set_title("%s - %.2f - %.2f" % (ifiles[i].split('/')[-1],l,dp))
            skyPlot.plot(W,sky_spectrum,'kx-')
            
            linePlot = fig.add_subplot(gs[ :, 8: ])           
            linePlot.plot(linewav,linespec,'kx-')
            linePlot.plot(Wsmooth,fit,'r-')
            linePlot.plot(Wsmooth,modelproper(Wsmooth),'b-')
            linePlot.set_xlim([l-3*fit_window,l+3*fit_window])
            
            fig.show()
            plt.waitforbuttonpress()
            plt.close()
            
            lineFit=True
            
            #Shift data along wavelength axis
            f[0].data = shift(f[0].data,(-dp,0,0),order=0)
            
            #Save wavelength-corrected (wc) cube
            wcPath = ifiles[i].replace(".fits",".wc.fits")
            f.writeto(wcPath,overwrite=True)
            print("Wrote %s" % wcPath)
            
            #Apply the same shift to the corresponding variance cube
            #try:
            vfits = fitsIO.open(vfiles[i])
            vfits[0].data = shift(vfits[0].data,(-dp,0,0),order=0)
            vfits.writeto(vfiles[i].replace(".fits",".wc.fits"),overwrite=True)  
            print("Wrote %s" % vfiles[i].replace(".fits",".wc.fits"))              
            #except: print("Error wavelength-correcting variance cube for %s" % ifiles[i])
            
            #Output shift information
            print "%20s %6.2f %5.2fpx\n" % (ifiles[i].split('/')[-1],l,dp)
            break
        
        else: continue
        
    if not lineFit: print r"No good sky line for wavelength range %i-%i\AA"%(int(wg0),int(wg1))
        
