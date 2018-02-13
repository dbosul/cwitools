##################################################################################################                       
# GIVEN FITS FILE AND 2D LOCATION - SUBTRACT CONTINUUM EMISSION AROUND THAT POINT
#
# 1. Refine location
# 2. Model continuum emission of source
# 3. Subtract 
#

from astropy.modeling.fitting import SimplexLSQFitter
from astropy.modeling.models import custom_model,Moffat1D
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from scipy.ndimage.interpolation import shift
from scipy.ndimage.measurements import find_objects
from scipy.optimize import least_squares,curve_fit
from scipy.signal import correlate,deconvolve,convolve,gaussian
from scipy.stats import signaltonoise

import numpy as np
import params #custom
import sys

import matplotlib.pyplot as plt

lines = [1216]

def cSubtract(fits,pos,redshift=None,vwindow=500,radius=5,mode='scale2D',errLimit=3):
    global lines
    
    ##### DEFINE SUB-METHODS
    
    def moffat(r,I0,r0,a,b): return I0*(1 + ((r-r0)/a)**2)**(-b)
    def line(x,m,c): return m*x + c

    ##### EXTRACT DATA FROM FITS
    
    data = fits[0].data         #data cube
    head = fits[0].header       #header
    w,y,x = data.shape          #Cube dimensions
    X = np.arange(x)            #Create domains X,Y and W
    Y = np.arange(y)    
    W = np.array([ head["CRVAL3"] + head["CD3_3"]*(i - head["CRPIX3"]) for i in range(w)])
    
    ##### CREATE USEFUl VARIABLES & DATA STRUCTURES
    
    cmodel = np.zeros_like(data)            #Cube to store 3D continuum model
    
    usewav = np.ones_like(W,dtype=bool) #Boolean array for whether or not to use wavelengths in fitting
    
    Xs = np.linspace(X[0],X[-1],10*x) #Smooth X-Y domains for PSF modelling
    Ys = np.linspace(Y[0],Y[-1],10*y)
    
    ydist = 3600*np.sqrt( np.cos(head["CRVAL2"]*np.pi/180)*head["CD1_2"]**2 + head["CD2_2"]**2 ) #X & Y pixel sizes in arcseconds
    xdist = 3600*np.sqrt( np.cos(head["CRVAL2"]*np.pi/180)*head["CD1_1"]**2 + head["CD2_1"]**2 )
    
    ry = int(round(radius/ydist)) #X and Y 'radius' extent in pixels 
    rx = int(round(radius/xdist))

    
    ##### EXCLUDE BAD WAVELENGTHS AND EMISSION LINES

    usewav[ W < head["WAVGOOD0"] ] = 0
    usewav[ W > head["WAVGOOD1"] ] = 0
    
    if redshift!=None:
        for line in lines:    
            wc = (redshift+1)*line
            dw =(vwindow*1e5/3e10)*wc
            a,b = params.getband(wc-dw,wc+dw,head) 
            usewav[a:b] = 0


    ##### OPTIMIZE CENTROID

    xc,yc = pos #Take input position tuple 
    x0,x1 = max(0,xc-rx),min(x,xc+rx+1)
    y0,y1 = max(0,yc-ry),min(y,yc+ry+1)

    
    img = np.sum(data[usewav,y0:y1,x0:x1],axis=0)
    xdomain,xdata = range(x1-x0), np.sum(img,axis=0)
    ydomain,ydata = range(y1-y0), np.sum(img,axis=1)
    
    fit = SimplexLSQFitter() #Get astropy fitter class

    moffat_bounds = {'amplitude':(0,float("inf")) }
    xMoffInit = Moffat1D(max(xdata),x_0=xc-x0,bounds=moffat_bounds) #Initial guesses
    yMoffInit = Moffat1D(max(ydata),x_0=yc-y0,bounds=moffat_bounds)
    
    xMoffFit = fit(xMoffInit,xdomain,xdata) #Fit Moffat1Ds to each axis
    yMoffFit = fit(yMoffInit,ydomain,ydata)
    
    xc_new = xMoffFit.x_0.value + x0
    yc_new = yMoffFit.x_0.value + y0
    
    #If the new centroid is beyond our anticipated error range away... just use scale method
    if abs(xc-xc_new)*xdist>errLimit or abs(yc-yc_new)*ydist>errLimit: mode='scale2D'
    
    #Otherwise, update the box to center better on our continuum source
    else:
        xc, yc = int(round(xc_new)),int(round(yc_new))
        x0,x1 = max(0,xc-rx),min(x,xc+rx+1)
        y0,y1 = max(0,yc-ry),min(y,yc+ry+1)        
                
    #This method creates a 2D continuum image and scales it at each wavelength.
    if mode=='scale2D':
    
        ##### CREATE CROPPED CUBE 
        
        cube = data[:,y0:y1,x0:x1].copy()              #Create smaller working cube to isolate continuum source

        cube-=np.median(cube)                          #Median subtract the cube
        
        flat = np.ndarray.flatten(cube) 
        flatAbs = np.abs(flat)
        
        minval = np.min( flatAbs[flatAbs!=0] )
        if minval<1: cube/=minval
        
             
        ##### CREATE 2D CONTINUUM IMAGE           
        cont2d = np.mean(cube[usewav],axis=0) #Create 2D continuum image
        
             
        ##### BUILD 3D CONTINUUM MODEL
        
        for i in range(cube.shape[0]):
           
            chisq = lambda A: np.sum(cube[i] - A*cont2d) #Optimization function
            
            A0 = np.sum(cube[i])/np.sum(cont2d) #Initial guess

            fit = least_squares(chisq,A0) #Run the fit    

            Afit = fit.x #Get optimzed scaling factor
            
            model = Afit*cont2d*minval #Add this wavelength layer to the model
            
            data[i,y0:y1,x0:x1] -= model #Subtract from data cube
        
            cmodel[i,y0:y1,x0:x1] += model #Add to larger model cube
            
        return data,cmodel

    #This method just fits a simple line to the spectrum each spaxel; for flat continuum sources.
    elif mode=='lineFit':
        
        #Define custom astropy model class (just a line)
        @custom_model
        def line(xx,m=0,c=0): return m*xx + c
           
        #Run through pixels in 2D region
        for yi in range(y0,y1):
            for xi in range(x0,x1):
                              
                m_init = line() #Create initial guess model
                m = fit(m_init, W[usewav], data[usewav,yi,xi]) #Optimize model
                
                model = m(W)
                
                cmodel[:,yi,xi] += model
                data[:,yi,xi] -= model
        
        return data,cmodel
        
    #This method extracts a central spectrum and fits it to each spaxel
    elif mode=='specFit':
    
        #Define custom astropy model class (just a line)
        @custom_model
        def line(xx,m=0,c=0): return m*xx + c


        ##### GET QSO SPECTRUM
        q_spec = data[:,yc,xc].copy()
        q_spec_fit = q_spec[usewav==1]

        #Run through slices
        for yi in range(y0,y1):
        
            print yi,
            sys.stdout.flush()
            
            #If this not the main QSO slice
            if yi!=yc:
                            
                #Extract QSO spectrum for this slice
                s_spec = data[:,yi,xc].copy() 
                s_spec_fit = s_spec[usewav==1]

                #Estimate wavelength shift needed
                corr = correlate(s_spec,q_spec)
                corrs = gaussian_filter1d(corr,5.0)
                w_offset = (np.nanargmax(corrs)-len(corrs)/2)/2.0

                #Find wavelength offset (px) for this slice
                chisq = lambda x: s_spec_fit[10:-10] - x[0]*shift(q_spec_fit,x[1],order=4,mode='reflect')[10:-10]

                p0 = [np.max(s_spec)/np.max(q_spec),w_offset]
                
                lbound = [0.0,-5]
                ubound = [5.1, 5]        
                for j in range(len(p0)):
                    if p0[j]<lbound[j]: p0[j]=lbound[j]
                    elif p0[j]>ubound[j]: p0[j]=ubound[j]
                
                p_fit = least_squares(chisq,p0,bounds=(lbound,ubound),jac='3-point')                

                A0,dw0 =p_fit.x

                q_spec_shifted = shift(q_spec_fit,dw0,order=3,mode='reflect')
     
            else:
                q_spec_shifted = q_spec_fit
                A0 = 0.5
                dw0=0
                
            lbound = [0.0,-5]
            ubound = [20.0,5]
                                  
            for xi in range(x0,x1):

                spec = data[:,yi,xi]
                spec_fit = spec[usewav==1]
                             
                #First fit to find wav offset for this slice
                chisq = lambda x: spec_fit - x[0]*shift(q_spec_fit,x[1],order=3,mode='reflect')

                
                
                p0 = [A0,dw0]
                for j in range(len(p0)):
                    if p0[j]<lbound[j]: p0[j]=lbound[j]
                    elif p0[j]>ubound[j]: p0[j]=ubound[j]
                    #elif abs(p0[j]<1e-6): p0[j]=0

                sys.stdout.flush()
                p_fit = least_squares(chisq,p0,bounds=(lbound,ubound),jac='3-point')

                A,dw = p_fit.x
                
                m_spec = A*shift(q_spec,dw,order=4,mode='reflect')
                
                #Do a linear fit to residual and correct linear errors
                residual = data[:,yi,xi]-m_spec
                
                ydata = residual[usewav==1]
                xdata = W[usewav==1]

                m_init = line() #Create initial guess model
                m = fit(m_init, xdata, ydata) #Optimize model
                
                linefit = m(W)
       
                model = linefit + m_spec
                residual = data[:,yi,xi] - model

                cmodel[:,yi,xi] += model
                data[:,yi,xi] -= model
        
        return data,cmodel  
        
        
def regSubtract(fits,mask,redshift=None):
    
    global lines
    
    ##### DEFINE SUB-METHODS
    
    def moffat(r,I0,r0,a,b): return I0*(1 + ((r-r0)/a)**2)**(-b)
    def line(x,m,c): return m*x + c

    ##### EXTRACT DATA FROM FITS
    
    data = fits[0].data         #data cube
    head = fits[0].header       #header
    w,y,x = data.shape          #Cube dimensions
    X = np.arange(x)            #Create domains X,Y and W
    Y = np.arange(y)    
    W = np.array([ head["CRVAL3"] + head["CD3_3"]*(i - head["CRPIX3"]) for i in range(w)])
    
    
    ##### EXTRACT DATA FROM MASK (make 3D mask first)
    mask3d = np.array( [mask for wi in range(w)])
    maskNums = np.unique(mask3d[mask3d>0])
    objects = [find_objects(mask3d==m)[0] for m in maskNums]

    ##### CREATE USEFUl VARIABLES & DATA STRUCTURES
    
    cmodel = np.zeros_like(data)            #Cube to store 3D continuum model
    
    usewav = np.ones_like(W,dtype=bool) #Boolean array for whether or not to use wavelengths in fitting
    
    Xs = np.linspace(X[0],X[-1],10*x) #Smooth X-Y domains for PSF modelling
    Ys = np.linspace(Y[0],Y[-1],10*y)

    ##### EXCLUDE BAD WAVELENGTHS AND EMISSION LINES

    usewav[ W < head["WAVGOOD0"] ] = 0
    usewav[ W > head["WAVGOOD1"] ] = 0
    
    if redshift!=None:
        for line in lines:    
            wc = (redshift+1)*line
            dw =(vwindow*1e5/3e10)*wc
            a,b = params.getband(wc-dw,wc+dw,head) 
            usewav[a:b] = 0

    for obj in objects:


        subcube = data[obj].copy()
        image2d = np.sum(subcube[usewav])
        print np.mean(subcube)/np.std(subcube)
        
        plt.figure()
        plt.subplot(211)
        plt.pcolor(np.sum(subcube[usewav==1],axis=0))
        plt.colorbar()
        plt.subplot(212)
        plt.pcolor(mask)
        plt.colorbar()
        plt.show()
