##################################################################################################                       
# GIVEN FITS FILE AND 2D LOCATION - SUBTRACT CONTINUUM EMISSION AROUND THAT POINT
#
# 1. Refine location
# 2. Model continuum emission of source
# 3. Subtract 
#

from astropy.modeling import fitting,models
from astropy.modeling.models import custom_model

import numpy as np
import params #custom
import scipy
import sys
import warnings

import matplotlib.pyplot as plt

lines = [1216]
skylines = [4360]

def cSubtract(fits,pos,redshift=None,vwindow=1000,radius=5,mode='scale2D',errLimit=3):
    global lines,skylines
    
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
    for wc in skylines:
        dw = 5
        a,b = params.getband(wc-dw,wc+dw,head)
        data[a:b] = 0
        usewav[a:b] = 0

    ##### OPTIMIZE CENTROID

    xc,yc = pos #Take input position tuple 
    x0,x1 = max(0,xc-rx),min(x,xc+rx+1)
    y0,y1 = max(0,yc-ry),min(y,yc+ry+1)

    
    img = np.sum(data[usewav,y0:y1,x0:x1],axis=0)
    xdomain,xdata = range(x1-x0), np.sum(img,axis=0)
    ydomain,ydata = range(y1-y0), np.sum(img,axis=1)
    
    fit = fitting.SimplexLSQFitter() #Get astropy fitter class

    moffat_bounds = {'amplitude':(0,float("inf")) }
    xMoffInit = models.Moffat1D(max(xdata),x_0=xc-x0,bounds=moffat_bounds) #Initial guesses
    yMoffInit = models.Moffat1D(max(ydata),x_0=yc-y0,bounds=moffat_bounds)
    
    xMoffFit = fit(xMoffInit,xdomain,xdata) #Fit Moffat1Ds to each axis
    yMoffFit = fit(yMoffInit,ydomain,ydata)
    
    xc_new = xMoffFit.x_0.value + x0
    yc_new = yMoffFit.x_0.value + y0
    
    #If the new centroid is beyond our anticipated error range away... just use scale method
    if abs(xc-xc_new)*xdist>errLimit or abs(yc-yc_new)*ydist>errLimit: mode='scale2D'
    
    #Otherwise, update the box to center better on our continuum source
    else:
        xc, yc = int(round(xc_new)),int(round(yc_new)) #Round to nearest integer
        x0,x1 = max(0,xc-rx),min(x,xc+rx+1) #Get new ranges
        y0,y1 = max(0,yc-ry),min(y,yc+ry+1)        
        xc = max(0,min(x-1,xc)) #Bound new variables to within image
        yc = max(0,min(y-1,yc)) 
    

    #This method creates a 2D continuum image and scales it at each wavelength.
    if mode=='scale2D':
    
        print 'scale2D',
        ##### CREATE CROPPED CUBE 
        
        cube = data[:,y0:y1,x0:x1].copy()              #Create smaller working cube to isolate continuum source

        cube-=np.median(cube)                          #Median subtract the cube

        ##### CREATE 2D CONTINUUM IMAGE           
        cont2d = np.mean(cube[usewav],axis=0) #Create 2D continuum image
        
             
        ##### BUILD 3D CONTINUUM MODEL
        
        for i in range(cube.shape[0]):
           
            chisq = lambda A: np.sum(cube[i] - A*cont2d) #Optimization function
            
            A0 = max(0,np.sum(cube[i])/np.sum(cont2d)) #Initial guess
              
            fit = scipy.optimize.least_squares(chisq,A0) #Run the fit    

            Afit = fit.x #Get optimzed scaling factor
            
            model = Afit*cont2d #Add this wavelength layer to the model
            
            data[i,y0:y1,x0:x1] -= model #Subtract from data cube
        
            cmodel[i,y0:y1,x0:x1] += model #Add to larger model cube
        
        return data,cmodel

    #This method just fits a simple line to the spectrum each spaxel; for flat continuum sources.
    elif mode=='lineFit':
        
        print 'lineFit',
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
    
        print 'specFit',
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
                corr = scipy.signal.correlate(s_spec,q_spec)
                corrs = scipy.ndimage.filters.gaussian_filter1d(corr,5.0)
                w_offset = (np.nanargmax(corrs)-len(corrs)/2)/2.0

                #Find wavelength offset (px) for this slice
                chisq = lambda x: s_spec_fit[10:-10] - x[0]*scipy.ndimage.interpolation.shift(q_spec_fit,x[1],order=4,mode='reflect')[10:-10]

                p0 = [np.max(s_spec)/np.max(q_spec),w_offset]
                
                lbound = [0.0,-5]
                ubound = [5.1, 5]        
                for j in range(len(p0)):
                    if p0[j]<lbound[j]: p0[j]=lbound[j]
                    elif p0[j]>ubound[j]: p0[j]=ubound[j]
                
                p_fit = scipy.optimize.least_squares(chisq,p0,bounds=(lbound,ubound),jac='3-point')                

                A0,dw0 =p_fit.x

                q_spec_shifted = scipy.ndimage.interpolation.shift(q_spec_fit,dw0,order=3,mode='reflect')
     
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
                chisq = lambda x: spec_fit - x[0]*scipy.ndimage.interpolation.shift(q_spec_fit,x[1],order=3,mode='reflect') 

                p0 = [A0,dw0]
                for j in range(len(p0)):
                    if p0[j]<lbound[j]: p0[j]=lbound[j]
                    elif p0[j]>ubound[j]: p0[j]=ubound[j]
                    #elif abs(p0[j]<1e-6): p0[j]=0

                sys.stdout.flush()
                p_fit = scipy.optimize.least_squares(chisq,p0,bounds=(lbound,ubound),jac='3-point')

                A,dw = p_fit.x
                
                m_spec = A*scipy.ndimage.interpolation.shift(q_spec,dw,order=4,mode='reflect') 
                
                #Do a linear fit to residual and correct linear errors
                residual = data[:,yi,xi]-m_spec
                
                ydata = residual[usewav==1]
                xdata = W[usewav==1]

                #m_init = line() #Create initial guess model
                #m = fit(m_init, xdata, ydata) #Optimize model
                
                #linefit = m(W)
       
                model =  m_spec
                #residual = data[:,yi,xi] - model

                if 0 and abs(yi-yc)<1 and abs(xi-xc)<1:

                    plt.figure(figsize=(16,8))
                    
                    plt.subplot(311)
                    plt.title(r"$A=%.4f,d\lambda=%.3fpx$" % (A,dw))
                    plt.plot(W,spec,'bx',alpha=0.5)
                    plt.plot(W[usewav==1],spec[usewav==1],'kx')
                    plt.plot(W,A*q_spec,'g-',alpha=0.8)
                    plt.plot(W,model,'r-')
                    plt.xlim([W[0],W[-1]])
                    plt.ylim([1.5*min(spec),max(spec)*1.5])
                    plt.subplot(312)
                    plt.xlim([W[0],W[-1]])

                    plt.plot(W,residual,'gx')
                    plt.ylim([1.5*min(residual),max(spec)*1.5])  
                                                      
                    plt.subplot(313)
                    plt.hist(residual)

                    plt.tight_layout()           
                    plt.show()
                            
                            
                cmodel[:,yi,xi] += model
                data[:,yi,xi] -= model

        return data,cmodel  
        
        
def bgSubtract(fits,mask,redshift=None,vwindow = 500,):

    global lines,skyline
    
    print "Subtracting background"
    
    ##### EXTRACT DATA FROM FITS    
    data = fits[0].data         #data cube
    head = fits[0].header       #header
    w,y,x = data.shape          #Cube dimensions
    X = np.arange(x)            #Create domains X,Y and W
    Y = np.arange(y)    
    W = np.array([ head["CRVAL3"] + head["CD3_3"]*(i - head["CRPIX3"]) for i in range(w)])
    
    ##### CREATE USEFUl VARIABLES & DATA STRUCTURES  
    bmodel = np.zeros_like(data)            #Cube to store 3D continuum model   
    copy = data.copy()                      #Copy data to avoid messing up original cube
    usewav = np.ones_like(W,dtype=bool)     #Boolean array for whether or not to use wavelengths in fitting    
    ydist = 3600*np.sqrt( np.cos(head["CRVAL2"]*np.pi/180)*head["CD1_2"]**2 + head["CD2_2"]**2 ) #X & Y pixel sizes in arcseconds
    xdist = 3600*np.sqrt( np.cos(head["CRVAL2"]*np.pi/180)*head["CD1_1"]**2 + head["CD2_1"]**2 )
    
    medker=[5,5,5]
    medker[0] = int(round(medker[0]/head["CD3_3"])) #Convert kernel from Ang. to pixels
    medker[1] = int(round(medker[1]/ydist)) #Convert kernel from arcsec to pixels    
    medker[2] = int(round(medker[2]/xdist)) 

    #Make sure kernel sizes are odd
    for i,k in enumerate(medker): medker[i] = k+1 if k%2==0 else k


    
    ##### EXCLUDE BAD WAVELENGTHS AND EMISSION LINES
    usewav[ W < head["WAVGOOD0"] ] = 0
    usewav[ W > head["WAVGOOD1"] ] = 0
    
    #Ignore possible emission wavelengths
    if redshift!=None:
        for wav in lines:    
            wc = (redshift+1)*wav
            dw =(vwindow*1e5/3e10)*wc
            a,b = params.getband(wc-dw,wc+dw,head) 
            copy[a:b] = 0
            usewav[a:b] = 0
            
    #Eliminate sky-lines
    for wc in skylines:
        dw = 5
        a,b = params.getband(wc-dw,wc+dw,head)
        copy[a:b] = 0
        usewav[a:b] = 0

    #Mask data before filtering
    for yi in Y:
        for xi in X:
            if mask[yi,xi]:
                copy[:,yi,xi] = 0
    
    for yi in Y:
        med = np.median(copy[:,yi,:])
        std = np.std(copy[:,yi,:])
        plt.figure()
        plt.pcolor(copy[:,yi,:],vmin=med-std,vmax=med+std)
        plt.colorbar()
        plt.show()
    #Get median-filtered cube            
    #meddata = scipy.signal.medfilt(copy,kernel_size=medker)
    
    
    data -= meddata
    bmodel = meddata
    
    return data,bmodel
            
#Return a 3D cube which is a simple 1D polynomial fit to each 2D spaxel            
def polyModel(cube,k=3,w0=0,w1=-1):

    print "\tPolyFit to masked cube. Slice:",

    #Useful data structures
    w,y,x = cube.shape
    model = np.zeros_like(cube)
    W = np.arange(w)
    
    #Optimizer and model
    fitter = fitting.LinearLSQFitter()
    p = models.Polynomial1D(degree=k) #Initialize model
    
    #Run through spaxels and fit
    for yi in range(y):
        print yi,
        sys.stdout.flush()
        for xi in range(x):        
            p = fitter(p,W[w0:w1],cube[w0:w1,yi,xi])     
            model[w0:w1,yi,xi] = p(W[w0:w1])
    print ""
    
    #Return model
    return model
         
            
            
            

    

        
