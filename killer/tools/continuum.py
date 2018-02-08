##################################################################################################                       
# GIVEN FITS FILE AND 2D LOCATION - SUBTRACT CONTINUUM EMISSION AROUND THAT POINT
#
# 1. Refine location
# 2. Model continuum emission of source
# 3. Subtract 
#
from scipy.optimize import least_squares
import numpy as np
import params

lines = [1216]

def cSubtract(fits,pos,redshift=None,vwindow=500,radius=5,mode='scale2D'):
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
    
    Xs = np.linspace(X[0],X[-1],10*x) #Smooth X-Y domains for PSF modelling
    Ys = np.linspace(Y[0],Y[-1],10*y)
    
    y_dist = 3600*np.sqrt( np.cos(head["CRVAL2"]*np.pi/180)*head["CD1_2"]**2 + head["CD2_2"]**2 ) #X & Y pixel sizes in arcseconds
    x_dist = 3600*np.sqrt( np.cos(head["CRVAL2"]*np.pi/180)*head["CD1_1"]**2 + head["CD2_1"]**2 )
    
    ry = int(round(radius/y_dist)) #X and Y 'radius' extent in pixels 
    rx = int(round(radius/x_dist))

    xc,yc = pos #Take input position tuple
    
    if mode=='scale2D':
    
        ##### CREATE CROPPED CUBE 
        
        cube = data[:,yc-ry:yc+ry+1,xc-rx:xc+rx+1].copy()   #Create smaller working cube to isolate continuum source
        contwavs = np.ones(len(data),dtype=bool)            #Create boolean index for continuum wavelengths  
        cube-=np.median(cube)                               #Median subtract the cube
        
        ##### EXCLUDE EMISSION LINE WAVELENGTHS
        
        usewav = np.zeros_like(W)
        if redshift!=None:
            for line in lines:    
                wc = (redshift+1)*line
                dw =(vwindow*1e5/3e10)*wc
                a,b = params.getband(wc-dw,wc+dw,head) 
                contwavs[a:b] = 0
                
        ##### CREATE 2D CONTINUUM IMAGE   
        
        cont2d = np.mean(cube[contwavs],axis=0) #Create 2D continuum image
        
        cmodel = np.zeros_like(data)            #Cube to store 3D continuum model
             
        ##### BUILD 3D CONTINUUM MODEL
        
        for i in range(cube.shape[0]):
           
            chisq = lambda A: np.sum(cube[i] - A*cont2d) #Optimization function
            
            A0 = np.sum(cube[i])/np.sum(cont2d) #Initial guess
            
            fit = least_squares(chisq,A0,jac='3-point') #Run the fit    

            Afit = fit.x #Get optimzed scaling factor
            
            model = Afit*cont2d #Add this wavelength layer to the model
            
            data[i,yc-ry:yc+ry+1,xc-rx:xc+rx+1] -= model #Subtract from data cube
        
            cmodel[i,yc-ry:yc+ry+1,xc-rx:xc+rx+1] += model #Add to larger model cube
            
        return data,cmodel

        
