#from astropy.convolution import convolve,Box1DKernel
from astropy.modeling import models,fitting
from matplotlib.figure import Figure
from matplotlib.widgets import Button,SpanSelector,Cursor,Slider
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from scipy.ndimage.interpolation import shift
from scipy.ndimage.measurements import center_of_mass
from scipy.optimize import least_squares,curve_fit
from scipy.signal import correlate,deconvolve,convolve,gaussian

import astropy
import difflib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys

global h_ra, h_dec, h_pa, h_imgnum

#######################################################################
#Output image as fits
def saveFits(data,path,header=""):
        print("""Saving %s""" % path)
        hdu = astropy.io.fits.PrimaryHDU(data)
        hdulist = astropy.io.fits.HDUList([hdu])
        hdulist[0].header = header
        hdulist.writeto(path,clobber=True)
        
#######################################################################
# Check for incomplete parameter data
def paramsMissing(params):
    #Search for default values in params dictionary and return 'incomplete' flag if found
    if '?' in params["INST"] or\
       '-' in params["WCROP"] or\
       '-' in params["XCROP"] or\
       '-' in params["YCROP"] or\
        -1 in params["PA"] or\
       -99 in params["QSO_XA"] or\
       -99 in params["QSO_YA"]: return True
    else: return False

#######################################################################
# Parse FITS headers for relevant stacking & subtracting parameters
def parseHeaders(params,fits):
    
    for i,f in enumerate(fits):
        
        header = f[0].header
        
        params["INST"][i] = header["INSTRUME"] #Get instrument

                    
        #Hardcode these for now - update later to use inst.config files
        if params["INST"][i]=="PCWI":
            params["PA"][i] = int(header["ROTPA"])
            params["XCROP"][i] = "10:-12"
            params["YCROP"][i] = "0:24"
            w0,w1 = header["WAVGOOD0"],header["WAVGOOD1"]
            w0i,w1i = getband(w0,w1,header)
            params["WCROP"][i] = "%i:%i" % (w0i,w1i)
        
        elif params["INST"][i]=="KCWI":
            params["PA"][i] = int(header["ROTPOSN"])
            params["XCROP"][i] = "0:-1"
            params["YCROP"][i] = "0:24"
            w0,w1 = header["WAVGOOD0"],header["WAVGOOD1"]
            w0i,w1i = getband(w0,w1,header)
            params["WCROP"][i] = "%i:%i" % (w0i,w1i)
            
    return params
            
            
                    
##################################################################################################
def writeparams(params,parampath):
    global param_keys
    
    paramfile = open(parampath,'w')

    print("Writing parameters to %s" % parampath)
      
    paramfile.write("#############################################################################################\
    \n# TARGET PARAMETER FILE FOR STACKING & SUBTRACTING (KiLLER Pipeline)\n\n""")

    paramfile.write("name = %s # Target name\n" % params["NAME"])
    paramfile.write("ra   = %.6f # Decimal RA\n" % params["RA"])
    paramfile.write("dec  = %.6f # Decimal DEC\n" % params["DEC"])
    paramfile.write("z    = %.6f # Redshift\n\n" % params["Z"])
    
    paramfile.write("data_dir = %s # Location of raw data cubes\n\n" % params["DATA_DIR"])
    
    paramfile.write("product_dir = %s # Where to store stacked and subtracted cubes\n\n" % params["PRODUCT_DIR"])
    
    paramfile.write("#############################################################################################\n")   
    paramfile.write("#%9s%10s%10s%10s%10s%10s%10s%10s%10s%10s\n"\
    % ("IMGNUM","INST","PA","QSO_X","QSO_Y","XCROP","YCROP","WCROP","QSO_XA","QSO_YA"))
    
    imgnums = params["IMGNUM"]
    keys = ["IMGNUM","INST","PA","QSO_X","QSO_Y","XCROP","YCROP","WCROP","QSO_XA","QSO_YA"]
    keystr = ">    %05i%10s%10i%10.2f%10.2f%10s%10s%10s%10.2f%10.2f\n"
    for key in keys:
    
        if params.has_key(key) and len(params[key])==len(params["IMGNUM"]): pass
        elif key=="INST": params[key] =  [ '?' for i in range(len(imgnums))]
        elif key=="PA": params[key] =  [ -1 for i in range(len(imgnums))]
        elif "CROP" in key: params[key] = [ '-' for i in range(len(imgnums))]
        else: params[key] = [ -99 for i in range(len(imgnums))] 

    for i in range(len(imgnums)): paramfile.write( keystr % tuple(params[keys[j]][i] for j in range(len(keys))))
    paramfile.close()

##################################################################################################   
def loadparams(parampath):
    
    paramfile = open(parampath,'r')

    print("Loading target parameters from %s" % parampath)
    
    params = {}
    params["IMGNUM"] = []
    params["INST"]   = []
    params["PA"]     = []
    params["QSO_X"]  = []
    params["QSO_Y"]  = []
    params["XCROP"]  = []
    params["YCROP"]  = []
    params["WCROP"]  = []
    params["QSO_XA"] = []
    params["QSO_YA"] = []
    
    for line in paramfile:
    
        if "=" in line:
            keyval = line.split("#")[0].replace(" ","")
            key,val = keyval.split("=")
            params[key.upper()] = float(val) if key in ['ra','dec','z'] else val

        elif line[0]=='>' and len(line[1:].split())==10:

            imgnum,inst,pa,qsox,qsoy,xcrop,ycrop,wcrop,qsoxa,qsoya = line[1:].split()
            params["IMGNUM"].append(int(imgnum))
            params["INST"].append(inst)
            params["PA"].append(int(pa))
            params["QSO_X"].append(float(qsox))
            params["QSO_Y"].append(float(qsoy))
            params["XCROP"].append(xcrop)
            params["YCROP"].append(ycrop)
            params["WCROP"].append(wcrop)
            params["QSO_XA"].append(float(qsoxa))
            params["QSO_YA"].append(float(qsoya))
            
        elif line[0]=='>' and len(line.split())==2:
        
            params["IMGNUM"].append(int(line[1:].split()[0]))

    print params["IMGNUM"]
    #If some image numbers have been added but not properly written to param file...
    if len(params["IMGNUM"]) > len(params["QSO_YA"]):
    
        paramfile.close() #Close the parameter file
        
        writeparams(params,parampath) #Rewrite the file to fit the correct format
    
        return loadparams(parampath) #Try to reload params and return those
        
    #Otherwise just return the loaded params
    else: return params
    
##################################################################################################     
def getband(_w1,_w2,_hd):
    w0,dw = _hd["CRVAL3"],_hd["CD3_3"]
    return ( int((_w1-w0)/dw), int((_w2-w0)/dw) )
        

##################################################################################################
def findfiles(params,cubetype):

    target = params["NAME"]
    datadir = params["DATA_DIR"]
    imgnums = params["IMGNUM"]  
    
    all_files = os.listdir(datadir)

    target_files = [ "" for i in range(len(params["IMGNUM"])) ]
    for f in all_files:
        if cubetype in f:
            for i,num in enumerate(imgnums):
                if str(num) in f: target_files[i] = os.path.abspath("%s/%s" % (datadir,f))
    
    return target_files             


##################################################################################################                
class qsoFinder():
    
    #Initialize QSO finder class
    def __init__(self,fits,z=None):       
        #Define some hardcoded variables
        self.dy = 2 #Slices to sum in x prof
        self.dx = 3 #Pixels to sum in y prof       
        #Extract raw data from fits to class structures
        self.fits = fits
        if z!=None: self.z = z
        else: self.z=-1 
        self.data = fits[0].data
        self.head = fits[0].header
        #Initialize figure
        self.fig = plt.figure()     
        #print self.head["IMGNUM"]        
        self.fig.canvas.set_window_title('QSO Finder 2.0')
        #Connect click event to handler
        self.fig.canvas.mpl_connect('button_press_event',self.onclick)
        mng = plt.get_current_fig_manager()
        mng.resize(*(800,800))#*mng.window.maxsize())              
        #Initialize data
        self.init_data()        
        #Model current data
        self.model_data()                
        #Initialize plots
        self.init_plots()
        self.update_plots()
        
    def run(self):      
        #Show figure
        self.fig.show()       
        #Fetch updates from plot until finished (i.e. 'ok' button is pressed)
        self.finished = False
        while not self.finished: self.fig.canvas.get_tk_widget().update()
        #Close figure when loop has been exited
        plt.close()
		    
        return [self.x_opt,self.y_opt]

    def spanSelect(self,wmin,wmax):
        self.w1 = wmin
        self.w2 = wmax
        self.w1i = self.getIndex(self.w1,self.W)
        self.w2i = self.getIndex(self.w2,self.W)
        self.update_cmap()
        self.update_plots()
    
              	                
    def init_plots(self):    
        #Establish relative sizes of different subcomponents
        button_size = 1
        sidebar_size = 0
        plot_size = 1
        map_size = 4        
        #Calculate total grid dimensions from components
        grid_height = button_size + plot_size*2 + map_size
        grid_width = sidebar_size + map_size + plot_size
    
        #Lay out plots
        gs = gridspec.GridSpec(grid_height,grid_width)   
        
        self.xplot = self.fig.add_subplot(gs[:plot_size,sidebar_size:-1])
        self.xplot.set_xlim([0,self.X[-1]])
        plt.tick_params( labelleft='off', labelbottom='off',labeltop='off' )
        
        self.yplot = self.fig.add_subplot(gs[plot_size:-plot_size-1,-plot_size:])
        self.yplot.set_ylim([0,self.Y[-1]])
        plt.tick_params( labelleft='off', labelbottom='off',labeltop='off',labelright='off')
        
        self.splot = self.fig.add_subplot(gs[-plot_size-1:,sidebar_size:-1])
        self.splot.set_xlim([self.W[0],self.W[-1]])
        plt.tick_params( labelleft='off', labelbottom='on',labeltop='off',labelright='off')    
        
        #Add span selector to spectral plot
        self.span = SpanSelector(self.splot, self.spanSelect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))          
        
        self.cmap = self.fig.add_subplot(gs[plot_size:-plot_size-1,sidebar_size:-plot_size])
        self.cmap.set_xlim([0,self.X[-1]])
        self.cmap.set_ylim([0,self.Y[-1]])
        plt.tick_params( labelleft='off', labelbottom='off',labeltop='off',labelright='off')
        self.cursor = Cursor(self.cmap, useblit=True, color='red', linewidth=1)
       
        #Insert 'skip' button 
        self.skip_grid = self.fig.add_subplot(gs[0,-1]) #Place for button
        self.skip_btn = Button(self.skip_grid,'SKIP')

        #Insert 'ok' button 
        self.ok_grid = self.fig.add_subplot(gs[-1,-1]) #Place for button
        self.ok_btn = Button(self.ok_grid,'OK')

        #Insert slider for smoothing scale 
        self.smooth_grid = self.fig.add_subplot(gs[-2,-1]) #Place for slider
        self.smooth_slider = Slider(self.smooth_grid,'Smooth',0.0,5.0,valinit=self.smooth)
        self.smooth_slider.on_changed(self.update_smooth)
    
    def init_data(self):
        LyA = 1216
        Nsmooth = 1000
        dW = 25 #Half window width in Angstrom
        
        #Get cube dimensions
        Nw,Ny,Nx = self.data.shape
        
        #Create wavelength, X and Y domains
        self.W = np.array( [ self.head["CRVAL3"] + i*self.head["CD3_3"] for i in range(Nw) ] )
        self.X = np.arange(Nx)
        self.Y = np.arange(Ny)
        
        #Create smooth domains from these limits
        self.Xs = np.linspace(self.X[0],self.X[-1],Nsmooth)
        self.Ys = np.linspace(self.Y[0],self.Y[-1],Nsmooth)
        
        #Get initial wavelength window around LyA and make pseudo-NB
        
        if self.z!=-1:
            self.w1 = (1+self.z)*LyA - dW
            self.w2 = (1+self.z)*LyA + dW
        else:
            self.w1 = self.W[Nw/2] - dW
            self.w2 = self.W[Nw/2] + dW
            
        self.w1i = self.getIndex(self.w1,self.W)
        self.w2i = self.getIndex(self.w2,self.W)
        
        self.smooth=0.0
        self.update_cmap()

        
        #Get initial positions for x,y,w1 and w1
        self.x = np.nanargmax(np.sum(self.im,axis=0))       
        self.y = np.nanargmax(np.sum(self.im,axis=1))
       
        self.update_xdata()
        self.update_ydata()
        self.update_sdata()

    def model_data(self):
        
        rx = 10
        ry = 10
        fitter = fitting.SimplexLSQFitter()
	    
        #Try to fit Moffat in X direction
        try:
            moffatx_init = models.Moffat1D(1.2*np.max(self.xdata), self.x, 1.0, 1.0)
            moffatx_fit = fitter(moffatx_init,self.X[self.x-rx:self.x+rx],self.xdata[self.x-rx:self.x+rx])
            self.xmoff = moffatx_fit(self.Xs)
            self.x_opt = moffatx_fit.x_0.value
        except:
            self.xmoff = np.zeros_like(self.Xs)
            self.x_opt = self.x

        #Try to fit Moffat in Y direction
        try:
            self.ydata -= np.median(self.ydata)          
            moffaty_init = models.Moffat1D(1.2*np.max(self.ydata), self.y, 1.0, 1.0)  
            moffaty_fit = fitter(moffaty_init,self.Y[self.y-ry:self.y+ry],self.ydata[self.y-ry:self.y+ry])
            self.ymoff = moffaty_fit(self.Ys)
            self.y_opt = moffaty_fit.x_0.value         
        except:
            self.ymoff = np.zeros_like(self.Ys)
            self.y_opt = self.y

    def update_plots(self):
        self.init_plots() #Clear and reset all plots
        self.xplot.plot(self.X,self.xdata,'ko')
        self.xplot.plot(self.Xs,self.xmoff,'b-')
        self.xplot.plot([self.x_opt,self.x_opt],[np.min(self.xdata),np.max(self.xdata)],'r-')
        self.yplot.plot(self.ydata,self.Y,'ko')
        self.yplot.plot(self.ymoff,self.Ys,'b-')
        self.yplot.plot([np.min(self.ydata),np.max(self.ydata)],[self.y_opt,self.y_opt],'r-')
        self.splot.plot(self.W,self.sdata,'ko')
        self.splot.plot([self.w1,self.w1],[np.min(self.sdata),np.max(self.sdata)],'r-')
        self.splot.plot([self.w2,self.w2],[np.min(self.sdata),np.max(self.sdata)],'r-')
        self.cmap.pcolor(self.im)
        self.fig.canvas.draw()
        
    def getIndex(self,wi,W): return np.nanargmin( np.abs(W-wi) )

    def onclick(self,event):

        if event.inaxes==self.ok_grid: self.finish()
        elif event.inaxes==self.skip_grid: self.skip()
        elif event.inaxes==self.cmap: self.update_pos(event.xdata,event.ydata)
    
    def skip(self):
        self.finished=True
        self.x_opt,self.y_opt = -99,-99

    def finish(self): self.finished = True

    def update_smooth(self,val):
        self.smooth = val
        self.update_cmap()
    
    def update_xdata(self): self.xdata = np.sum(self.im[self.y-self.dy:self.y+self.dy+1],axis=0)
    
    def update_ydata(self): self.ydata = np.sum(self.im[:,self.x-self.dx:self.x+self.dx+1],axis=1)
    
    def update_sdata(self): self.sdata = np.sum(np.sum(self.data[:,self.y-self.dy:self.y+self.dy+1,self.x-self.dx:self.x+self.dx+1],axis=1),axis=1)
    
    def update_cmap(self):
        self.im = np.sum(self.data[self.w1i:self.w2i],axis=0)        
        if self.smooth>0.0: self.im = gaussian_filter(self.im,self.smooth)
            
    def update_pos(self,xi,yi):
        self.x = xi
        self.y = yi
        self.update_xdata()
        self.update_ydata()
        self.update_sdata()
        self.model_data()
        self.update_plots()
        
 
##################################################################################################                       
def qsoSubtract(fits,inst='PCWI',pos=None,redshift=None,wx=1,vwindow=2000,returnqso=False,limit=1e-6,plot=False):
    
    ##### DEFINE CONSTANTS
    rx=20
       
    ##### DEFINE METHODS
    def moffat(r,I0,r0,a,b): return I0*(1 + ((r-r0)/a)**2)**(-b)
    def line(x,m,c): return m*x + c

    data = fits[0].data #data cube
    head = fits[0].header #header
    
    #ROTATE (TEMPORARILY) SO THAT AXIS 2 IS 'IN-SLICE' for KCWI DATA
    if inst=='KCWI':
        data_rot = np.zeros( (data.shape[0],data.shape[2],data.shape[1]) )
        for wi in range(len(data)): data_rot[wi] = np.rot90( data[wi], k=3 )
        data = data_rot    
        pos = (pos[1],pos[0])
        
    ##### EXTRACT DATA FROM FITS
    backup = data.copy()        
    head = fits[0].header #header
    qsoc = np.zeros_like(data) #Cube for QSO model
    w,y,x = data.shape #Cube dimensions
    X = np.arange(x) #Create domains X,Y and W
    Xs = np.linspace(X[0],X[-1],10*x)
    Y = np.arange(y)
    Ys = np.linspace(Y[0],Y[-1],10*y)
    W = np.array([ head["CRVAL3"] + i*head["CD3_3"] for i in range(w)])
    
    fits[0].data = data
    
    ##### GET QSO CENTER
    if pos==None:    
        qfinder = qsoFinder(fits) #Get QSO Finder tool
        xc,yc = qfinder.run() #Run tool to get x,y pos of qso
    else: xc,yc = pos #If (x,y) provided as input, just assign
    ycen = yc 

    xc = int(round(xc))
    yc = int(round(yc))     
    ##### GET QSO SPECTRUM
    q_spec = data[:,yc,xc].copy()

    ##### EXCLUDE LYA+/-v WAVELENGTHS
    usewav = np.zeros_like(q_spec)
    if redshift!=None:
        lyA = (redshift+1)*1216
        vwav =(vwindow*1e5/3e10)*lyA
        w1,w2 = lyA-vwav/2,lyA+vwav
        usewav[W < w1] = 1 #Use wavelengths below lower limit
        usewav[W > w2] = 1 #Use wavelengths above upper limit
    else: usewav[:] = 1 #Use all wavelengths if no redshift provided

    ##### MAKE CONTINUUM IMAGE
    cont_img = np.sum(data[usewav==1],axis=0)

    ##### CROP TO 'WAVGOOD' RANGE ONLY - IF AVAILABLE
    try:
        wg0,wg1 = head["WAVGOOD0"],head["WAVGOOD1"]
        usewav[ W < wg0 ] = 0 #Exclude lower wavelengths
        usewav[ W > wg1 ] = 0 #Exclude upper wavelengths
    except: pass
   
    q_spec_fit = q_spec[usewav==1]
    
    #Run through slices
    for yi in Y:
        print yi,'/',Y[-1]
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
            chaisq = lambda x: s_spec_fit[10:-10] - x[0]*shift(q_spec_fit,x[1],order=4,mode='reflect')[10:-10]

            p0 = [np.max(s_spec)/np.max(q_spec),w_offset]
            
            lbound = [0.0,-5]
            ubound = [5.1, 5]        
            for j in range(len(p0)):
                if p0[j]<lbound[j]: p0[j]=lbound[j]
                elif p0[j]>ubound[j]: p0[j]=ubound[j]
            
            p_fit = least_squares(chaisq,p0,bounds=(lbound,ubound),jac='3-point')                

            #A,dw,sig =p_fit.x
            A,dw =p_fit.x
            

            q_spec_shifted = shift(q_spec_fit,dw,order=3,mode='reflect')
 
        else:
            q_spec_shifted = q_spec_fit
            dw=0
            
        lbound = [0.0,-5]
        ubound = [1.0,5]
                              
        for xi in X:

            spec = data[:,yi,xi]
            spec_fit = spec[usewav==1]
                          
            #First fit to find wav offset for this slice
            chaisq = lambda x: spec_fit - x[0]*shift(q_spec_fit,x[1],order=3,mode='reflect')

            p0 = [np.max(s_spec)/np.max(q_spec),dw]
            for j in range(len(p0)):
                if p0[j]<lbound[j]: p0[j]=lbound[j]
                elif p0[j]>ubound[j]: p0[j]=ubound[j]
            
               
            p_fit = least_squares(chaisq,p0,bounds=(lbound,ubound),jac='3-point')             
            A,dw = p_fit.x
            
            m_spec = A*shift(q_spec,dw,order=4,mode='reflect')
            
            #Do a linear fit to residual and correct linear errors
            residual = data[:,yi,xi]-m_spec
            ydata = residual
            xdata = W
            popt,pcov = curve_fit(line,xdata,ydata,p0=(0.0,0.0))
            linefit = line(W,popt[0],popt[1])
                       
            m_spec2 = linefit+m_spec
            residual2 = data[:,yi,xi] - m_spec2

            if plot:

                plt.figure(figsize=(16,8))
                plt.subplot(311)
                #plt.title(r"$A=%.2f,\sigma=%.3f,d\lambda=%.3fpx$" % (A,sig,dw))
                plt.title(r"$A=%.4f,d\lambda=%.3fpx$" % (A,dw))
                plt.plot(W,spec,'kx',alpha=0.5)
                plt.plot(W[usewav==1],spec[usewav==1],'kx')
                plt.plot(W,A*q_spec,'g-',alpha=0.8)
                #plt.plot(W,(A*np.max(q_spec)/np.max(backup[:,yi,xc]))*backup[:,yi,xc],'b-',alpha=0.5)
                plt.plot(W,m_spec2,'r-')
                plt.xlim([W[0],W[-1]])
                #plt.xlim([4380,4480])
                plt.subplot(312)
                plt.xlim([W[0],W[-1]])
                #plt.xlim([4380,4480])
                plt.plot(W,residual2,'gx-')                                      
                plt.subplot(313)
                plt.hist(residual2)
                #plt.subplot(322)
                #plt.plot(range(-dx,dx),np.sum(data[usewav==1,yi,xc-dx:xc+dx],axis=0))   
                #plt.subplot(324)
                #plt.plot(range(-dx,dx),dws,'ko-')
                #plt.plot(np.arange(-dx,dx)[dws!=0],dws[dws!=0],'ro')
                #plt.subplot(326)
                #plt.xlim([W[0],W[-1]])
                #plt.plot(W,res2,'gx-')
                plt.tight_layout()           
                plt.show()
                            
            data[:,yi,xi] -= m_spec2
            qsoc[:,yi,xi] += m_spec2     
        
    #ROTATE BACK IF ROTATED AT START
    if inst=='KCWI':
        data_rot = np.zeros( (data.shape[0],data.shape[2],data.shape[1]) )
        qsoc_rot = np.zeros( (data.shape[0],data.shape[2],data.shape[1]) )
        for wi in range(len(data)):
            data_rot[wi] = np.rot90( data[wi], k=1 )
            qsoc_rot[wi] = np.rot90( data[wi], k=1 )
        data = data_rot
 
    #Return either the data cube or data cube and qso model                                                        
    if returnqso: return (data,qsoc)
    else: return data
    

