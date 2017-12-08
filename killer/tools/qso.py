from astropy.modeling import models,fitting
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from scipy.optimize import least_squares,curve_fit
from sys import platform

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

#LINUX OS
if platform == "linux" or platform == "linux2":
    from matplotlib.figure import Figure
    from matplotlib.widgets import Button,SpanSelector,Cursor,Slider
    plt.style.use("ggplot")
#MAC OS
elif platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.widgets import Button,SpanSelector,Cursor,Slider

#WINDOWS
elif platform == "win32":
    print("This code has not been tested on Windows. God's speed, you damn brave explorer.")
    from matplotlib.figure import Figure
    from matplotlib.widgets import Button,SpanSelector,Cursor,Slider



class qsoFinder():
    
    #Initialize QSO finder class
    def __init__(self,fits,z=None,title=None):
        
        #Define some hardcoded variables
        self.dy = 2 #Slices to sum in x prof
        self.dx = 3 #Pixels to sum in y prof       
        #Extract raw data from fits to class structures
        self.fits = fits
        if z!=None: self.z = z
        else: self.z=-1
        if title!=None: self.title = title
        else: self.title = ""
        self.data = fits[0].data
        self.head = fits[0].header
        #Initialize figure
        self.fig = plt.figure()             
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
        self.xplot.set_xlim([1,self.X[-1]])
        self.xplot.set_ylim([-0.1,1.1])
        self.xplot.set_title(self.title)
        plt.tick_params( labelleft='off', labelbottom='off',labeltop='off' )
        
        self.yplot = self.fig.add_subplot(gs[plot_size:-plot_size-1,-plot_size:])
        self.yplot.set_ylim([1,self.Y[-1]])
        self.yplot.set_xlim([-0.1,1.1])
        plt.tick_params( labelleft='off', labelbottom='off',labeltop='off',labelright='off')
        
        self.splot = self.fig.add_subplot(gs[-plot_size-1:,sidebar_size:-1])
        self.splot.set_xlim([self.W[0],self.W[-1]])
        self.splot.set_xlabel("Drag-select spectral band to use for image")
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
        self.smooth_slider = Slider(self.smooth_grid,'',0.0,5.0,valinit=self.smooth)
        self.smooth_slider.on_changed(self.update_smooth)
        self.smooth_slider.ax.set_xlabel("Smoothing Scale",fontsize=10)
        self.smooth_slider.valtext.set_text("")

    def init_data(self):
        LyA = 1216
        Nsmooth = 1000
        dW = 25 #Half window width in Angstrom
        
        #Get cube dimensions
        Nw,Ny,Nx = self.data.shape
        
        #Create wavelength, X and Y domains
        self.W = np.array( [ self.head["CRVAL3"] + i*self.head["CD3_3"] for i in range(Nw) ] )
        self.X = np.arange(Nx) + 1 #Shift into 1-indexed arrays (python uses 0 but DS9 etc. use 1)
        self.Y = np.arange(Ny) + 1
        
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
        
        rx = min(10,self.x,self.X[-1]-self.x) #Make sure selection for fitting does not wrap around
        ry = min(10,self.y,self.Y[-1]-self.y)
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
    
    def update_xdata(self):
        self.xdata = np.sum(self.im[self.y-self.dy:self.y+self.dy+1],axis=0)
        self.xdata /= np.sum(self.xdata) #Normalize
    def update_ydata(self):
        self.ydata = np.sum(self.im[:,self.x-self.dx:self.x+self.dx+1],axis=1)
        self.ydata /= np.max(self.ydata) #Normalize

    def update_sdata(self): self.sdata = np.sum(np.sum(self.data[:,self.y-self.dy:self.y+self.dy+1,self.x-self.dx:self.x+self.dx+1],axis=1),axis=1)
    
    def update_cmap(self):
        self.im = np.sum(self.data[self.w1i:self.w2i],axis=0)        
        if self.smooth>0.0: self.im = gaussian_filter(self.im,self.smooth)
            
    def update_pos(self,xi,yi):
        self.x = int(round(xi))+1
        self.y = int(round(yi))+1
        self.update_xdata()
        self.update_ydata()
        self.update_sdata()
        self.model_data()
        self.update_plots()

##################################################################################################                       
def qsoSubtract(fits,pos,instrument,redshift=None,wx=1,vwindow=2000,returnqso=False,limit=1e-6,plot=False):
    
    ##### DEFINE CONSTANTS
    rx=20
       
    ##### DEFINE METHODS
    def moffat(r,I0,r0,a,b): return I0*(1 + ((r-r0)/a)**2)**(-b)
    def line(x,m,c): return m*x + c

    data = fits[0].data #data cube
    head = fits[0].header #header
    
    #ROTATE (TEMPORARILY) SO THAT AXIS 2 IS 'IN-SLICE' for KCWI DATA
    if instrument=='KCWI':
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
    
    xc,yc = pos
    
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
    except:
        print "Error cropping to good wavelength range for subtraction"
        
   
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
                plt.title(r"$A=%.4f,d\lambda=%.3fpx$" % (A,dw))
                plt.plot(W,spec,'kx',alpha=0.5)
                plt.plot(W[usewav==1],spec[usewav==1],'kx')
                plt.plot(W,A*q_spec,'g-',alpha=0.8)
                plt.plot(W,m_spec2,'r-')
                plt.xlim([W[0],W[-1]])

                plt.subplot(312)
                plt.xlim([W[0],W[-1]])

                plt.plot(W,residual2,'gx-')                                      
                plt.subplot(313)
                plt.hist(residual2)

                plt.tight_layout()           
                plt.show()
                            
            data[:,yi,xi] -= m_spec2
            qsoc[:,yi,xi] += m_spec2     
        
    #ROTATE BACK IF ROTATED AT START
    if instrument=='KCWI':
        data_rot = np.zeros( (data.shape[0],data.shape[2],data.shape[1]) )
        qsoc_rot = np.zeros( (data.shape[0],data.shape[2],data.shape[1]) )
        for wi in range(len(data)):
            data_rot[wi] = np.rot90( data[wi], k=1 )
            qsoc_rot[wi] = np.rot90( data[wi], k=1 )
        data = data_rot
 
    #Return either the data cube or data cube and qso model                                                        
    if returnqso: return (data,qsoc)
    else: return data       
