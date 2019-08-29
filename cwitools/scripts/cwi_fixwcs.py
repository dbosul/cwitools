
"""CWITools QSO-Finder class for interactive PSF fitting.

This module contains the class definition for the interactive tool 'QSO Finder.'
QSO finder is used to accurately locate point sources (usually QSOs) when
running fixWCS in CWITools.reduction.

"""


from astropy.modeling import models,fitting
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from scipy.ndimage.interpolation import shift
from scipy.optimize import least_squares,curve_fit
from scipy.signal import correlate,deconvolve,convolve,gaussian
from sys import platform

import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys

#LINUX OS
if platform == "linux" or platform == "linux2":
    from matplotlib.figure import Figure
    from matplotlib.widgets import Button,SpanSelector,Cursor,Slider

#MAC OS
elif platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.widgets import Button,SpanSelector,Cursor,Slider

#WINDOWS
elif platform == "win32":
    print("This code has not been tested on Windows. Good luck, you brave explorer.")
    from matplotlib.figure import Figure
    from matplotlib.widgets import Button,SpanSelector,Cursor,Slider

class qsoFinder():

    #Initialize QSO finder class
    def __init__(self,fits,z=-1,title=None):

        #Astropy Simplex LSQ Fitter for PSFs
        self.fitter = fitting.SimplexLSQFitter()

        #Hard-coded radius for fitting in arcsec
        fit_radius = 5 #arcsec

        #Hard-coded default NB window size in Angstrom
        wav_window = 30

        #Extract raw data from fits to class structures
        self.fits = fits
        if z!=None: self.z = z
        else: self.z=-1
        if title!=None: self.title = title
        else: self.title = ""
        self.data = fits[0].data
        self.head = fits[0].header

        #X & Y pixel sizes in arcseconds
        ydist = 3600*np.sqrt( np.cos(self.head["CRVAL2"]*np.pi/180)*self.head["CD1_2"]**2 + self.head["CD2_2"]**2 )
        xdist = 3600*np.sqrt( np.cos(self.head["CRVAL2"]*np.pi/180)*self.head["CD1_1"]**2 + self.head["CD2_1"]**2 )


        self.dy = int(round(fit_radius/ydist)) #Slices to sum in x prof
        self.dx = int(round(fit_radius/xdist)) #Pixels to sum in y prof
        self.dw = int(round(wav_window)/self.head["CD3_3"])

        if ydist>2: self.dy=3

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
        self.model_xData()
        self.model_yData()

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

    def spanSelectW(self,wmin,wmax):
        self.w1 = wmin
        self.w2 = wmax
        self.w1i = self.getIndex(self.w1,self.W)
        self.w2i = self.getIndex(self.w2,self.W)
        self.update_cmap()
        self.update_plots()

    def spanSelectX(self,xmin,xmax):
        self.x0 = int(round(xmin))
        self.x1 = int(round(xmax))
        self.update_ydata()
        self.update_xdata()
        self.model_yData()
        self.model_xData()
        self.update_plots()

    def spanSelectY(self,ymin,ymax):
        self.y0 = int(round(ymin))
        self.y1 = int(round(ymax))
        self.update_ydata()
        self.update_xdata()
        self.model_yData()
        self.model_xData()
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
        self.xplot.set_ylim([-0.1,1.0])
        self.xplot.set_title(self.title)
        plt.tick_params( labelleft='off', labelbottom='off',labeltop='off' )

        #Add span selector to spectral plot
        self.spanX = SpanSelector(self.xplot, self.spanSelectX, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='blue'))

        self.yplot = self.fig.add_subplot(gs[plot_size:-plot_size-1,-plot_size:])
        self.yplot.set_ylim([1,self.Y[-1]])
        self.yplot.set_xlim([-0.1,1.1])
        plt.tick_params( labelleft='off', labelbottom='off',labeltop='off',labelright='off')

        #Add span selector to spectral plot
        self.spanY = SpanSelector(self.yplot, self.spanSelectY, 'vertical', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='blue'))

        self.splot = self.fig.add_subplot(gs[-plot_size-1:,sidebar_size:-1])
        self.splot.set_xlim([self.W[0],self.W[-1]])
        self.splot.set_xlabel("Drag-select spectral band to use for image")
        plt.tick_params( labelleft='off', labelbottom='on',labeltop='off',labelright='off')

        #Add span selector to spectral plot
        self.spanW = SpanSelector(self.splot, self.spanSelectW, 'horizontal', useblit=True,
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

        #Get cube dimensions
        Nw,Ny,Nx = self.data.shape

        #Create wavelength, X and Y domains
        self.W = np.array([ self.head["CRVAL3"] + self.head["CD3_3"]*(i - self.head["CRPIX3"]) for i in range(Nw)])
        self.X = np.arange(Nx)
        self.Y = np.arange(Ny)

        #Create smooth domains from these limits
        self.Xs = np.linspace(self.X[0],self.X[-1],Nsmooth)
        self.Ys = np.linspace(self.Y[0],self.Y[-1],Nsmooth)

        #Get initial wavelength window around LyA and make pseudo-NB

        if self.z!=-1 and self.z>2:
            self.w1 = (1+self.z)*LyA - self.dw
            self.w2 = (1+self.z)*LyA + self.dw
        else:
            self.w1 = self.W[int(Nw/2)] - self.dw
            self.w2 = self.W[int(Nw/2)] + self.dw

        self.w1i = self.getIndex(self.w1,self.W)
        self.w2i = self.getIndex(self.w2,self.W)

        self.smooth=0.0
        self.update_cmap()


        #Get initial positions for x,y,w1 and w1
        self.x = np.nanargmax(np.sum(self.im,axis=0))
        self.y = np.nanargmax(np.sum(self.im,axis=1))

        #Initialize upper and lower bounds for fitting PSF
        self.x0 = max(0,self.x - self.dx)
        self.x1 = min(Nx-1,self.x + self.dx)

        self.y0 = max(0,self.y - self.dy)
        self.y1 = min(Ny-1,self.y + self.dy)

        self.update_xdata()
        self.update_ydata()
        self.update_sdata()

    def model_xData(self):

        #Try to fit Moffat in X direction
        #try:
        moffatx_init = models.Moffat1D(1.2*np.max(self.xdata[self.x0:self.x1]), self.x, 1.0, 1.0)
        moffatx_init.x_0.max = self.x1
        moffatx_init.x_0.min = self.x0
        moffatx_init.amplitude.min = 0

        moffatx_fit = self.fitter(moffatx_init,self.X[self.x0:self.x1],self.xdata[self.x0:self.x1])
        self.xmoff = moffatx_fit(self.Xs)
        self.x_opt = moffatx_fit.x_0.value

        #except:
        #    self.xmoff = np.zeros_like(self.Xs)
        #    self.x_opt = self.x

    def model_yData(self):

        #Try to fit Moffat in Y direction
        try:
            moffaty_init = models.Moffat1D(1.2*np.max(self.ydata[self.y0:self.y1]), self.y, 1.0, 1.0)
            moffaty_init.x_0.max = self.y1
            moffaty_init.x_0.min = self.y0
            moffaty_init.amplitude.min = 0
            moffaty_fit = self.fitter(moffaty_init,self.Y[self.y0:self.y1],self.ydata[self.y0:self.y1])
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
        self.xplot.plot([self.x,self.x],[np.min(self.xdata),np.max(self.xdata)],'r--')
        self.xplot.plot([self.x0,self.x0],[np.min(self.xdata),np.max(self.xdata)],'b--')
        self.xplot.plot([self.x1,self.x1],[np.min(self.xdata),np.max(self.xdata)],'b--')
        self.xplot.set_ylim([0,np.max(self.xdata[self.x0:self.x1])*1.2])

        self.yplot.plot(self.ydata,self.Y,'ko')
        self.yplot.plot(self.ymoff,self.Ys,'b-')
        self.yplot.plot([np.min(self.ydata),np.max(self.ydata)],[self.y_opt,self.y_opt],'r-')
        self.yplot.plot([np.min(self.ydata),np.max(self.ydata)],[self.y,self.y],'r--')
        self.yplot.plot([np.min(self.ydata),np.max(self.ydata)],[self.y0,self.y0],'b--')
        self.yplot.plot([np.min(self.ydata),np.max(self.ydata)],[self.y1,self.y1],'b--')
        self.yplot.set_xlim([np.min(self.ydata),np.max(self.ydata)])
        self.yplot.set_ylim([0,self.Y[-1]])

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
        self.xdata = np.mean(self.im[self.y0:self.y1],axis=0)
        self.xdata -= np.median(self.xdata)
        self.xdata[self.xdata<0] = 0
        self.xdata /= np.max(self.xdata) #Normalize
    def update_ydata(self):
        self.ydata = np.mean(self.im[:,self.x0:self.x1],axis=1)
        self.ydata -= np.median(self.im,axis=1)
        self.ydata[self.ydata<0] = 0
        self.ydata /= np.max(self.ydata) #Normalize

    def update_sdata(self): self.sdata = np.sum(np.sum(self.data[:,self.y0:self.y1,self.x0:self.x1],axis=1),axis=1)

    def update_cmap(self):
        self.im = np.sum(self.data[self.w1i:self.w2i],axis=0)
        self.im -= np.median(self.im)
        if self.smooth>0.0: self.im = gaussian_filter(self.im,self.smooth)

    def update_pos(self,xi,yi):
        self.x = xi
        self.y = yi
        self.x0 = int(round(self.x-self.dx))
        self.x1 = int(round(self.x+self.dx))
        self.y0 = int(round(self.y-self.dy))
        self.y1 = int(round(self.y+self.dy))

        self.update_xdata()
        self.update_ydata()
        self.update_sdata()
        self.model_xData()
        self.model_yData()
        self.update_plots()

def fix_radec(fits,ra,dec):
    """Measures and returns the correct header values for spatial axes.

    Args:
        fits (astropy FITS object): The FITS file to be corrected.
        ra (float): The Right-Ascension of the known source, in degrees.
        dec (float): The Declination of the known source, in degrees.

    Returns:
        String tuple: Corrected CRVAL1, CRVAL2, CRPIX1, CRPIX2 header values.
    """

    h = fits[0].header
    plot_title = "Select the object at RA:%.4f DEC:%.4f" % (ra,dec)

    qfinder = qso.qsoFinder(fits,title=plot_title)
    x,y = qfinder.run()

    # Assign spatial center values to WCS
    if "RA" in h["CTYPE1"] and "DEC" in h["CTYPE2"]:
        crval1,crval2 = ra,dec
        crpix1,crpix2 = x,y
    elif "DEC" in h["CTYPE1"] and "RA" in h["CTYPE2"]:
        crval1,crval2 = dec,ra
        crpix1,crpix2 = y,x
    else:
        print("Bad header WCS. CTYPE1/CTYPE2 should be RA/DEC or DEC/RA")
        sys.exit()

    crpix1 +=1
    crpix2 +=1

    return crval1,crval2,crpix1,crpix2

def fix_wav(fits,instrument,skyLine=None):
    """Measures and returns the correct header values for the wavelength axis.

    Args:
        fits (astropy FITS object): The FITS file to be corrected.
        instrument (str): The instrument being used ('PCWI' or 'KCWI').
        skyLine (float): The precise wavelength of a known, fittable skyLine.

    Returns:
        String tuple: Corrected CRVAL3, CRPIX3 header values.

    """
    #Extract header info
    h = fits[0].header
    N = len(fits[0].data)
    wg0,wg1 = h["WAVGOOD0"],h["WAVGOOD1"]
    w0,dw,w0px = h["CRVAL3"],h["CD3_3"],h["CRPIX3"]
    xc = int(h["CRPIX1"])
    yc = int(h["CRPIX2"])

    #Load sky emission lines
    skyDataDir = os.path.dirname(__file__).replace('/libs','/data/sky')
    if instrument=="PCWI":
        skyLines = np.loadtxt(skyDataDir+"/palomar_lines.txt")
        fwhm_A = 5
    elif instrument=="KCWI":
        skyLines = np.loadtxt(skyDataDir+"/keck_lines.txt")
        fwhm_A = 3
    else:
        print("Instrument not recognized.")

        sys.exit()

    # Make wavelength array
    wav = np.array([w0 + dw*(j - w0px) for j in range(N)])

    #If user provided sky line and it is valid, add it at start of line list
    if skyLine!=None:
        if (wav[0]+fwhm_A)<=skyLine<=(wav[-1]-fwhm_A): skyLines = np.insert(skyLines,0,skyLine)
        else: print(("Provided skyLine (%.1fA) is outside fittable wavelength range. Using default lists."%skyLine))

    # Take normalized spatial median of cube
    sky = np.sum(fits[0].data,axis=(1,2))
    sky /=np.max(sky)


    #Run through sky lines until one is useable
    for l in skyLines:

        if wav[0]<=l<=wav[-1]:

            offset = getWavOffset(wav,sky,l,dW=fwhm_A,plot=True)

            return w0+offset, w0px

    #If we get to here, no line was found
    print("No known sky lines in range %.1f-%.1f. Wavelength solution will not be corrected.")
    return w0,w0px

def fixwcs(paramPath,icubeType,instrument,fixRADEC=True,fixWav=False,skyLine=None,RA=None, DEC=None):
    """Corrects the world-coordinate system of cubes using interactive tools.

    Args:
        paramPath (str): Path to the CWITools parameter file.
        icubeType (str): Type of icube to work with.
        instrument (str): Which CWI we're working with here (PCWI/KCWI)
        fixRADEC (bool): Fix the spatial axes (Default: True)
        fixWav (bool): Fix the wavelength axis (Default: True)
        skyLine (float): Known wavelength of a fittable sky-line.
            This parameter is required for fixing the wavelength solution.
        RA (float): RA (dd.dd) of source to use (overrides param file)
        DEC (float): DEC (dd.dd) of source to use (overrides param file)

    """

    #Load params
    params = libs.params.loadparams(paramPath)

    #Find icubes files
    ifileList = libs.params.findfiles(params,icubeType)

    #Run through all images now and perform corrections
    for i,fileName in enumerate(ifileList):

        #Get current CD matrix
        crval1,crval2,crval3 = ( fits[i][0].header["CRVAL%i"%(k+1)] for k in range(3) )
        crpix1,crpix2,crpix3 = ( fits[i][0].header["CRPIX%i"%(k+1)] for k in range(3) )

        #Get RA/DEC values if fixWAV requested
        if fixRADEC:

            radecFITS = fits.open(fileName)
            crval1,crval2,crpix1,crpix2 = libs.cubes.fixRADEC(radecFITS,RA,DEC)
            radecFITS.close()

        #Get wavelength WCS values if fixWav requested
        if fixWav:

            skyFile   = fileName.replace('icube','scube')
            skyFITS   = fitsIO.open(skyFile)
            crval3,crpix3 = libs.cubes.fixWav(skyFITS,inst[i],skyLine=skyLine)
            skyFITS.close()

        #Create lists of crval/crpix values, whether updated or not
        crvals = [ crval1, crval2, crval3 ]
        crpixs = [ crpix1, crpix2, crpix3 ]


        #Make list of relevant cubes to be corrected - scube doesn't matter as much
        cubes = ['icube','scube','ocube','vcube']

        #Load fits, modify header and save for each cube type
        for c in cubes:

            #Get filepath for this cube
            filePath = fileName.replace('icube',c)

            #Try to load, but continue upon failure
            try: f = fitsIO.open(filePath)
            except:
                print("Could not open %s. Cube will not be corrected." % filePath)
                continue

            #Fix each of the header values
            for k in range(3):

                f[0].header["CRVAL%i"%(k+1)] = crvals[k]
                f[0].header["CRPIX%i"%(k+1)] = crpixs[k]

            #Save WCS corrected cube
            wcPath = filePath.replace('.fits','.wc.fits')
            f[0].writeto(wcPath,overwrite=True)
            print("Saved %s"%wcPath)

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Use RA/DEC and Wavelength reference points to adjust WCS.')


    parser.add_argument('params',
                        type=str,
                        metavar='str',
                        help='CWITools Parameter file (used to load cube list etc.)'
    )
    parser.add_argument('icubetype',
                        type=str,
                        help='Type of cubes to work with. Must be icube.fits/icubes.fits etc.',
                        choices=['icube.fits','icubep.fits','icubed.fits','icubes.fits','icuber.fits']
    )
    parser.add_argument('inst',
                        type=str,
                        help='Which CWI instrument we are working with (KCWI or PCWI)',
                        choices=['PCWI','KCWI']
    )
    parser.add_argument('-fixWav',
                        type=str,
                        metavar='boolean',
                        help='Set to True/False to turn Wavelength correction on/off',
                        choices=["True","False"],
                        default="True"
    )
    parser.add_argument('-skyLine',
                        type=float,
                        metavar='float',
                        help='Wavelength of sky line to use for correcting WCS. (angstrom)'
    )
    parser.add_argument('-fixRADEC',
                        type=str,
                        metavar='boolean',
                        help='Set to True/False to turn RA/DEC correction on/off',
                        choices=["True","False"],
                        default="True"
    )
    parser.add_argument('-ra',
                        type=float,
                        metavar='float (deg)',
                        help='RA of source you are using for this - if not the same as parameter file target',
    )
    parser.add_argument('-dec',
                        type=float,
                        metavar='float (deg)',
                        help='DEC of source you are using for this - if not the same as parameter file target',
    )
    args = parser.parse_args()

    #Parse str boolean flags to bool types
    args.fixWav = True if args.fixWav=="True" else False
    args.fixRADEC = True if args.fixRADEC=="True" else False
    args.simpleMode = True if args.simpleMode=="True" else False

    fixwcs(args.paramFile,args.icubeType,args.instrument
        fixRADEC=args.fixRADEC,
        fixWav=args.fixWav,
        skyLine=args.skyLine,
        RA=args.ra,
        DEC=args.dec,
    )

if __name__=="__main__": main()
