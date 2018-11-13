from astropy.io import fits as fitsIO
from astropy.modeling import models,fitting
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units
from scipy.ndimage.measurements import center_of_mass as CoM

import matplotlib.pyplot as plt
import numpy as np
import pyregion
import sys
import libs

#Timer start
tStart = time.time()

#Define some constants
c   = 3e5       # Speed of light in km/s
lyA = 1215.6    # Wavelength of LyA in Angstrom
v   = 2000      # Velocity window for line emission in km/s

#PSF Fitting parameters
rSub = 2        # Radius of subtraction area in arcseconds
rFit = 1        # Radius to use for fitting in arcseconds

#Take minimum input 
paramPath = sys.argv[1]
cubeType  = sys.argv[2]

#Take any additional input params, if provided
settings = {"level":"coadd","line":"lyA"}
if len(sys.argv)>3:
    for item in sys.argv[3:]:      
        key,val = item.split('=')
        if settings.has_key(key): settings[key]=val
        else:
            print "Input argument not recognized: %s" % key
            sys.exit()
            
#Load parameters
params = libs.params.loadparams(paramPath)

#Check if parameters are complete
libs.params.verify(params)

#Get filenames     
if settings["level"]=="coadd":   files = [ '%s%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubeType) ]
elif settings["level"]=="input": files = libs.io.findfiles(params,cubeType)
else:
    print("Setting 'level' must be either 'coadd' or 'input'. Exiting.")
    sys.exit()

#Calculate wavelength range
if settings["line"]=="lyA":
    wC = (1+params["ZLA"])*lyA
else:
    print("Setting - line:%s - not recognized."%settings["line"])
    sys.exit()
        
#Open regionfile
regpath = params["REG_FILE"]
if regpath=="None": 
    print "\nERROR: Region file indicating positions of continuum sources is required for psfSub.py.\nPlease add to your parameter file and re-run."
    sys.exit()
else: regfile = pyregion.open(regpath)

#Pick fitter to use for scaling PSF
fitter = fitting.LevMarLSQFitter()

#Run through files to be cropped
for fileName in files:
    
    #Open FITS and extract info
    f = fitsIO.open(fileName)
    h = f[0].header

    #Try to open corresponding variance cube
    #try:
    V = fitsIO.open(fileName.replace('icube','vcube'))
    #except:
    #    print("Error opening variance cube for this target. Variance will not be updated.")
    #    V = None
        
    #Get 
    w,y,x = f[0].data.shape
    WW = np.array([ h["CRVAL3"] + h["CD3_3"]*(i - h["CRPIX3"]) for i in range(w)])
   
    #Get continuum wavelength mask
    contWavs = np.ones(w,dtype=bool)
    w1,w2 =  wC*(1-v/c), wC*(1+v/c) 
    a,b = libs.cubes.getband(w1,w2,h)
    a,b = max(0,a), min(w-1,b)
    contWavs[a:b] = 0
    skyMask = libs.cubes.get_skyMask(f)
    contWavs[skyMask==1] = 0
    
    #Create median-subtracted continuum-wavelength image (integrating over N*dw angstrom)
    contImage = np.mean(f[0].data[contWavs],axis=0)*h["CD3_3"]
    contImage -= np.median(contImage)
    
    #Save FITS of cropped data   
    contFITS  = fitsIO.HDUList([fitsIO.PrimaryHDU(contImage)])
    contFITS[0].header =  libs.cubes.get2DHeader(h)
    contFITS.writeto(fileName.replace('.fits','.ct.fits'),overwrite=True)    

    #Create mesh grids and boolean masks for subtracting/fitting
    wcs     = WCS(libs.cubes.get2DHeader(h))
    X,Y     = np.arange(x),np.arange(y) #Create X/Y image coordinate domains
    XX,YY   = np.meshgrid(X, Y) 
    RA, DEC = wcs.wcs_pix2world(XX, YY, 0) #Get meshes of RA/DEC
    
    #Run through regions in source region file
    for i,reg in enumerate(regfile):

        ra0,dec0,R = reg.coord_list #Extract location and default radius    
        RR = np.sqrt( (np.cos(DEC*np.pi/180)*(RA-ra0))**2 + (DEC-dec0)**2 ) #Create meshgrid of distance to source 
        RR *= 3600
        
        #Only continue if source is in FOV
        if np.min(RR) <= rFit:
          
            #Get boolean masks for subtraction/fitting
            subIm = RR<rSub
            fitIm = RR<rFit
            
            #Run through wavelength layers in this cube
            for wi in range(w):
            
                #Extract 2D layer at this wavelength
                layer2D  = f[0].data[wi].copy()
                layer2D -= np.median(layer2D)
                
                #Take a guess at the initial scaling
                scaleGuess = models.Scale(factor=np.max(layer2D)/np.max(contImage))
                scaleGuess.factor.min = 0
                scaleGuess.factor.max = np.max(layer2D)/np.max(contImage)
                
                #Crop down to fitting wavelengths
                layer2DFit = layer2D[fitIm]
                contImageFit  = contImage[fitIm]
                
                #Fit scale model to data
                scaleFit = fitter(scaleGuess,contImageFit,layer2DFit)
                
                #Use fitted scale value to make model
                model = scaleFit.factor.value*contImage
                model[subIm==0] = 0
                      
                #Subtract model from data
                f[0].data[wi] -= model
    
                #Update variance if possible
                if V!=None:
                    V[0].data[wi] += scaleFit.factor.value*model
                    
    #Median subtract data
    f[0].data -= np.median(f[0].data[contWavs],axis=0)
                
    #Write out PSF-subtracted fits
    outFile = fileName.replace('.fits','.ps.fits')
    f.writeto(outFile,overwrite=True)
    print("Saved %s" % outFile)
    
    if V!=None:
        VFileOut=fileName.replace('icube','vcube').replace('.fits','.ps.fits')
        V.writeto(VFileOut,overwrite=True)
        print("Saved %s" % VFileOut)
        
#Timer end
tFinish = time.time()
print("Elapsed time: %.2f seconds" % (tFinish-tStart))        
