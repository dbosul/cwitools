#!/usr/bin/env python
#
# Cubes Library - Methods for manipulating 3D FITS cubes (masking, aligning, coadding etc)
# 

from astropy.io import fits as fitsIO
from astropy.modeling import models,fitting
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import convolve, gaussian_filter1d,uniform_filter
from scipy.stats import mode

from shapely.geometry import Polygon

import astropy.io as apIO
import astropy.utils as utils
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pyregion
import sys

import qso

#Calculate indices in cube (bounded by ends of array)
#Return indices for a given wavelength band (w1,w2) in angstroms
def getband(_w1,_w2,_hd):
    w0,dw,p0 = _hd["CRVAL3"],_hd["CD3_3"],_hd["CRPIX3"]
    w0 -= p0*dw
    return ( int((_w1-w0)/dw), int((_w2-w0)/dw) )
	    
def get2DHeader(hdr3D):
    hdr2D = hdr3D.copy()
    for key in hdr2D.keys():
        if '3' in key:
            del hdr2D[key]
    hdr2D["NAXIS"]   = 2
    hdr2D["WCSDIM"]  = 2   
    return hdr2D   

def fixRADEC(fits,ra,dec):

    #
    # RA/DEC Correction using source (usually QSO)
    #
    
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
    
    return crval1,crval2,crpix1,crpix2

def fixWav(fits,instrument):

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
        print "Instrument not recognized."
        sys.exit()
                    
    # Make wavelength array
    wav = np.array([w0 + dw*(j - w0px) for j in range(N)])
    
    #Crop to good wavelengths (only if there is a useable line in that range)
    usewav = np.ones_like(wav,dtype='bool')
    if any([ wg0<=sl<=wg1 for sl in skyLines]):
        usewav[wav<wg0] = 0
        usewav[wav>wg1] = 0             
    wav = wav[usewav]

    
    # Take normalized spatial median of cube
    sky = np.sum(fits[0].data,axis=(1,2))
    sky /=np.max(sky)
    sky = sky[usewav]
    
    #Run through sky lines until one is useable
    for l in skyLines:
                
        if wav[0]<=l<=wav[-1]:
            
            offset = getWavOffset(wav,sky,l,dW=fwhm_A,plot=True)
            
            return w0+offset, w0px
                     
    #If we get to here, no line was found
    print("No known sky lines in range %.1f-%.1f. Wavelength solution will not be corrected.")
    return w0,w0px
    
def getWavOffset(W,S,L,dW=3,iters=2,plot=False):
  
    #Get smooth wavelength array
    Ws = np.linspace(W[0],W[-1],10*len(W))

    #Median subtract the sky spectrum
    S-=np.median(S)
    
    #Get fitter
    lineFitter = fitting.SimplexLSQFitter()
    
    #Run iterative fitting loop
    wC = L
    for it in range(iters):
        
        #Identify good fitting wavelengths 
        fitwav = np.ones_like(W,dtype='bool')
        fitwav[ W < wC-dW ] = 0
        fitwav[ W > wC+dW ] = 0                    

        #Extract line from spectrum
        linespec = S[fitwav] - np.median(S)
        linespec = gaussian_filter1d(linespec,1.0)
        
        linewav = W[fitwav]
        
        #Fit Gaussian to line
        A0 = np.max(linespec)
        l0 = linewav[np.nanargmax(linespec)]
        
        modelguess = models.Gaussian1D(amplitude=A0,mean=l0,stddev=1.0)
        modelguess.mean.min = modelguess.mean.value - dW
        modelguess.mean.max = modelguess.mean.value + dW 
        modelguess.stddev.min = 0.1
        modelguess.stddev.max = 2.0       
        modelguess.amplitude.min = 0                                    
        modelfit = lineFitter(modelguess,linewav,linespec)

        wC = modelfit.mean.value

    if plot:
    
        fit = modelfit(Ws)
                
        grid_width = 10
        grid_height = 1
        gs = gridspec.GridSpec(grid_height,grid_width)   
       
        fig = plt.figure(figsize=(16,4))    
        axes = [ fig.add_subplot(gs[ :, 0:7 ]),  fig.add_subplot(gs[ :, 7: ]) ]
        
        axes[0].plot(W,S,'k-',label="Sky Spectrum")
        axes[0].set_xlabel("Wavelength (A)")
        axes[0].set_ylabel("Normalized Flux/Counts")
        axes[0].plot([L,L],[0,1],'r-',label="Line to fit")
        axes[0].set_title("Full sky spectrum")
        axes[0].legend()
        
        axes[1].set_xlabel("Wavelength (A)")
        axes[1].plot(W,S,'k-',label="Data")
        axes[1].plot(Ws,fit,'b-',label="Fit")
        axes[1].plot([L,L],[0,1],'r-',label="True")
        axes[1].get_xaxis().get_major_formatter().set_useOffset(False)
        axes[1].set_title("Line Fit Zoom-In")
        axes[1].set_xlim( [L-3*dW,L+3*dW] )
        axes[1].set_ylim( [0,np.max(fit)] )
        axes[1].legend()
        
        fig.tight_layout()
        fig.show()
        plt.waitforbuttonpress()
        plt.close()
    
    return L-modelfit.mean.value
    

def cropFITS(fits,params):
    
    for i,f in enumerate(fits):
        
        h = f[0].header

        #Parse crop values from parameters      
        xx = tuple( int(x) for x in params["XCROP"][i].split(':'))
        yy = tuple( int(y) for y in params["YCROP"][i].split(':'))
        ww = tuple( int(w) for w in params["WCROP"][i].split(':'))
        wA,wB = getband(ww[0],ww[1],h)
        
        #Crop cube        
        cropData = f[0].data[wA:wB,yy[0]:yy[1],xx[0]:xx[1]].copy()
        
        #Change RA/DEC/WAV reference pixels
        cropHeader = h.copy()
        cropHeader["CRPIX1"] -= xx[0]
        cropHeader["CRPIX2"] -= yy[0]
        cropHeader["CRPIX3"] -= wA
        
        #Update FITS
        f[0].header = cropHeader
        f[0].data = cropData
    
    #Return list of fits objects
    return fits
    
def coadd(fileList,params,expThreshold = 0.1,pxlThreshold=0.9,propVar=True):

    #
    # STAGE 0: PREPARATION
    # 

    # Open custom FITS-3D objects
    fitsList = [fitsIO.open(f) for f in fileList] 
    
    # If option is set to propagate variance alongside coadd
    if propVar:
    
        # Check all input is "icube" type (Variance cubes correspond to these)
        propVar  = np.all([ "icube" in fileName for fileName in fileList])
        
        # Create parallel list of FITS images
        try:varList = [ fitsIO.open(f.replace("icube","vcube")) for f in fileList ]
        except:
            print("Could not load variance input cubes from data directory. Error will not be propagated throughout coadd.")
            propVar=False
        
    # Make all data products in 10^16 Flam
    for i,f in enumerate(fitsList):
    
        # Only need to scale up Palomar data
        if params['INST'][i]=='PCWI' and f[0].header["BUNIT"]=='FLAM':
            
            #Scale data up to same units as KCWI data (10^16)*F_lambda
            f[0].data *= 1e16
            f[0].header["BUNIT"] = 'FLAM16'  
            
            # Square variance to account for this scaling
            if propVar: varList[i][0].data*=1e32 
                  
    # Extract basic header info
    hdrList    = [ f[0].header for f in fitsList ]
    wcsList    = [ WCS(h) for h in hdrList ]
    pxScales   = np.array([ proj_plane_pixel_scales(wcs) for wcs in wcsList ])
    raQ,decQ = params["RA"],params["DEC"]

    # Get 2D headers, WCS and on-sky footprints
    h2DList    = [ get2DHeader(h) for h in hdrList]
    w2DList    = [ WCS(h) for h in h2DList ]
    footPrints = np.array([ w.calc_footprint() for w in w2DList ])

    # Exposure times
    expKeys  =  [ "TELAPSE" if inst=="KCWI" else "EXPTIME" for inst in params["INST"] ]   
    expTimes =  [ h[expKeys[i]] for i,h in enumerate(hdrList) ]

    # Extract into useful data structures
    xScales,yScales,wScales = ( pxScales[:,i] for i in range(3) )
    
    # Determine coadd scales
    coadd_xyScale = np.min(np.abs(pxScales[:,:2]))
    coadd_wScale  = np.min(np.abs(pxScales[:,2]))

    #
    # STAGE 1: WAVELENGTH ALIGNMENT
    # 
    
    # Check that the scale (Ang/px) of each input image is the same
    if len(set(wScales))!=1:
    
        print("ERROR: Wavelength axes must be equal in scale for current version of code.")
        print("Continue stacking without wavelength alignment? (y/n) >")
        answer = raw_input("")
        if not( answer=="y" or answer=="Y" or answer=="yes" ): sys.exit()
        else: print("Proceeding with stacking without any wavelength axis shifts.")
        
    else:
       
        # Get common wavelength scale
        cd33 = hdrList[0]["CD3_3"]
          
        # Get lower and upper wavelengths for each cube
        wav0s = [ h["CRVAL3"] - (h["CRPIX3"]-1)*cd33 for h in hdrList ]
        wav1s = [ wav0s[i] + h["NAXIS3"]*cd33 for i,h in enumerate(hdrList) ]
        
        # Get new wavelength axis
        wNew = np.arange(min(wav0s)-cd33, max(wav1s)+cd33,cd33)

        print("Aligning wavelength axes."),
                
        # Adjust each cube to be on new wavelenght axis
        for i,f in enumerate(fitsList):

            print('.'),
            
            # Pad the end of the cube with zeros to reach same length as wNew
            f[0].data = np.pad( f[0].data, ( (0, len(wNew)-f[0].header["NAXIS3"]), (0,0) , (0,0) ) , mode='constant' )
           
            # Get the wavelength offset between this cube and wNew
            dw = (wav0s[i] - wNew[0])/cd33
            
            # Split the wavelength difference into an integer and sub-pixel shift
            intShift = int(dw)
            spxShift = dw - intShift
        
            # Perform integer shift with np.roll
            f[0].data = np.roll(f[0].data,intShift,axis=0)
            
            # Create convolution matrix for subpixel shift (in effect; linear interpolation)
            K = np.array([ spxShift, 1-spxShift ])

            # Shift data along axis by convolving with K      
            f[0].data = np.apply_along_axis(lambda m: np.convolve(m, K, mode='same'), axis=0, arr=f[0].data)
    
            f[0].header["NAXIS3"] = len(wNew)
            f[0].header["CRVAL3"] = wNew[0]
            f[0].header["CRPIX3"] = 1

            #Update the variance for the integer shift and kernel convolution
            if propVar:
                vFITS = varList[i]
                vdata = np.pad( vFITS[0].data, ( (0, len(wNew) - vFITS[0].header["NAXIS3"]), (0,0) , (0,0) ) , mode='constant' )
                vdata = np.roll(vdata,intShift,axis=0)
                vdata = np.apply_along_axis(lambda m: np.convolve(m, K**2, mode='same'), axis=0, arr=vdata)
                varList[i][0].data = vdata
        
        print("")

         
    #   
    # Stage 2 - SPATIAL ALIGNMENT
    #
    
    # Get center and bounds of coadd canvas in RA/DEC space
    ra0,ra1   = np.max(footPrints[:,:,0]),np.min(footPrints[:,:,0])
    dec0,dec1 = np.min(footPrints[:,:,1]),np.max(footPrints[:,:,1])

    #Extend bounds to account for the fact the pixel 'edges' lie outside the footprint
    maxScale = np.max(pxScales[:,1:])
    ra0  += maxScale
    dec0 -= maxScale
    ra1  -= maxScale
    dec1 += maxScale
    
    #    
    # Create header structure for coadd cube
    #
    
    # Center coordinates and pixels
    coaddHdr = hdrList[-1].copy()
    coaddHdr["CRVAL1"] = ra0
    coaddHdr["CRVAL2"] = dec0
    coaddHdr["CRVAL3"] = wNew[0]        
    coaddHdr["CRPIX1"] = 1
    coaddHdr["CRPIX2"] = 1
    coaddHdr["CRPIX3"] = 1
    
    yxRatio = np.max(np.abs(pxScales[:,:2]))/np.min(np.abs(pxScales[:,:2]))

    # Set position angle to zero (TO DO: Include 'mode' version of coadd code)
    coaddHdr["CD1_1"]  = -coadd_xyScale
    coaddHdr["CD2_2"]  = coadd_xyScale
    coaddHdr["CD1_2"]  = 0
    coaddHdr["CD2_1"]  = 0    
    coaddHdr["ROTPA"]  = 0
    
    # Create WCS object with this orientation & reference RA/DEC
    coaddHdr2D = get2DHeader(coaddHdr)
    coaddWCS   = WCS(coaddHdr2D)
    coaddFP    = coaddWCS.calc_footprint()
    
    # Get X,Y bounds for canvas   
    x1,y1 = coaddWCS.all_world2pix(ra1,dec1,0) 
    
    # Update Canvas size and re-generate header/WCS/footprint
    coaddHdr["NAXIS1"] = int(x1+1)
    coaddHdr["NAXIS2"] = int(y1+1)
    coaddHdr["NAXIS3"] = len(wNew)
    coaddHdr2D = get2DHeader(coaddHdr)
    coaddWCS   = WCS(coaddHdr2D)
    coaddFP    = coaddWCS.calc_footprint()

    # Create data structures to store coadded cube and corresponding exposure time mask
    coaddData = np.zeros((len(wNew),coaddHdr["NAXIS2"],coaddHdr["NAXIS1"]))
    coaddExp  = np.zeros_like(coaddData)
    
    # Create equivalent structure to propagate error if needed
    if propVar: varData = np.zeros_like(coaddData)
        
    W,Y,X = coaddData.shape
       
    makePlots=0
    if makePlots:
        fig1,ax = plt.subplots(1,1)
        for fp in footPrints:
            ax.plot( fp[0:2,0],fp[0:2,1],'k-')
            ax.plot( fp[1:3,0],fp[1:3,1],'k-')
            ax.plot( fp[2:4,0],fp[2:4,1],'k-')
            ax.plot( [ fp[3,0], fp[0,0] ] , [ fp[3,1], fp[0,1] ],'k-')
        for fp in [coaddFP]:
            ax.plot( fp[0:2,0],fp[0:2,1],'r-')
            ax.plot( fp[1:3,0],fp[1:3,1],'r-')
            ax.plot( fp[2:4,0],fp[2:4,1],'r-')
            ax.plot( [ fp[3,0], fp[0,0] ] , [ fp[3,1], fp[0,1] ],'r-')
        ax.plot(raQ,decQ,'ro',alpha=0.8)        
        fig1.show()
        plt.waitforbuttonpress()
        plt.close()
    
        plt.ion()

        grid_width  = 2
        grid_height = 2
        gs = gridspec.GridSpec(grid_height,grid_width)   
       
        fig2 = plt.figure(figsize=(12,12))    
        inAx  = fig2.add_subplot(gs[ :1, : ])
        skyAx = fig2.add_subplot(gs[ 1:, :1 ]) 
        imgAx = fig2.add_subplot(gs[ 1:, 1: ]) 

                               
    # Run through each input frame
    for i,f in enumerate(fitsList):
        
        #Get shape of current cube
        w,y,x = f[0].data.shape
        
        # Create intermediate frame to build up coadd contributions pixel-by-pixel
        buildFrame = np.zeros_like(coaddData)
        
        # Create intermediate frame for variance data
        if propVar: vbuildFrame = np.zeros_like(coaddData)
        
        # Fract frame stores a coverage fraction for each coadd pixel
        fractFrame = np.zeros_like(coaddData)

        # Get wavelength coverage of this FITS
        wavIndices    = np.ones(f[0].data.shape[0],dtype=bool)
        wavIndices[wNew<wav0s[i]] = 0
        wavIndices[wNew>wav1s[i]] = 0

        # Convert to a flux-like unit if the input data is in counts
        if "electrons" in f[0].header["BUNIT"]:
        
            # Scale data to be in counts per unit time
            f[0].data /= expTimes[i] 
            f[0].header["BUNIT"] = "electrons/sec" 
            
            # Account for this scaling in variance
            if propVar: varList[i][0] /= expTimes[i]**2
            
        print("Mapping %s to coadd frame (%i/%i)"%(params["IMG_ID"][i],i+1,len(fitsList))),
        
        if makePlots:
            qXin = params["SRC_X"][i]-int(params["XCROP"][i].split(':')[0])
            qYin = params["SRC_Y"][i]-int(params["YCROP"][i].split(':')[0])
            qRA,qDEC = w2DList[i].all_pix2world(qXin,qYin,0)
            qXco,qYco = coaddWCS.all_world2pix(qRA,qDEC,0)

            inAx.clear()
            skyAx.clear()
            imgAx.clear()
            inAx.set_title("Input Frame Coordinates")
            skyAx.set_title("Sky Coordinates")
            imgAx.set_title("Coadd Coordinates")
            imgAx.set_xlabel("X")
            imgAx.set_ylabel("Y")
            skyAx.set_xlabel("RA (hh.hh)")
            skyAx.set_ylabel("DEC (dd.dd)")
            xU,yU = x,y
            inAx.plot( [0,xU], [0,0], 'k-')
            inAx.plot( [xU,xU], [0,yU], 'k-')
            inAx.plot( [xU,0], [yU,yU], 'k-')
            inAx.plot( [0,0], [yU,0], 'k-')
            inAx.set_xlim( [-5,xU+5] )
            inAx.set_ylim( [-5,yU+5] )
            inAx.plot(qXin,qYin,'ro')
            inAx.set_xlabel("X")
            inAx.set_ylabel("Y")                   
            xU,yU = X,Y
            imgAx.plot( [0,xU], [0,0], 'r-')
            imgAx.plot( [xU,xU], [0,yU], 'r-')
            imgAx.plot( [xU,0], [yU,yU], 'r-')
            imgAx.plot( [0,0], [yU,0], 'r-')
            imgAx.set_xlim( [-0.5,xU+1] )
            imgAx.set_ylim( [-0.5,yU+1] )
            for fp in footPrints[i:i+1]:              
                skyAx.plot( -fp[0:2,0],fp[0:2,1],'k-')
                skyAx.plot( -fp[1:3,0],fp[1:3,1],'k-')
                skyAx.plot( -fp[2:4,0],fp[2:4,1],'k-')
                skyAx.plot( [ -fp[3,0], -fp[0,0] ] , [ fp[3,1], fp[0,1] ],'k-')
            for fp in [coaddFP]:             
                skyAx.plot( -fp[0:2,0],fp[0:2,1],'r-')
                skyAx.plot( -fp[1:3,0],fp[1:3,1],'r-')
                skyAx.plot( -fp[2:4,0],fp[2:4,1],'r-')
                skyAx.plot( [ -fp[3,0], -fp[0,0] ] , [ fp[3,1], fp[0,1] ],'r-')      
            skyAx.plot(-qRA,qDEC,'ro')
            skyAx.plot(-raQ,decQ,'kx')
            #skyAx.set_xlim([ra0+0.001,ra1-0.001])
            skyAx.set_ylim([dec0-0.001,dec1+0.001])
            
            imgAx.plot(qXco,qYco,'ro')
            
        # Loop through spatial pixels in this input frame
        for yj in range(y):
        
            print("."),
            sys.stdout.flush()
            
            for xk in range(x):
               
                # Define BL, TL, TR, BR corners of pixel as coordinates
                inPixVertices =  np.array([ [xk-0.5,yj-0.5], [xk-0.5,yj+0.5], [xk+0.5,yj+0.5], [xk+0.5,yj-0.5] ]) 
    
                # Convert these vertices to RA/DEC positions
                inPixRADEC = w2DList[i].all_pix2world(inPixVertices,0)

                # Convert the RA/DEC vertex values into coadd frame coordinates
                inPixCoadd = coaddWCS.all_world2pix(inPixRADEC,0)

                #Create polygon object for projection of this input pixel onto coadd grid
                pixIN = Polygon( inPixCoadd )                 

                #Get bounding pixels on coadd grid  
                xP0,yP0,xP1,yP1 = (int(x) for x in list(pixIN.bounds))  


                if makePlots:
                    inAx.plot( inPixVertices[:,0], inPixVertices[:,1],'kx')
                    skyAx.plot(-inPixRADEC[:,0],inPixRADEC[:,1],'kx')
                    imgAx.plot(inPixCoadd[:,0],inPixCoadd[:,1],'kx')                
                
                #Get bounds of pixel in coadd image
                xP0,yP0,xP1,yP1 = (int(round(x)) for x in list(pixIN.exterior.bounds))
                
                # Upper bounds need to be increased to include full pixel
                xP1+=1
                yP1+=1

                # Run through pixels on coadd grid and add input data
                for xC in range(xP0,xP1):
                    for yC in range(yP0,yP1):

                        # Define BL, TL, TR, BR corners of pixel as coordinates
                        cPixVertices =  np.array( [ [xC-0.5,yC-0.5], [xC-0.5,yC+0.5], [xC+0.5,yC+0.5], [xC+0.5,yC-0.5] ]   )       
                        
                        # Create Polygon object and store in array  
                        pixCA = Polygon( cPixVertices )
                        
                        # Calculation fractional overlap between input/coadd pixels
                        overlap = pixIN.intersection(pixCA).area/pixIN.area

                        # Add fraction to fraction frame
                        fractFrame[wavIndices,yC,xC] += overlap

                        # Add data to build frame
                        buildFrame[:,yC,xC] += overlap*f[0].data[:,yj,xk]

                        # Update variance build frame accordingly
                        if propVar: vbuildFrame[:,yC,xC]  += (overlap**2)*varList[i][0].data[:,yj,xk]
                            
                            
               
        # Add weights to mask (denominator of weighted mean)     
        if makePlots:
            fig2.canvas.draw()
            plt.waitforbuttonpress()        

        # Get mask of non-zero voxels in build frame
        M = fractFrame<1
        
        # Get the ratio of coadd pixel size to input pixel size
        f0 = pxlThreshold*(coadd_xyScale**2)/(xScales[i]*yScales[i])

        # Trim edge pixels (and also change all 0s to 1s to avoid NaNs)
        ff = fractFrame.flatten()
        bb = buildFrame.flatten()
        vv = vbuildFrame.flatten()
        bb[ff<f0] = 0
        vv[ff<f0] = 0
        ff[ff<f0] = 1
        fractFrame  = np.reshape(ff,coaddData.shape)
        buildFrame  = np.reshape(bb,coaddData.shape)
        vbuildFrame = np.reshape(vv,coaddData.shape)
        
        # Divide by the sum of overlap fractions
        if "FLAM" in f[0].header["BUNIT"]: 
            buildFrame/=fractFrame           
            if propVar: vbuildFrame/=fractFrame**2
                    
        
        # Create 3D mask of observations
        M = np.reshape( ff<1, coaddData.shape)

        # Add weight*data to coadd (numerator of weighted mean with exptime as weight)
        coaddData += expTimes[i]*buildFrame
        
        # Propagate error via variance cubes again
        if propVar: varData += (expTimes[i]**2)*vbuildFrame
        
        #Add to exposure mask
        coaddExp += expTimes[i]*M

        print("")
        
        
    if makePlots: plt.close() 
    
    # Create 1D exposure time profiles
    expSpec = np.mean(coaddExp,axis=(1,2))
    expXMap = np.mean(coaddExp,axis=(0,1))
    expYMap = np.mean(coaddExp,axis=(0,2))

    # Normalize the profiles
    expSpec/=np.max(expSpec)
    expXMap/=np.max(expXMap)
    expYMap/=np.max(expYMap)
        
    # Convert 0s to 1s in exposure time cube
    ee = coaddExp.flatten()
    ee[ee==0] = 1
    coaddExp = np.reshape( ee, coaddData.shape )
    
    # Divide by sum of weights (or square of sum)
    coaddData /= coaddExp
    if propVar: varData /= coaddExp**2
    
    # Create FITS object
    coaddHDU = apIO.fits.PrimaryHDU(coaddData)
    coaddFITS = apIO.fits.HDUList([coaddHDU])
    coaddFITS[0].header = coaddHdr

    #Exposure time threshold, relative to maximum exposure time, below which to crop.
    useW = expSpec>expThreshold
    useX = expXMap>expThreshold
    useY = expYMap>expThreshold

    #Trim the data
    coaddFITS[0].data = coaddFITS[0].data[useW]
    coaddFITS[0].data = coaddFITS[0].data[:,useY]
    coaddFITS[0].data = coaddFITS[0].data[:,:,useX]

    #Get 'bottom/left/blue corner of cropped data
    W0 = np.argmax(useW)
    X0 = np.argmax(useX)
    Y0 = np.argmax(useY)
        
    #Update the WCS to account for trimmed pixels
    coaddFITS[0].header["CRPIX3"] -= W0
    coaddFITS[0].header["CRPIX2"] -= Y0
    coaddFITS[0].header["CRPIX1"] -= X0
    
    #Create FITS for variance data if we are propagating that
    if propVar:
        varHDU = apIO.fits.PrimaryHDU(varData)
        varFITS = apIO.fits.HDUList([varHDU])
        varFITS[0].header = coaddHdr    
        return coaddFITS,varFITS
        
    # Crop new fit object to trim zero edges     
    else: return coaddFITS,None
        

def get_regMask(fits,regfile,scaling=2):

    #EXTRACT/CREATE USEFUL VARS############
    data3D = fits[0].data
    head3D = fits[0].header

    W,Y,X = data3D.shape #Dimensions
    mask = np.zeros((Y,X),dtype=int) #Mask to be filled in
    x,y = np.arange(X),np.arange(Y) #Create X/Y image coordinate domains
    xx, yy = np.meshgrid(x, y) #Create meshgrid of X, Y
    ww = np.array([ head3D["CRVAL3"] + head3D["CD3_3"]*(i - head3D["CRPIX3"]) for i in range(W)])
    
    yPS = np.sqrt( np.cos(head3D["CRVAL2"]*np.pi/180)*head3D["CD1_2"]**2 + head3D["CD2_2"]**2 ) #X & Y plate scales (deg/px)
    xPS = np.sqrt( np.cos(head3D["CRVAL2"]*np.pi/180)*head3D["CD1_1"]**2 + head3D["CD2_1"]**2 )
        
    fit = fitting.SimplexLSQFitter() #Get astropy fitter class
    Lfit = fitting.LinearLSQFitter() 
    
    usewav = np.ones_like(ww,dtype=bool)
    usewav[ww<head3D["WAVGOOD0"]] = 0
    usewav[ww>head3D["WAVGOOD1"]] = 0
    
    data2D = np.sum(data3D[usewav],axis=0)
    med = np.median(data2D)

    #BUILD MASK############################
    if regfile[0].coord_format=='image':

        rr = np.sqrt( (xx-x0)**2 + (yy-y0)**2 )
        mask[rr<=R] = i+1          
                    
    elif regfile[0].coord_format=='fk5':  
    
        #AIC = 2k + n Log(RSS/n) [ - (2k**2 +2k )/(n-k-1) ]
        def AICc(dat,mod,k):
            RSS = np.sum( (dat-mod)**2 )
            n = np.size(dat)
            return 2*k + n*np.log(RSS/n) #+ (2*k**2 + 2*k)/(n-k-1)
            
        head2D = get2DHeader(head3D)
        wcs = WCS(head2D)    
        ra, dec = wcs.wcs_pix2world(xx, yy, 0) #Get meshes of RA/DEC
        
        for i,reg in enumerate(regfile):    
        
            ra0,dec0,R = reg.coord_list #Extract location and default radius    
            rr = np.sqrt( (np.cos(dec*np.pi/180)*(ra-ra0))**2 + (dec-dec0)**2 ) #Create meshgrid of distance to source 
            
            if np.min(rr) > R:
                continue #Skip any sources more than one default radius outside the FOV
            
            else:
                
                yc,xc = np.where( rr == np.min(rr) ) #Take input position tuple 
                xc,yc = xc[0],yc[0]
                
                rx = 2*int(round(R/xPS)) #Convert angular radius to distance in pixels
                ry = 2*int(round(R/yPS))

                x0,x1 = max(0,xc-rx),min(X,xc+rx+1) #Get bounding box for PSF fit
                y0,y1 = max(0,yc-ry),min(Y,yc+ry+1)

                img = np.mean(data3D[usewav,y0:y1,x0:x1],axis=0) #Not strictly a white-light image
                img -= np.median(img) #Correct in case of bad sky subtraction
                
                xdomain,xdata = range(x1-x0), np.mean(img,axis=0) #Get X and Y domains/data
                ydomain,ydata = range(y1-y0), np.mean(img,axis=1)
                
                moffat_bounds = {'amplitude':(0,float("inf")) }
                xMoffInit = models.Moffat1D(max(xdata),x_0=xc-x0,bounds=moffat_bounds) #Initial guess Moffat profiles
                yMoffInit = models.Moffat1D(max(ydata),x_0=yc-y0,bounds=moffat_bounds)
                xLineInit = models.Linear1D(slope=0,intercept=np.mean(xdata))
                yLineInit = models.Linear1D(slope=0,intercept=np.mean(ydata))
                
                xMoffFit = fit(xMoffInit,xdomain,xdata) #Fit Moffat1Ds to each axis
                yMoffFit = fit(yMoffInit,ydomain,ydata)
                xLineFit = Lfit(xLineInit,xdomain,xdata) #Fit Linear1Ds to each axis
                yLineFit = Lfit(yLineInit,ydomain,ydata)
                
                kMoff = len(xMoffFit.parameters) #Get number of parameters in each model
                kLine = len(xLineFit.parameters)
                
                xMoffAICc = AICc(xdata,xMoffFit(xdomain),kMoff) #Get Akaike Information Criterion for each
                xLineAICc = AICc(xdata,xLineFit(xdomain),kLine)
                yMoffAICc = AICc(ydata,yMoffFit(ydomain),kMoff)
                yLineAICc = AICc(ydata,yLineFit(ydomain),kLine)
                
                xIsMoff = xMoffAICc < xLineAICc # Determine if Moffat is a better fit than a simple line
                yIsMoff = yMoffAICc < yLineAICc
                
                if xIsMoff and yIsMoff: #If source has detectable moffat profile (i.e. bright source) expand mask

                    xfwhm = xMoffFit.gamma.value*2*np.sqrt(2**(1/xMoffFit.alpha.value) - 1) #Get FWHMs
                    yfwhm = yMoffFit.gamma.value*2*np.sqrt(2**(1/yMoffFit.alpha.value) - 1)

                    R = scaling*max(xfwhm*xPS,yfwhm*yPS)
                
                mask[rr <= R] = i+1

    return mask


def get_skyMask(fits,inst="PCWI"):
    
    
    skyDataDir = os.path.dirname(__file__).replace('/libs','/data/sky')
    
    if inst=="PCWI":
        skyLines = np.loadtxt(skyDataDir+"/palomar_lines.txt")
        fwhm_A = 5
    elif inst=="KCWI":
        skyLines = np.loadtxt(skyDataDir+"/keck_lines.txt")
        fwhm_A = 3
    else:
        print "Instrument not recognized."
        sys.exit()
        
    h = fits[0].header
    w0,wpix0,dw,Nw = h["CRVAL3"],h["CRPIX3"],h["CD3_3"],fits[0].data.shape[0]
    W = np.array([ w0 + (i-wpix0)*dw for i in range(Nw) ] )
    
    fwhm_px = fwhm_A/dw
    hwhm_px = fwhm_px/2
    
    skyMask = np.zeros(Nw)

    for sL in skyLines:
        
        a = int(round(((sL-hwhm_px)-w0)/dw + wpix0))
        b = int(round(((sL+hwhm_px)-w0)/dw + wpix0))

        if a>0 and b<Nw-1:  skyMask[a:b] = 1
    
    return skyMask
    
    
    
       
def apply_mask(cube,mask,mode='zero',inst='PCWI'):


    if mode=='zero':
        
        #Just replace with zeros
        for wi in range(cube.shape[0]): cube[wi][mask>0] = 0
    
    elif mode=='cubemedian':
    
        #Replace with cube-wide median
        cubemed = np.median(cube)
        for wi in range(cube.shape[0]): cube[wi][mask>0] = cubemed
    
    elif mode=='xmedian':
        
        if inst=='PCWI':
            
            for yi in range(cube.shape[1]):
            
                #Get 1D median wavelength profile of slice
                slicemedprof = np.median(cube[:,yi,:],axis=1)     
                
                #Apply to spaxels that are masked  
                for xi in range(cube.shape[2]):
                    
                    if mask[yi,xi] > 0: cube[:,yi,xi] = slicemedprof
                    
        elif inst=='KCWI':
        
            for xi in range(cube.shape[2]):    
            
                #Get 1D median wavelength profile of slice
                slicemedprof = np.median(cube[:,:,xi],axis=1)     
                
                #Apply to spaxels that are masked  
                for yi in range(cube.shape[1]):
                    
                    if mask[yi,xi] > 0: cube[:,yi,xi] = slicemedprof
                                    
                
            
    else: print "Apply_Mask: Mode not recognized."
    
    return cube
    
           
