"""CWITools library for 3D data-cube manipulation.

This module contains a number of useful functions for manipulating
three-dimensional FITS data cubes."""

from . import qso
from . import params

from astropy.io import fits as fitsIO
from astropy.modeling import models,fitting
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.path import Path #TEST
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import convolve, gaussian_filter1d,uniform_filter
from scipy.stats import mode
from shapely.geometry import box, Polygon

import astropy.io as apIO
import astropy.utils as utils
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pyregion
import sys

def get_band(w1,w2,header):
    """Returns wavelength indices for two given wavelengths in Angstrom

    Args:
        w1 (float): Lower wavelength, in Angstrom.
        w2 (float): Upper wavelength, in Angstrom.
        header (astropy FITS header): FITS header for this data cube.

    Returns:
        int tuple: The lower and upper wavelength indices for this range.

    """
    w0,dw,p0 = hd["CRVAL3"],hd["CD3_3"],hd["CRPIX3"]
    w0 -= p0*dw
    return ( int((w1-w0)/dw), int((w2-w0)/dw) )

def get_header1D(hdr3D):
    """Remove the spatial axes froma a 3D FITS Header."""

    hdr1D = hdr3D.copy()
    for key,val in list(hdr3D.items()):
        if '1' in key or '2' in key:
            del hdr1D[key]
        elif '3' in key:
            hdr1D[key.replace('3','1')] = val
            del hdr1D[key]
    del hdr1D["NAXIS1"]
    hdr1D.insert(2,"NAXIS1")

    hdr1D["NAXIS1"]  = hdr3D["NAXIS3"]
    hdr1D["NAXIS"]   = 1
    hdr1D["WCSDIM"]  = 1
    return hdr1D

def get_header2D(hdr3D):
    """Remove the spectral axis from a 3D FITS Header"""
    hdr2D = hdr3D.copy()
    for key in list(hdr2D.keys()):
        if '3' in key:
            del hdr2D[key]
    hdr2D["NAXIS"]   = 2
    hdr2D["WCSDIM"]  = 2
    return hdr2D

def getWavAxis(hdr):
    """Returns a NumPy array representing the wavelength axis of a cube."""
    if hdr["NAXIS"]==3: return np.array([ hdr["CRVAL3"] + (i-hdr["CRPIX3"])*hdr["CD3_3"] for i in range(hdr["NAXIS3"])])
    elif hdr["NAXIS"]==1: return np.array([ hdr["CRVAL1"] + (i-hdr["CRPIX1"])*hdr["CD1_1"] for i in range(hdr["NAXIS1"])])

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

def trim_cube(fits,params):

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

def coadd(fileList,pxThresh=0.5,expThresh=0.1,PA=0,plot=False,varData=False):
    """Coadd a list of fits images into a master frame.

    Args:
        fileList: List of paths of cubes to be coadded.
        pxThresh (float): Minimum fractional pixel overlap.
            This is the overlap between an input pixel and a pixel in the
            output frame. If a given pixel from an input frame covers less
            than this fraction of an output pixel, its contribution will be
            rejected.
        expThresh (float): Minimum exposure time, as fraction of maximum.
            If an area in the coadd has a stacked exposure time less than
            this fraction of the maximum overlapping exposure time, it will be
            trimmed from the coadd. Default: 0.1.
        PA (float): The desired position-angle of the output data.
        plot (bool): For debugging purposes, show plots of pixel mapping.
        varData (bool): Set to TRUE when coadding variance.

    Returns:
        astropy FITS object tuple: the stacked FITS.

    """

    #
    # STAGE 0: PREPARATION
    #

    # Open custom FITS-3D objects
    fitsList = [fitsIO.open(f) for f in fileList]

    # Extract basic header info
    hdrList    = [ f[0].header for f in fitsList ]
    wcsList    = [ WCS(h) for h in hdrList ]
    pxScales   = np.array([ proj_plane_pixel_scales(wcs) for wcs in wcsList ])

    # Get 2D headers, WCS and on-sky footprints
    h2DList    = [ get2DHeader(h) for h in hdrList]
    w2DList    = [ WCS(h) for h in h2DList ]
    footPrints = np.array([ w.calc_footprint() for w in w2DList ])

    # Exposure times
    expKeys  =  [ "TELAPSE" if inst=="KCWI" else "EXPTIME" for inst in params["INST"] ]
    expTimes =  [ h[expKeys[i]] for i,h in enumerate(hdrList) ]

    # Extract into useful data structures
    xScales,yScales,wScales = ( pxScales[:,i] for i in range(3) )
    pxAreas = [ (xScales[i]*yScales[i]) for i in range(len(xScales)) ]
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
        answer = input("")
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

        print("Aligning wavelength axes.",end='')

        # Adjust each cube to be on new wavelenght axis
        for i,f in enumerate(fitsList):

            print(('.'), end=' ')

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
            if varData: K=np.power(K,2)

            f[0].data = np.apply_along_axis(lambda m: np.convolve(m, K, mode='same'), axis=0, arr=f[0].data)

            f[0].header["NAXIS3"] = len(wNew)
            f[0].header["CRVAL3"] = wNew[0]
            f[0].header["CRPIX3"] = 1

        print("")


    #
    # Stage 2 - SPATIAL ALIGNMENT
    #

    #Take first header as template for coadd header
    hdr0 = h2DList[0]

    #Get 2D WCS
    wcs0 = WCS(hdr0)

    #Get plate-scales
    dx0,dy0 = proj_plane_pixel_scales(wcs0)

    #Make aspect ratio in terms of plate scales 1:1
    if   dx0>dy0: wcs0.wcs.cd[:,0] /= dx0/dy0
    elif dy0>dx0: wcs0.wcs.cd[:,1] /= dy0/dx0
    else: pass

    #Set coadd canvas to desired orientation

    #Try to load orientation from header
    pa0 = None
    for rotKey in ["ROTPA","ROTPOSN"]:
        if rotKey in hdr0:
            pa0=hdr0[rotKey]
            break

    #If no value was found, set to desired PA so that no rotation takes place
    if pa0==None:
        print("No header key for PA (ROTPA or ROTPOSN) found in first input file. Cannot guarantee output PA.")
        pa0 = PA

    #Rotate WCS to the input PA
    wcs0 = rotate(wcs0,pa0-PA)

    #Set new WCS - we will use it later to create the canvas
    wcs0.wcs.set()

    # We don't know which corner is which for an arbitrary rotation, so map each vertex to the coadd space
    x0,y0 = 0,0
    x1,y1 = 0,0
    for fp in footPrints:
        ras,decs = fp[:,0],fp[:,1]
        xs,ys = wcs0.all_world2pix(ras,decs,0)

        xMin,yMin = np.min(xs),np.min(ys)
        xMax,yMax = np.max(xs),np.max(ys)

        if xMin<x0: x0=xMin
        if yMin<y0: y0=yMin

        if xMax>x1: x1=xMax
        if yMax>y1: y1=yMax

    #These upper and lower x-y bounds to shift the canvas
    dx = int(round((x1-x0)+1))
    dy = int(round((y1-y0)+1))

    #
    ra0,dec0 = wcs0.all_pix2world(x0,y0,0)
    ra1,dec1 = wcs0.all_pix2world(x1,y1,0)

    #Set the lower corner of the WCS and create a canvas
    wcs0.wcs.crpix[0] = 1
    wcs0.wcs.crval[0] = ra0
    wcs0.wcs.crpix[1] = 1
    wcs0.wcs.crval[1] = dec0
    wcs0.wcs.set()

    hdr0 = wcs0.to_header()

    #
    # Now that WCS has been figured out - make header and regenerate WCS
    #
    coaddHdr = hdrList[0].copy()

    coaddHdr["NAXIS1"] = dx
    coaddHdr["NAXIS2"] = dy
    coaddHdr["NAXIS3"] = len(wNew)

    coaddHdr["CRPIX1"] = hdr0["CRPIX1"]
    coaddHdr["CRPIX2"] = hdr0["CRPIX2"]
    coaddHdr["CRPIX3"] = 1

    coaddHdr["CRVAL1"] = hdr0["CRVAL1"]
    coaddHdr["CRVAL2"] = hdr0["CRVAL2"]
    coaddHdr["CRVAL3"] = wNew[0]

    coaddHdr["CD1_1"]  = wcs0.wcs.cd[0,0]
    coaddHdr["CD1_2"]  = wcs0.wcs.cd[0,1]
    coaddHdr["CD2_1"]  = wcs0.wcs.cd[1,0]
    coaddHdr["CD2_2"]  = wcs0.wcs.cd[1,1]

    coaddHdr2D = get2DHeader(coaddHdr)
    coaddWCS   = WCS(coaddHdr2D)
    coaddFP = coaddWCS.calc_footprint()


    #Get scales and pixel size of new canvas
    coadd_dX,coadd_dY = proj_plane_pixel_scales(coaddWCS)
    coadd_pxArea = (coadd_dX*coadd_dY)

    # Create data structures to store coadded cube and corresponding exposure time mask
    coaddData = np.zeros((len(wNew),coaddHdr["NAXIS2"],coaddHdr["NAXIS1"]))
    coaddExp  = np.zeros_like(coaddData)

    W,Y,X = coaddData.shape

    if plot:
        fig1,ax = plt.subplots(1,1)
        for fp in footPrints:
            ax.plot( -fp[0:2,0],fp[0:2,1],'k-')
            ax.plot( -fp[1:3,0],fp[1:3,1],'k-')
            ax.plot( -fp[2:4,0],fp[2:4,1],'k-')
            ax.plot( [ -fp[3,0], -fp[0,0] ] , [ fp[3,1], fp[0,1] ],'k-')
        for fp in [coaddFP]:
            ax.plot( -fp[0:2,0],fp[0:2,1],'r-')
            ax.plot( -fp[1:3,0],fp[1:3,1],'r-')
            ax.plot( -fp[2:4,0],fp[2:4,1],'r-')
            ax.plot( [ -fp[3,0], -fp[0,0] ] , [ fp[3,1], fp[0,1] ],'r-')

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

        # Fract frame stores a coverage fraction for each coadd pixel
        fractFrame = np.zeros_like(coaddData)

        # Get wavelength coverage of this FITS
        wavIndices    = np.ones(f[0].data.shape[0],dtype=bool)
        wavIndices[wNew<wav0s[i]] = 0
        wavIndices[wNew>wav1s[i]] = 0

        # Convert to a flux-like unit if the input data is in counts
        if "electrons" in f[0].header["BUNIT"]:

            # Scale data to be in counts per unit time
            if varData: f[0].data /= expTimes[i]**2
            else: f[0].data /= expTimes[i]

            f[0].header["BUNIT"] = "electrons/sec"

        print(("Mapping %s to coadd frame (%i/%i)"%(params["IMG_ID"][i],i+1,len(fitsList))), end=' ')

        if plot:
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
            #inAx.plot(qXin,qYin,'ro')
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


            #skyAx.set_xlim([ra0+0.001,ra1-0.001])
            skyAx.set_ylim([dec0-0.001,dec1+0.001])


        # Loop through spatial pixels in this input frame
        for yj in range(y):

            print(("."), end=' ')
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


                if plot:
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

                        try:
                            # Define BL, TL, TR, BR corners of pixel as coordinates
                            cPixVertices =  np.array( [ [xC-0.5,yC-0.5], [xC-0.5,yC+0.5], [xC+0.5,yC+0.5], [xC+0.5,yC-0.5] ]   )

                            # Create Polygon object and store in array
                            pixCA = box( xC-0.5, yC-0.5, xC+0.5, yC+0.5 )

                            # Calculation fractional overlap between input/coadd pixels
                            overlap = pixIN.intersection(pixCA).area/pixIN.area

                            #print overlap
                            # Add fraction to fraction frame
                            fractFrame[wavIndices,yC,xC] += overlap

                            if varData: overlap=overlap**2

                            # Add data to build frame
                            buildFrame[:,yC,xC] += overlap*f[0].data[:,yj,xk]

                        except: continue
        if plot:
            fig2.canvas.draw()
            plt.waitforbuttonpress()

        #Calculate ratio of coadd pixel area to input pixel area
        pxAreaRatio = coadd_pxArea/pxAreas[i]

        # Max value in fractFrame should be pxAreaRatio - it's the biggest fraction of an input pixel that can add to one coadd pixel
        # We want to use this map now to create a flatFrame - where the values represent a covering fraction for each pixel
        flatFrame = fractFrame/pxAreaRatio

        #Replace zero-values with inf values to avoid division by zero when flat correcting
        flatFrame[flatFrame==0] = np.inf

        #Perform flat field correction for pixels that are not fully covered
        buildFrame /= flatFrame

        #Zero any pixels below user-set pixel threshold, and set flat value to inf
        buildFrame[flatFrame<pxThresh] = 0
        flatFrame[flatFrame<pxThresh] = np.inf

        # Create 3D mask of non-zero voxels from this frame
        M = flatFrame<np.inf

        # Add weight*data to coadd (numerator of weighted mean with exptime as weight)
        if varData: coaddData += (expTimes[i]**2)*buildFrame
        else: coaddData += expTimes[i]*buildFrame

        #Add to exposure mask
        coaddExp += expTimes[i]*M
        coaddExp2D = np.sum(coaddExp,axis=0)
        print("")


    if plot: plt.close()

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

    if varData: coaddData /= coaddExp**2
    else: coaddData /= coaddExp

    # Create FITS object
    coaddHDU = apIO.fits.PrimaryHDU(coaddData)
    coaddFITS = apIO.fits.HDUList([coaddHDU])
    coaddFITS[0].header = coaddHdr

    #Exposure time threshold, relative to maximum exposure time, below which to crop.
    useW = expSpec>expThresh
    useX = expXMap>expThresh
    useY = expYMap>expThresh

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
    return coaddFITS
