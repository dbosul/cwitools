"""CWITools library for 3D data-cube manipulation.

This module contains a number of useful functions for manipulating
three-dimensional FITS data cubes."""

from . import qso
from . import params

import numpy as np
import os
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
