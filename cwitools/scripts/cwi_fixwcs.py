
"""CWITools QSO-Finder class for interactive PSF fitting.

This module contains the class definition for the interactive tool 'QSO Finder.'
QSO finder is used to accurately locate point sources (usually QSOs) when
running fixWCS in CWITools.reduction.

"""
from cwitools.libs import cubes,params

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from photutils import DAOStarFinder

import argparse
import numpy as np
import os
import warnings

def get_crmatrix12(inputFits,src_ra,src_dec,instrument):

    cube = inputFits[0].data.copy()
    header = inputFits[0].header

    wcs3D  = WCS(header)
    px_scl = proj_plane_pixel_scales(wcs3D)
    print(px_scl)

    wavgood0 = header["WAVGOOD0"]
    wavgood1 = header["WAVGOOD1"]

    wav_axis = cubes.get_wavaxis(header)

    use_wav = (wav_axis > wavgood0) & (wav_axis < wavgood1)
    cube[use_wav == 1] = 0

    wl_img = np.sum(cube,axis=0)

    #Run source finder
    if instrument=="KCWI":
        major_axis = 90 #Slices run vertically


    starfinder = DAOStarFinder(fwhm=, threshold=auto*stddev)
    autoSrcs = daofind(wlImg)

    #Get list of peak values
    peaks = list(autoSrcs['peak'])

    #Make list of sources
    sources = []
    for i in range(len(autoSrcs['xcentroid'])):
        sources.append((autoSrcs['xcentroid'][i], autoSrcs['ycentroid'][i]))



def get_crmatrix3(fitsFile,instrument,skyLine=None):
    """Measures and returns the correct header values for the wavelength axis.

    Args:
        fits (astropy FITS object): The FITS file to be corrected.
        instrument (str): The instrument being used ('PCWI' or 'KCWI').
        skyLine (float): The precise wavelength of a known, fittable skyLine.

    Returns:
        String tuple: Corrected CRVAL3, CRPIX3 header values.

    """
    #Extract header info
    header = fitsFile[0].header
    n_wav  = len(fitsFile[0].data)

    crpix3 = header["CRPIX3"]
    crval3 = header["CRVAL3"]

    wavgood0 = header["WAVGOOD0"]
    wavgood1 = header["WAVGOOD1"]

    wav_axis = cubes.get_wavaxis(header)

    #Load sky emission lines
    skyDataDir = os.path.dirname(__file__).replace('/libs','/data/sky')

    if instrument=="PCWI":
        skyLines = np.loadtxt(skyDataDir+"/palomar_lines.txt")
        fwhm_A = 5
    elif instrument=="KCWI":
        skyLines = np.loadtxt(skyDataDir+"/keck_lines.txt")
        fwhm_A = 3

    else: raise ValueError("Instrument not recognized.")

    #If user provided sky line and it is valid, add it at start of line list
    if skyLine!=None:
        if skyLine>wav_axis[0]+fwhm_A and skyLine<wav_axis[-1]-fwhm_A:
            skyLines = np.insert(skyLines,0,skyLine)
        else: warnings.warn("Provided skyLine (%.1fA) is outside fittable wavelength range. Using default lists."%skyLine)
    else: warnings.warn("No -skyLine provided. Loading defaults for %s instead."%instrument)

    # Take normalized spatial median of cube
    sky = np.sum(fitsFile[0].data,axis=(1,2))
    sky /=np.max(sky)


    #Run through sky lines until one is useable
    for l in skyLines:

        if wav[0]<=l<=wav[-1]:

            offset = getWavOffset(wav,sky,l,dW=fwhm_A)

            new_crval3 = crval3 + offset
            new_crpix3 = crpix3

            return new_crval3,new_crpix3

    #If we get to here, no line was found
    warnings.warn("No known sky lines in range %.1f-%.1f. Wavelength solution will not be corrected.")
    return crval3,crpix3

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
    parameters = params.loadparams(paramPath)

    if RA == None:
        if parameters["ALIGN_RA"] == None: RA = parameters["TARGET_RA"]
        else: RA = parameters["ALIGN_RA"]

    print(DEC)
    if DEC == None:
        if parameters["ALIGN_DEC"] == None: DEC = parameters["TARGET_DEC"]
        else: DEC = parameters["ALIGN_DEC"]
    print(RA)
    print(DEC)
    print(parameters)
    #Find icubes files
    ifileList = params.findfiles(parameters,icubeType)

    #Run through all images now and perform corrections
    for i,fileName in enumerate(ifileList):

        currentHeader = fits.getheader(fileName)

        #Get current CD matrix
        crval1,crval2,crval3 = ( currentHeader["CRVAL%i"%(k+1)] for k in range(3) )
        crpix1,crpix2,crpix3 = ( currentHeader["CRPIX%i"%(k+1)] for k in range(3) )


        #Get RA/DEC values if fixWAV requested
        if fixRADEC:

            radecFITS = fits.open(fileName)
            crval1,crval2,crpix1,crpix2 = get_crmatrix12(radecFITS,RA,DEC)
            radecFITS.close()

        #Get wavelength WCS values if fixWav requested
        if fixWav:

            skyFile   = fileName.replace('icube','scube')
            skyFITS   = fits.open(skyFile)
            crval3,crpix3 = get_crmatrix3(skyFITS,instrument,skyLine=skyLine)
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
            try: data,header = fits.getdata(filePath,header=True)
            except:
                warnings.warn("Could not open %s. Not WCS corrected." % filePath)
                continue

            #Fix each of the header values
            for k in range(3):

                header["CRVAL%i"%(k+1)] = crvals[k]
                header["CRPIX%i"%(k+1)] = crpixs[k]

            #Save WCS corrected cube
            wcPath = filePath.replace('.fits','.wc.fits')
            newFits = cubes.make_fits(data,header)
            newFits.writeto(wcPath)
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
    parser.add_argument('instrument',
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

    fixwcs(args.params,args.icubetype,args.instrument,
        fixRADEC=args.fixRADEC,
        fixWav=args.fixWav,
        skyLine=args.skyLine,
        RA=args.ra,
        DEC=args.dec,
    )

if __name__=="__main__": main()
