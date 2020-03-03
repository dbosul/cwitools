from cwitools import coordinates, parameters

from astropy.io import fits
from astropy.modeling.models import Gaussian1D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils import DAOStarFinder
from scipy.optimize import differential_evolution
from scipy.signal import correlate
from scipy.interpolate import interp1d

import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import warnings
import sys

if sys.platform == 'linux': matplotlib.use('TkAgg')

def get_wavoffset(wav, skyspec, skyline, fit_range = 20, wav_err = 5,
                 std_min = 1, std_max = 30, amp_min = 0, amp_max = 10,
                 plot = False):


    #Start line center at given value
    line_center = skyline
    spec_max = np.max(skyspec)

    #Set upper and lower bounds on gaussian model parameters
    gaussian_bounds = [ (amp_min*spec_max, amp_max*spec_max),
                        (skyline - wav_err, skyline + wav_err),
                        (std_min,std_max)
    ]

    #Create an objective function taking gaussian params and x,y data
    def objFunc(par, *f_args):
        return np.sum( np.power( f_args[1] - gauss1d(par, f_args[0]), 2) )

    #Get x,y args by narrowing down to fitting wavelength range
    fit_ind = (wav >= skyline - fit_range) & (wav <= skyline + fit_range)
    fit_wav = wav[fit_ind]
    fit_spc = skyspec[fit_ind]
    fit_args = (fit_wav, fit_spc)

    #Run optimization
    fit_result = differential_evolution(objFunc, gaussian_bounds, args=fit_args)

    #Separate into parameters
    fit_amp, fit_mean, fit_stddev = fit_result.x

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        ax.step(wav, skyspec, 'k-', label='Sky')
        ax.plot([skyline]*2, [skyspec.min(), skyspec.max()], 'k--', label="Expected Position")
        ax.plot(wav, gauss1D(fit_result.x, wav), 'r-', label="Fit")
        ax.plot([fit_mean]*2, [skyspec.min(), skyspec.max()], 'r--', label="Fit Position")
        ax.set_xlabel("Wavelength [A]", fontsize=14)
        ax.set_ylabel("Flux [norm]", fontsize=14)
        ax.legend()
        fig.tight_layout()
        fig.show()
        plt.waitforbuttonpress()#input("")
        plt.close()

    #Get offset from given value
    wav_off = skyline - fit_mean

    return wav_off

def get_crmatrix3(fitsFile, instrument, skyLine=None, plot=False):
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

    wav_axis = coordinates.get_wav_axis(header)

    #Load sky emission lines
    skyDataDir = os.path.dirname(__file__).replace('/scripts','/data/sky')

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
        else: print("WARNING: Provided skyLine (%.1fA) is outside fittable wavelength range. Using default lists.\n"%skyLine)
    else: print("WARNING: No -skyLine provided. Loading defaults for %s instead.\n"%instrument)

    # Take normalized spatial median of cube
    sky = np.sum(fitsFile[0].data,axis=(1,2))
    sky /=np.max(sky)


    #Run through sky lines until one is useable
    for line in skyLines:

        if wav_axis[0] <= line <= wav_axis[-1]:

            print("Using sky-line at %.2fA"%(line))

            offset = get_wavoffset(wav_axis, sky, line, wav_err=fwhm_A, plot=plot)

            new_crval3 = crval3 + offset
            new_crpix3 = crpix3

            print("Measured Offset: %.2fA"%offset)
            print("---------------------------")
            return new_crval3,new_crpix3

    #If we get to here, no line was found
    print("No known sky lines in range %.1f-%.1f. Wavelength solution will not be corrected.")

    return crval3,crpix3

def fixwcs(paramPath,icubeType,instrument,fixRADEC=True,fixWav=False,
           skyLine=None,RA=None, DEC=None, plot=False, alignWav=False):
    """Corrects the world-coordinate system of cubes using interactive tools.

    Args:
        paramPath (str): Path to the CWITools parameter file.
        icubeType (str): Type of icube to work with.
        instrument (str): Which CWI we're working with here (PCWI/KCWI)
        fixRADEC (bool): Fix the spatial axes (Default: True)
        fixWav (bool): Fix the wavelength axis (Default: True)
        alignWav (bool): Align the wavelength axes without absolute correction
            alignWav is an alternative to fixWav, intended for use when no sky
            line is available for absolute correction.
        skyLine (float): Known wavelength of a fittable sky-line.
            This parameter is required for fixing the wavelength solution.
        RA (float): RA (dd.dd) of source to use (overrides param file)
        DEC (float): DEC (dd.dd) of source to use (overrides param file)
        plot (bool): Display diagnostic plots (Default: False)

    """

    #Load params
    par = parameters.load_params(paramPath)

    if RA == None:
        if par["ALIGN_RA"] == None: RA = par["TARGET_RA"]
        else: RA = par["ALIGN_RA"]

    if DEC == None:
        if par["ALIGN_DEC"] == None: DEC = par["TARGET_DEC"]
        else: DEC = par["ALIGN_DEC"]

    #Find icubes files
    ifileList = parameters.find_files(
        par["ID_LIST"],
        par["INPUT_DIRECTORY"],
        icubeType,
        depth=par["SEARCH_DEPTH"]
    )

    #An alternative to -fixWav, -alignWav checks if the wavelength axes are
    #all consistent with each other, but does not correct the absolute value
    #This is useful when there are no clear sky-lines to fit to for -fixWav
    if alignWav:


        #Extract wavelength axes and normalized sky spectra from each fits
        wavs, spcs = [], []
        crpix3s_input = []
        for i, filename in enumerate(ifileList):
            skyFile = filename.replace('icube','scube')
            skyFits = fits.open(skyFile)
            crpix3s_input.append(skyFits[0].header["CRPIX3"])
            wav = coordinates.get_wav_axis(skyFits[0].header)
            sky = np.sum(skyFits[0].data, axis=(1, 2))
            sky[np.isnan(sky)] = 0
            sky /= np.max(sky)
            spcs.append(sky)
            wavs.append(wav)


        #Create common wavelength axis to interpolate sky spectra onto
        w0 = np.min(wavs)
        w1 = np.max(wavs)
        dws = [x[1]-x[0] for x in wavs]
        dw_min = np.min(dws)
        Nw =  int( (w1-w0)/dw_min ) + 1
        wav_common = np.linspace(w0, w1, Nw)

        #Interpolate (linearly) spectra onto common wavelength axis
        spc_interps = [ interp1d(wavs[i], spcs[i])(wav_common) for i in range(len(spcs))]

        if plot: fig, ax = plt.subplots(1, 1)

        #Cross-correlate interpolated spectra to look for shifts between them
        corrs = []
        for i, spc_int in enumerate(spc_interps):
            if plot: ax.plot(wav_common, spc_interps[i], label=ifileList[i])
            corr_ij = correlate(spc_interps[0], spc_int, mode='full')


            corrs.append(np.nanargmax(corr_ij))

        #Subtract first self-correlation (reference point)
        corrs = np.array(corrs)
        corrs -= corrs[0]

        crpix3s_alignwav = crpix3s_input
        if np.all(corrs == 0):
            print("AlignWav: Input wavelength axes are already well aligned.")
            print("AlignWav: No wavelength correction applied.")

        else:
            print("AlignWav: Some input wavelength axes are mis-aligned.")
            print("AlignWav: Applying fixes:")

        for i, crpix3 in enumerate(crpix3s_input):
            crpix3s_alignwav[i] -= corrs[i]
            print("\t%s: CRPIX3 -= %i px" % (ifileList[i], corrs[i]))

        if plot:
            ax.set_xlabel("Wavelength [A]", fontsize=14)
            ax.set_ylabel("Flux [norm]", fontsize=14)
            ax.set_title("AlignWav: Relative wavelength axis alignment")
            fig.tight_layout()
            fig.show()
            input("")
            plt.close()

    #Run through all images now and perform corrections
    for i,fileName in enumerate(ifileList):

        fileNameShort = fileName.split('/')[-1].split('_')[0]
        currentHeader = fits.getheader(fileName)

        #Get current CD matrix
        crval1,crval2,crval3 = ( currentHeader["CRVAL%i"%(k+1)] for k in range(3) )
        crpix1,crpix2,crpix3 = ( currentHeader["CRPIX%i"%(k+1)] for k in range(3) )


        #Get RA/DEC values if fixWAV requested
        if fixRADEC:


            print("\n\nCorrecting spatial axes for %s"%fileNameShort)
            print("------------------------------------------------")
            radecFITS = fits.open(fileName)
            crval1,crval2,crpix1,crpix2 = get_crmatrix12(radecFITS, RA, DEC, instrument, plot=plot)
            radecFITS.close()

        #Get wavelength WCS values if fixWav requested
        if fixWav:

            skyFile   = fileName.replace('icube','scube')
            if os.path.isfile(skyFile):

                print("\nCorrecting wavelength axis for %s"%fileNameShort)
                print("------------------------------------------------")
                skyFITS   = fits.open(skyFile)
                crval3,crpix3 = get_crmatrix3(skyFITS,instrument,skyLine=skyLine, plot=plot)
                skyFITS.close()
            else:
                print("Sky cube could not be found for %s.\nWavelength solution will not be corrected."%fileName)

        #Create lists of crval/crpix values, whether updated or not
        crvals = [ crval1, crval2, crval3 ]
        crpixs = [ crpix1, crpix2, crpix3 ]

        print("\nNEW CR MATRIX:")
        print("---------------------------------")
        print("%5s %10s %10s"%("AXIS","CRVAL","CRPIX"))
        print("*********************************")
        for ax in range(3):
            if ax<2: print("%5i %10.5f %10i"%(ax+1,crvals[ax],crpixs[ax]))
            else: print("%5i %10.2f %10i"%(ax+1,crvals[ax],crpixs[ax]))
        print("*********************************")
        #Make list of relevant cubes to be corrected - scube doesn't matter as much
        cubetypes = ['icube','scube','ocube','vcube']

        #Load fits, modify header and save for each cube type
        for ct in cubetypes:

            #Get filepath for this cube
            filePath = fileName.replace('icube',ct)

            #Try to load, but continue upon failure
            try: data,header = fits.getdata(filePath,header=True)
            except:
                print("Could not open %s. Not corrected." % filePath)
                continue

            #Fix each of the header values
            for k in range(3):

                header["CRVAL%i"%(k+1)] = crvals[k]
                header["CRPIX%i"%(k+1)] = crpixs[k]

            #Save WCS corrected cube
            wcPath = filePath.replace('.fits','.wc.fits')
            newFits = fits.HDUList([fits.PrimaryHDU(data)])
            newFits[0].header = header
            newFits.writeto(wcPath,overwrite=True)
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
    wavelength_corr = parser.add_mutually_exclusive_group()
    wavelength_corr.add_argument('-fixWav',
                        help='Correct wavelength axes by fitting to a known sky line',
                        action='store_true'
    )
    wavelength_corr.add_argument('-alignWav',
                        help='Align wavelength axes using cross-correlation.',
                        action='store_true'
    )
    parser.add_argument('-skyLine',
                        type=float,
                        metavar='float',
                        help='Wavelength of sky line to use for -fixWav to use. (Angstrom)'
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
    parser.add_argument('-plot',help="Display fits with Matplotlib.",action='store_true')

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
        alignWav=args.alignWav,
        plot=args.plot
    )

if __name__=="__main__": main()
