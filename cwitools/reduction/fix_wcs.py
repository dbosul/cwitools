from .. import libs

from astropy.io import fits

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt

def run(paramPath,icubeType,instrument,fixRADEC=True,fixWav=False,skyLine=None,RA=None, DEC=None):
    """Corrects the world-coordinate system of data cubes using interactive tools.

    Args:
        paramPath (str): Path to the CWITools parameter file.
        icubeType (str): Type of icube to work with (e.g. icubes.fits, icuber.fits)
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

#If executed on the command line:
if __name__=="__main__":

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Use RA/DEC and Wavelength reference points to adjust WCS.')


    parser.add_argument('paramFile',
                        type=str,
                        metavar='str',
                        help='CWITools Parameter file (used to load cube list etc.)'
    )
    parser.add_argument('icubeType',
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
    args.simpleMode = True if args.simpleMode=="True" else False

    run(args.paramFile,args.icubeType,args.instrument
        fixRADEC=args.fixRADEC,
        fixWav=args.fixWav,
        skyLine=args.skyLine,
        RA=args.ra,
        DEC=args.dec,
    )
