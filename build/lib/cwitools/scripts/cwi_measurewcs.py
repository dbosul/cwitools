
"""CWITools QSO-Finder class for interactive PSF fitting.

This module contains the class definition for the interactive tool 'QSO Finder.'
QSO finder is used to accurately locate point sources (usually QSOs) when
running fixWCS in CWITools.reduction.

"""
from cwitools import coordinates, reduction, parameters

from astropy.io import fits
import argparse
import numpy as np
import warnings

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Measure WCS parameters and save to WCS correction file.')
    parser.add_argument('param',
                        type=str,
                        metavar='paramfile',
                        help='CWITools parameter file.'
    )
    parser.add_argument('cubetype',
                        type=str,
                        help='Type of cubes to work with. Must be icube.fits/icubes.fits etc.',
                        choices=['icube.fits','icubep.fits','icubed.fits','icubes.fits','icuber.fits']
    )
    parser.add_argument('-plot',
                        help="Display fits with Matplotlib.",
                        action='store_true'
    )
    parser.add_argument('-out',
                        help="Correction file to save. Default is same as parameter file with .wcs extension",
                        default=None
    )
    parser.add_argument('-fit_box',
                        type=float,
                        help="Size of box around initial RA/DEC to fit source, in arcsec. Default=10.",
                        default=10
    )
    parser.add_argument('-alignZ',
                        help='Set this flag to align the input wavelength axes.',
                        action='store_true'
    )
    parser.add_argument('-fitRADEC',
                        help='Set this flag to align the input axes without absolute correction.',
                        action='store_true'
    )
    args = parser.parse_args()

    #try:
    par = parameters.load_params(args.param)
    #except:
    #        raise ValueError("Could not load %s" % args.param)

    #Load the default alignment RA and DEC
    crval1 = par["ALIGN_RA"] if par["ALIGN_RA"] != "TARGET_RA" else par["TARGET_RA"]
    crval2 = par["ALIGN_DEC"] if par["ALIGN_DEC"] != "TARGET_DEC" else par["TARGET_DEC"]

    in_files = parameters.find_files(
        par["ID_LIST"],
        par["INPUT_DIRECTORY"],
        args.cubetype,
        depth=par["SEARCH_DEPTH"]
    )
    outstr = "INPUT_DIRECTORY=%s\n" % par["INPUT_DIRECTORY"]
    outstr += "SEARCH_DEPTH=%i\n" % par["SEARCH_DEPTH"]
    outstr += "#%19s %15s %15s %10s %10s %10s %10s\n" % (
    "ID", "CRVAL1", "CRVAL2", "CRVAL3", "CRPIX1", "CRPIX2", "CRPIX3")

    #If wavelength alignment has been requested
    if args.alignZ:

        sky_fits = [fits.open(x.replace('icube','scube')) for x in in_files]
        crpix3_vals_new = reduction.align_crpix3(sky_fits)
        for i, crpix3 in enumerate(crpix3_vals_new):

            #If no change in CRPIX3 value - set to -1 for later
            if crpix3 == sky_fits[i][0].header["CRPIX3"]:
                crval3s[i] = -1
                crpix3s[i] = -1

            #If changed, set to new value for later
            else:
                crpix3s[i] = crpix3_vals_new[i]

        if np.all(np.array(crpix3s) == -1):
            print("Wavelength axes already aligned. No corrections applied.")

    #If no wavelength correction has been requested
    else:

        crval3s = [-1 for file in in_files]
        crpix3s = [-1 for file in in_files]

    print("Fitting source positions...")
    for i, in_file in enumerate(in_files):

        print("\t%s" % in_file)
        in_fits = fits.open(in_file)

        crpix1, crpix2 = reduction.get_crpix12(in_fits, crval1, crval2,
            plot=args.plot,
            box_size=args.fit_box
        )

        outstr += ">%19s %15.7f %15.7f %10.3f %10.1f %10.1f %10.1f\n" % (
        par["ID_LIST"][i], crval1, crval2, crval3s[i], crpix1, crpix2, crpix3s[i])


    if args.out == None:
        outfilename = args.param.replace(".param", ".wcs")
    else:
        outfilename = args.out

    #Create the correction file
    outfile = open(outfilename, 'w')
    outfile.write(outstr)
    outfile.close()
    print("Saved %s" % outfilename)



if __name__=="__main__":
    main()
