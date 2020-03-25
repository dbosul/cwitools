"""Create a WCS correction table by measuring the input data."""
from cwitools import coordinates, reduction, parameters, utils
from astropy.io import fits
import argparse
import numpy as np
import warnings
import sys

def main():

    parser = argparse.ArgumentParser(description='Measure WCS parameters and save to WCS correction file.')
    parser.add_argument('clist',
                        metavar="cube_list",
                        type=str,
                        help='CWITools cube list.'
    )
    parser.add_argument('-ctype',
                        type=str,
                        metavar="<cube_type>",
                        help='Type of input cube to work with.',
                        default="icubes.fits"
    )
    parser.add_argument('-xymode',
                        help='Which method to use for correcting X/Y axes',
                        default="src_fit",
                        choices=["src_fit", "none"]
    )
    parser.add_argument('-ra',
                        metavar="<dd.ddd>",
                        type=float,
                        help="Right-ascension of source to fit.",
                        default=None
    )
    parser.add_argument('-dec',
                        metavar="<dd.ddd>",
                        type=float,
                        help="Declination of source to fit.",
                        default=None
    )
    parser.add_argument('-box',
                        metavar="<box_size>",
                        type=float,
                        help="Box size (arcsec) for fitting source.",
                        default=10
    )
    parser.add_argument('-zmode',
                        help='Which method to use for correcting z-azis',
                        default="none",
                        choices=["none", "xcor"]
    )
    parser.add_argument('-plot',
                        help="Display fits with Matplotlib.",
                        action='store_true'
    )
    parser.add_argument('-out',
                        metavar="",
                        help="Output table name.",
                        default=None
    )
    parser.add_argument('-log',
                        metavar="<log_file>",
                        type=str,
                        help="Log file to save this command in",
                        default=None
    )
    args = parser.parse_args()

    #Load the default alignment RA and DEC
    if args.xymode == 'src_fit':
        ra = args.ra
        dec = args.dec
        if ra == None and dec == None:
            raise ValueError("-ra and -dec must be set if using src_fit. ")

    #Parse cube list
    clist = utils.parse_cubelist(args.clist)

    #Load input files
    in_files = utils.find_files(
        clist["ID_LIST"],
        clist["INPUT_DIRECTORY"],
        args.ctype,
        depth=clist["SEARCH_DEPTH"]
    )

    #Prepare table output
    outstr = "INPUT_DIRECTORY=%s\n" % clist["INPUT_DIRECTORY"]
    outstr += "SEARCH_DEPTH=%i\n" % clist["SEARCH_DEPTH"]
    outstr += "#%19s %15s %15s %10s %10s %10s %10s\n" % (
    "ID", "CRVAL1", "CRVAL2", "CRVAL3", "CRPIX1", "CRPIX2", "CRPIX3")

    #If wavelength alignment has been requested
    if args.zmode == "align":

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

        crpix1, crpix2 = reduction.get_crpix12(in_fits, ra, dec,
            plot=args.plot,
            box_size=args.box
        )

        outstr += ">%19s %15.7f %15.7f %10.3f %10.1f %10.1f %10.1f\n" % (
        clist["ID_LIST"][i], ra, dec, crval3s[i], crpix1, crpix2, crpix3s[i])

    if args.out == None:
        outfilename = args.clist.replace(".list", ".wcs")
    else:
        outfilename = args.out

    #Create the correction file
    outfile = open(outfilename, 'w')
    outfile.write(outstr)
    outfile.close()
    print("Saved %s" % outfilename)

    #Log command upon successful completion
    utils.log_command(sys.argv, logfile=args.log)

if __name__=="__main__":
    main()
