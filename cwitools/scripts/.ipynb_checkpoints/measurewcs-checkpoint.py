"""Create a WCS correction table by measuring the input data."""
from astropy.io import fits
from astropy.wcs import WCS
from cwitools import coordinates, reduction, utils
from datetime import datetime

import argparse
import cwitools
import numpy as np
import warnings
import sys

# testing
import pdb


def parser_init():

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
                        choices=["src_fit", "xcor", "none"]
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
                        default=None #10
    )
    parser.add_argument('-crpix1s',
                        metavar="<list of floats>",
                        nargs='+',
                        help="List of CRPIX1. Set to -1s to use the header of the corresponding files.",
                        default=None
                       )
    parser.add_argument('-crpix2s',
                        metavar="<list of floats>",
                        nargs='+',
                        help="List of CRPIX2. Set to -1s to use the header of the corresponding files.",
                        default=None
                       )
    parser.add_argument('-background_sub',
                        metavar="",
                        help="Subtract background before xcor.",
                        default=False
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
                        help="Log file to save output in.",
                        default=None
    )
    parser.add_argument('-silent',
                        help="Set flag to suppress standard terminal output.",
                        action='store_true'
    )

    return parser


def core(args, parser):
    
    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_MEASUREWCS:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)        

    #Parse cube list
    clist = utils.parse_cubelist(args.clist)
    
    #Load the default alignment RA and DEC
    if args.xymode == 'src_fit':
        ra = args.ra
        dec = args.dec
        if ra == None and dec == None:
            raise ValueError("-ra and -dec must be set if using src_fit.")
        if args.box == None:
            utils.output('\t-ra and -dec not set for src_fit, setting to 10 arcsec.\n')
            args.box = 10. 
    elif args.xymode == 'xcor':
        crpix1s = np.zeros(len(clist['ID_LIST'])) - 1
        crpix2s = np.zeros(len(clist['ID_LIST'])) - 1
        if (args.crpix1s is None) != (args.crpix2s is None):
            raise ValueError("-crpix1s and -crpix2s must be set simultaneously.")
        if args.crpix1s is not None:
            for i, crpix1 in enumerate(args.crpix1s):
                crpix1s[i] = float(crpix1)
        if args.crpix2s is not None:
            for i, crpix2 in enumerate(args.crpix2s):
                crpix2s[i] = float(crpix2)

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

    crval3s = [-1 for file in in_files]
    crpix3s = [-1 for file in in_files]

    #If wavelength alignment has been requested
    if args.zmode == "xcor":
        utils.output("\tAligning z-axes...\n")
        sky_fits = [fits.open(x.replace('icube','scube')) for x in in_files]
        crpix3_vals_new = reduction.xcor_crpix3(sky_fits)

        for i, crpix3 in enumerate(crpix3_vals_new):

            #If no change in CRPIX3 value - set to -1 for later
            if crpix3 == sky_fits[i][0].header["CRPIX3"]:
                crval3s[i] = -1
                crpix3s[i] = -1

            #If changed, set to new value for later
            else:
                crpix3s[i] = crpix3_vals_new[i]

        if np.all(np.array(crpix3s) == -1):
            utils.output("\t\tInput z-axes already well aligned.\n\n")


    if args.xymode == "src_fit":
        utils.output("\tFitting source positions...\n")
        for i, in_file in enumerate(in_files):

            in_fits = fits.open(in_file)

            crpix1, crpix2 = reduction.fit_crpix12(in_fits, ra, dec,
                plot=args.plot,
                box_size=args.box
            )
            
            istring = "\t\t{0}: {1:.2f}, {2:.1f}\n".format(clist["ID_LIST"][i], crpix1, crpix2)
            utils.output(istring)

            outstr += ">%19s %15.7f %15.7f %10.3f %10.1f %10.1f %10.1f\n" % (
            clist["ID_LIST"][i], ra, dec, crval3s[i], crpix1, crpix2, crpix3s[i])
            
    elif args.xymode == "xcor":
        utils.output("\tCross-correlating in 2D...\n")
        for i, in_file in enumerate(in_files):

            in_fits = fits.open(in_file)
            if (crpix1s[i] != -1) and (crpix2s[i] != -1):
                crpix = [crpix1s[i], crpix2s[i]]
            else:
                crpix = None
                
            # Use i=0  as the reference image
            if i == 0:
                hdr = in_fits[0].header
                
                if args.ra is not None:
                    wcs0 = WCS(hdr)
                    tmp = wcs0.all_world2pix(args.ra, args.dec, 1, 1)
                    
                    crpix1, crpix2, crval1, crval2 = float(tmp[0]), float(tmp[1]), args.ra, args.dec
                else:
                    crpix1, crpix2, crval1, crval2 = hdr['CRPIX1'], hdr['CRPIX2'], hdr['CRVAL1'], hdr['CRVAL2']
                
                ref_fits = in_fits
            else:
                crpix1, crpix2, crval1, crval2 = reduction.xcor_cr12(in_fits,
                    ref_fits,
                    ra=args.ra, 
                    dec=args.dec,
                    crpix=crpix,
                    background_subtraction = args.background_sub,
                    plot=int(args.plot)*2,
                    box_size=args.box
                )
            
            istring = "\t\t{0}: {1:.2f}, {2:.1f}, {3:.4f}, {4:.4f}\n".format(clist["ID_LIST"][i], crpix1, crpix2, crval1, crval2)
            utils.output(istring)

            outstr += ">%19s %15.7f %15.7f %10.3f %10.1f %10.1f %10.1f\n" % (
            clist["ID_LIST"][i], crval1, crval2, crval3s[i], crpix1, crpix2, crpix3s[i])
        


    if args.out == None:
        outfilename = args.clist.replace(".list", ".wcs")
    else:
        outfilename = args.out

    #Create the correction file
    outfile = open(outfilename, 'w')
    outfile.write(outstr)
    outfile.close()

    utils.output("\n\tSaved corrections table to %s\n" % outfilename)

if __name__=="__main__":
    parser = parser_init()
    args = parser.parse_args()
    core(args, parser)


def main(clist, ctype=None, xymode=None, ra=None, dec=None, box=None,
         crpix1s=None, crpix2s=None, background_sub=None, zmode=None, 
         plot=None, out=None, log=None, silent=None):
        
    parser = parser_init()
    
    # construct args
    str_list=[clist]
    if ctype is not None:
        str_list.append('-ctype')
        str_list.append(str(ctype))
    if xymode is not None:
        str_list.append('-xymode')
        str_list.append(str(xymode))
    if ra is not None:
        str_list.append('-ra')
        str_list.append(str(ra))
    if dec is not None:
        str_list.append('-dec')
        str_list.append(str(dec))
    if box is not None:
        str_list.append('-box')
        str_list.append(str(box))
    if crpix1s is not None:
        str_list.append('-crpix1s')
        for i in crpix1s:
            str_list.append(str(i))
    if crpix2s is not None:
        str_list.append('-crpix2s')
        for i in crpix2s:
            str_list.append(str(i))
    if background_sub is not None:
        str_list.append('-background_sub')
        str_list.append(str(background_sub))
    if zmode is not None:
        str_list.append('-zmode')
        str_list.append(str(zmode))
    if plot is not None:
        if plot:
            str_list.append('-plot')
    if out is not None:
        str_list.append('-out')
        str_list.append(str(out))
    if log is not None:
        str_list.append('-log')
        str_list.append(str(log))
    if silent is not None:
        if silent:
            str_list.append('-silent')
    args = parser.parse_args(str_list)
        
    core(args, parser)

