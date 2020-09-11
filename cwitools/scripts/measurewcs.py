"""Measure WCS: Create a WCS correction table by measuring the input data."""

#Standard Imports
import argparse
import warnings

#Third-party Imports
from astropy.io import fits
from astropy.wcs import WCS

#Local Imports
from cwitools import reduction, utils, config

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description='Measure WCS parameters and save to WCS correction file.'
        )
    parser.add_argument(
        'clist',
        metavar="cube_list",
        type=str,
        help='CWITools cube list.'
        )
    parser.add_argument(
        '-ctype',
        type=str,
        metavar="<cube_type>",
        help='Type of input cube to work with.',
        default="icubes.fits"
        )
    parser.add_argument(
        '-xymode',
        help='Which method to use for correcting X/Y axes',
        default="src_fit",
        choices=["src_fit", "xcor", "none"]
        )
    parser.add_argument(
        '-ra',
        metavar="<dd.ddd>",
        type=float,
        help="Right-ascension of source to fit.",
        default=None
        )
    parser.add_argument(
        '-dec',
        metavar="<dd.ddd>",
        type=float,
        help="Declination of source to fit.",
        default=None
        )
    parser.add_argument(
        '-box',
        metavar="<box_size>",
        type=float,
        help="Box size (arcsec) for fitting source or cross-correlating.",
        default=None
        )
    parser.add_argument(
        '-crpix1s',
        metavar="<list of floats>",
        nargs='+',
        help="List of CRPIX1. Use 'H' to use existing header value.",
        default=None
        )
    parser.add_argument(
        '-crpix2s',
        metavar="<list of floats>",
        nargs='+',
        help="List of CRPIX2. Use 'H' to use existing header value.",
        default=None
        )
    parser.add_argument(
        '-background_sub',
        metavar="",
        help="Subtract background before xcor.",
        default=False
        )
    parser.add_argument(
        '-zmode',
        help='Which method to use for correcting z-azis',
        default=None,
        choices=['none', "xcor"]
        )
    parser.add_argument(
        '-plot',
        help="Display fits with Matplotlib.",
        action='store_true'
        )
    parser.add_argument(
        '-out',
        metavar="",
        help="Output table name.",
        default=None
        )
    parser.add_argument(
        '-log',
        metavar="<log_file>",
        type=str,
        help="Log file to save output in.",
        default=None
        )
    parser.add_argument(
        '-silent',
        help="Set flag to suppress standard terminal output.",
        action='store_true'
        )

    return parser

def main(clist, ctype="icubes.fits", xymode="src_fit", ra=None, dec=None, box=10.0,
         crpix1s=None, crpix2s=None, background_sub=False, zmode='none',
         plot=False, out=None, log=None, silent=None):
    """Automatically create a WCS correction table for a list of input cubes.

    Args:
        clist (str): Path to the CWITools .list file
        ctype (str): File extension for type of cube to use as input.
            Default value: 'icubes.fits' (.fits extension should be included)
        xymode (str): Method to use for XY alignment:
            'src_fit': Fit 1D profiles to a known point source (interactive)
            'xcor': Perform 2D (XY) cross-correlation of input images.
            'none': Do not align the spatial axes.
        ra (float): Decimal RA coordinate of source, if using 'src_fit'
        dec (float): Decimal DEC coordinate of source, if using 'src_fit'
        box (float): Size of box (in arcsec) to use for finding/fitting source,
            if using 'src_fit'
        crpix1s (list): List of CRPIX1 values, if
    """


    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("MEASURE_WCS", locals())

    #Parse cube list
    cdict = utils.parse_cubelist(clist)

    #Load input files
    in_files = utils.find_files(
        cdict["ID_LIST"],
        cdict["INPUT_DIRECTORY"],
        ctype,
        depth=cdict["SEARCH_DEPTH"]
    )
    #Load scube (or ocube) files
    int_fits = [fits.open(x) for x in in_files]
    sky_fits = [fits.open(x.replace("icube", "scube")) for x in in_files]

    #Prepare table output
    outstr = "INPUT_DIRECTORY=%s\n" % cdict["INPUT_DIRECTORY"]
    outstr += "SEARCH_DEPTH=%i\n" % cdict["SEARCH_DEPTH"]
    outstr += "#%19s %15s %15s %10s %10s %10s %10s\n" % (
        "ID", "CRVAL1", "CRVAL2", "CRVAL3", "CRPIX1", "CRPIX2", "CRPIX3")

    #WAVELENGTH ALIGNMENT - XCOR
    if zmode == "xcor":
        utils.output("\tAligning z-axes...\n")
        crval3s = [i_f[0].header["CRVAL3"] for i_f in int_fits]
        crpix3s = reduction.xcor_crpix3(sky_fits)

    elif zmode == "none":
        warnings.warn("No wavelength WCS correction applied.")
        crval3s = [i_f[0].header["CRVAL3"] for i_f in int_fits]
        crpix3s = [i_f[0].header["CRPIX3"] for i_f in int_fits]
    else:
        raise ValueError("zmode can only be 'none' or 'xcor'")

    #Basic checks and user-message for each xy-mode
    if xymode == "src_fit":
        if ra is None and dec is None:
            raise ValueError("'ra' and 'dec' must be set if using src_fit.")
        if box is None:
            box = 10.0
        utils.output("\tFitting source positions...\n")

    elif xymode == "xcor":
        if (crpix1s is None) != (crpix2s is None):
            raise ValueError("'crpix1s' and 'crpix2s' must be set together")
        utils.output("\tCross-correlating in 2D...\n")

    #SPATIAL ALIGNMENT
    for i, i_f in enumerate(int_fits):

        hdr = i_f[0].header

        if xymode == "src_fit":
            crval1, crval2 = ra, dec
            crpix1, crpix2 = reduction.fit_crpix12(
                i_f, crval1, crval2,
                plot=plot,
                box_size=box
            )

            istring = "\t\t{0}: {1:.2f}, {2:.1f}\n".format(cdict["ID_LIST"][i], crpix1, crpix2)
            utils.output(istring)

        #SPATIAL ALIGNMENT  - XCOR
        elif xymode == "xcor":

            #If CRPIX1s given, and current value is not 'Header' indicator
            if crpix1s is not None and crpix1s[i] != 'H' and crpix2s[i] != 'H':
                crpix = [float(crpix1s[i]), float(crpix2s[i])]
            else:
                crpix = None

            # Use i=0  as the reference image
            if i == 0:

                if ra is not None:
                    wcs0 = WCS(hdr)
                    tmp = wcs0.all_world2pix(ra, dec, 1, 1)
                    crpix1, crpix2 = float(tmp[0]), float(tmp[1])
                    crval1, crval2 = ra, dec

                else:
                    crpix1, crpix2 = hdr['CRPIX1'], hdr['CRPIX2']
                    crval1, crval2 = hdr['CRVAL1'], hdr['CRVAL2']

                ref_fits = i_f

            else:
                crpix1, crpix2, crval1, crval2 = reduction.xcor_cr12(
                    i_f,
                    ref_fits,
                    ra=ra,
                    dec=dec,
                    crpix=crpix,
                    background_subtraction=background_sub,
                    plot=int(plot)*2,
                    box_size=box
                )

            istring = "\t\t{0}: {1:.2f}, {2:.1f}, {3:.4f}, {4:.4f}\n".format(
                cdict["ID_LIST"][i], crpix1, crpix2, crval1, crval2)
            utils.output(istring)


        elif xymode == 'none':

            crpix1 = hdr["CRPIX1"]
            crpix2 = hdr["CRPIX2"]
            crval1 = hdr["CRVAL1"]
            crval2 = hdr["CRVAL2"]

        else:
            raise ValueError("xymode can only be 'none', 'src_fit', or 'xcor'")


        outstr += ">%19s %15.7f %15.7f %10.3f %10.1f %10.1f %10.1f\n" % (
            cdict["ID_LIST"][i], crval1, crval2, crval3s[i], crpix1, crpix2, crpix3s[i])

    if out is None:
        outfilename = clist.replace(".list", ".wcs")
    else:
        outfilename = out

    #Create the correction file
    outfile = open(outfilename, 'w')
    outfile.write(outstr)
    outfile.close()

    utils.output("\n\tSaved corrections table to %s\n" % outfilename)
    config.restore_output_mode()

#Call using dict and argument parser if run from command-line
if __name__ == "__main__":

    arg_parser = parser_init()
    args = arg_parser.parse_args()

    main(**vars(args))
