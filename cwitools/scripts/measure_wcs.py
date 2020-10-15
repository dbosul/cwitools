"""Measure WCS: Create a WCS correction table by measuring the input data."""

#Standard Imports
import argparse
import os
import sys
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
        default="none",
        choices=["src_fit", "xcor", "none"]
        )
    parser.add_argument(
        '-radec',
        metavar="<dd.ddd>",
        type=float,
        nargs=2,
        help="Right-ascension and declination (decimal degrees, space-separated) of source to fit."
        )
    parser.add_argument(
        '-box',
        metavar="<box_size>",
        type=float,
        help="Box size (arcsec) for fitting source or cross-correlating."
        )
    parser.add_argument(
        '-crpix1s',
        type=float,
        metavar="<list of floats>",
        nargs='+',
        help="List of CRPIX1. Use 'H' to use existing header value."
        )
    parser.add_argument(
        '-crpix2s',
        metavar="<list of floats>",
        type=float,
        nargs='+',
        help="List of CRPIX2. Use 'H' to use existing header value."
        )
    parser.add_argument(
        '-background_sub',
        help="Subtract background before xcor.",
        default=False
        )
    parser.add_argument(
        '-zmode',
        help='Which method to use for correcting z-azis',
        default='none',
        choices=['none', "fit", "xcor"]
        )
    parser.add_argument(
        '-crval3',
        metavar="<float>",
        type=float,
        help="Wavelength (Angstrom) of the sky-line to fit, if using '-zmode fit'"
        )
    parser.add_argument(
        '-zwindow',
        type=float,
        metavar="<float>",
        help="Window size [Angstrom] to use when fitting sky-line, if using '-zmode fit'",
        default=20
        )
    parser.add_argument(
        '-sky_type',
        type=str,
        metavar="<float>",
        help="The type of cube to load for the sky spectrum, if using 'xcor' or 'fit' for zmode.\
        Default is to replace 'icube' with 'scube' for the main input type"
        )
    parser.add_argument(
        '-plot',
        help="Display fits with Matplotlib.",
        action='store_true'
        )
    parser.add_argument(
        '-out',
        metavar="",
        help="Output table name."
        )
    parser.add_argument(
        '-log',
        metavar="<log_file>",
        type=str,
        help="Log file to save output in."
        )
    parser.add_argument(
        '-silent',
        help="Set flag to suppress standard terminal output.",
        action='store_true'
        )

    return parser

def measure_wcs(clist, ctype="icubes.fits", xymode='none', radec=None, box=10.0,
                crpix1s=None, crpix2s=None, background_sub=False, zmode='none', crval3=None,
                zwindow=20, sky_type=None, plot=False, out=None, log=None, silent=None):
    """Automatically create a WCS correction table for a list of input cubes.

    Args:
        clist (str): Path to the CWITools .list file
        ctype (str): File extension for type of cube to use as input.
            Default value: 'icubes.fits' (.fits extension should be included)
        xymode (str): Method to use for XY alignment:
            'src_fit': Fit 1D profiles to a known point source (interactive)
            'xcor': Perform 2D (XY) cross-correlation of input images.
            'none': Do not align the spatial axes.
        radec (float tuple): Tuple of (RA, DEC) of source in decimal degrees,
            if using 'src_fit'
        box (float): Size of box (in arcsec) to use for finding/fitting source,
            if using 'src_fit'
        crpix1s (list): List of CRPIX1 values to serve as initial estimates of
            spatial alignment, if using xymode=xcor
        crpix2s (list): List of CRPIX2 values, for the same reason as crpix2s.
        background_sub (bool): Set to TRUE to subtract background before
            cross-correlating spatially.
        zmode (str): Method to use for Z alignment:
            'fit': Fit a 1D Gaussian to a known emission line at 'crval3'
            'xcor': Cross-correlate the spectra and provide relative alignment
            'none': Do not align z-axes
        crval3 (float): The central wavelength [Angstrom] of the fittable sky-line, if using
            zmode='fit'
        zwindow (float): If using zmode='fit', the window-size [Angstrom] to use when fitting
            the sky emission line. Default is 20A (i.e. +/- 10A)
        sky_type (str): The type of cube to load for the sky spectrum (e.g. scubes.fits)
        plot (bool): Set to TRUE to show diagnostic plots.
        out (str): File extension to use for masked FITS (".M.fits")
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """


    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("MEASURE_WCS", locals())

    #Parse cube list
    cdict = utils.parse_cubelist(clist)

    #Load input files
    in_files = utils.find_files(
        cdict["ID_LIST"],
        cdict["DATA_DIRECTORY"],
        ctype,
        depth=cdict["SEARCH_DEPTH"]
    )

    #Load scube (or ocube) files
    int_fits = [fits.open(x) for x in in_files]

    #Prepare table output
    outstr = "DATA_DIRECTORY=%s\n" % cdict["DATA_DIRECTORY"]
    outstr += "SEARCH_DEPTH=%i\n" % cdict["SEARCH_DEPTH"]
    outstr += "#%19s %15s %15s %10s %10s %10s %10s\n" % (
        "ID", "CRVAL1", "CRVAL2", "CRVAL3", "CRPIX1", "CRPIX2", "CRPIX3")

    if zmode == "none" and xymode == "none":
        utils.output("ERROR: At least one of 'zmode' or 'xymode' must be set!\n")
        sys.exit()

    if zmode in ["fit", "xcor"]:
        if sky_type is not None:
            sky_files = utils.find_files(
                cdict["ID_LIST"],
                cdict["DATA_DIRECTORY"],
                sky_type,
                depth=cdict["SEARCH_DEPTH"]
            )
            if len(sky_files) != len(in_files):
                utils.output("Sky files not found for all input files. Exiting.\n")
                sys.exit()

        else:
            sky_files = []
            for i_file in in_files:
                if os.path.isfile(i_file.replace("icube", "scube")):
                    sky_files.append(i_file.replace("icube", "scube"))
                elif os.path.isfile(i_file.replace("icube", "ocube")):
                    sky_files.append(i_file.replace("icube", "ocube"))
                elif zmode == 'xcor':
                    utils.output("WARNING: No sky cubes found for z-axis correction. Using input\
                    cube type (%s).\n" % ctype)
                    sky_files.append(i_file)
                else:
                    utils.output("ERROR: No scube/ocube found for %s, cannot use zmode='fit'.\
 Please fix and try again, or provide 'sky_type' argument.\n" % i_file)
                    sys.exit()

    #WAVELENGTH ALIGNMENT - XCOR
    if zmode == "fit":

        if crval3 is None:
            raise ValueError("crval3 must be provided if using zmode='fit'")

        #Try to load sky files for z-axis cross-correlation. If it fails, use input cubes.
        crpix3s = []
        crval3s = []

        for i, s_f in enumerate(sky_files):
            sky_fits = fits.open(s_f)
            crpix3_fit = reduction.wcs.fit_crpix3(sky_fits, crval3, window=zwindow, plot=plot)
            if crpix3_fit == -1:
                utils.output("WARNING: Sky-line fit failed for %s. WCS not updated." % sky_files[i])
                crpix3s.append(s_f[0].header["CRPIX3"])
                crval3s.append(s_f[0].header["CRVAL3"])
            else:
                crpix3s.append(crpix3_fit)
                crval3s.append(crval3)

    elif zmode == "xcor":

        #Try to load sky files for z-axis cross-correlation. If it fails, use input cubes.
        sky_fits = [fits.open(x) for x in sky_files]

        utils.output("\tAligning z-axes...\n")
        crval3s = [i_f[0].header["CRVAL3"] for i_f in int_fits]
        crpix3s = reduction.wcs.xcor_crpix3(sky_fits)

    elif zmode == "none":
        warnings.warn("No wavelength WCS correction applied.")
        crval3s = [i_f[0].header["CRVAL3"] for i_f in int_fits]
        crpix3s = [i_f[0].header["CRPIX3"] for i_f in int_fits]

    else:
        raise ValueError("zmode can only be 'none' or 'xcor'")

    #Basic checks and user-message for each xy-mode
    if xymode == "src_fit":
        if radec is None:
            raise ValueError("'radec' must be set if using src_fit.")
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
            crval1, crval2 = radec[0], radec[1]
            if crpix1s is not None and crpix1s[i] != 'H'\
            and crpix2s is not None and crpix2s[i] != 'H':
                crpix_guess = (crpix1s[i], crpix2s[i])
            else:
                crpix_guess = None
            crpix1, crpix2 = reduction.wcs.fit_crpix12(
                i_f, crval1, crval2,
                plot=plot,
                box_size=box,
                crpix12_guess=crpix_guess
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

                if radec is not None:
                    wcs0 = WCS(hdr)
                    tmp = wcs0.all_world2pix(radec[0], radec[1], 1, 1)
                    crpix1, crpix2 = float(tmp[0]), float(tmp[1])
                    crval1, crval2 = radec

                else:
                    crpix1, crpix2 = hdr['CRPIX1'], hdr['CRPIX2']
                    crval1, crval2 = hdr['CRVAL1'], hdr['CRVAL2']

                ref_fits = i_f

            else:
                crpix1, crpix2, crval1, crval2 = reduction.wcs.xcor_crpix12(
                    i_f,
                    ref_fits,
                    ra=radec[0],
                    dec=radec[1],
                    crpix=crpix,
                    bg_subtraction=background_sub,
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


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    measure_wcs(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
