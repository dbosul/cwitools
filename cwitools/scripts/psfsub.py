"""Subtract point sources from 3D data."""
from astropy.io import fits
from astropy.wcs import WCS
from cwitools import extraction, utils
from cwitools.coordinates import get_header2d
from datetime import datetime

import argparse
import cwitools
import numpy as np
import os
import sys

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Perform PSF subtraction on a data cube.')
    parser.add_argument('cube',
                        type=str,
                        help='Individual cube or cube type to be subtracted.',
                        default=None
    )
    parser.add_argument('-list',
                        type=str,
                        metavar='<cube_list>',
                        help='CWITools cube list'
    )
    parser.add_argument('-var',
                        metavar='<var_cube/type>',
                        type=str,
                        help="Variance cube or variance cube type."
    )
    srcGroup = parser.add_mutually_exclusive_group(required=False)
    srcGroup.add_argument('-xy',
                        metavar='<x.xx,y.yy>',
                        type=str,
                        help='Position of one source (x, y) to subtract.',
                        default=None
    )
    srcGroup.add_argument('-radec',
                        metavar='<dd.dd,dd.dd>',
                        type=str,
                        help='Position of one source (ra, dec) to subtract.',
                        default=None
    )
    srcGroup.add_argument('-reg',
                        metavar='<DS9RegFile>',
                        type=str,
                        help='DS9 region file of sources to subtract.',
                        default=None
    )
    srcGroup.add_argument('-auto',
                        metavar='<SNR_Thresh>',
                        type=float,
                        help='SNR threshold for automatic source detection',
                        default=7
    )
    parser.add_argument('-method',
                        type=str,
                        help="2D PSF fitting or slice-by-slice 1D fitting",
                        choices=['1d', '2d'],
                        default='2d'
    )
    parser.add_argument('-rfit',
                        type=float,
                        metavar='<arcsec>',
                        help='Radius (arcsec) used to fit the PSF (default 1)',
                        default=1
    )
    parser.add_argument('-rsub',
                        type=float,
                        metavar='<arcsec>',
                        help='Radius (arcsec) of subtraction area (default 15).',
                        default=15
    )
    parser.add_argument('-wlwindow',
                        type=int,
                        metavar='<Angstrom>',
                        help='Window (angstrom) used to create WL image of PSF (default 150).',
                        default=150
    )
    parser.add_argument('-wmask',
                        metavar='<w0:w1,w2:w3,...>',
                        type=str,
                        help='Wavelength range(s) to mask when fitting',
                        default=None
    )
    parser.add_argument('-mask_neb',
                        metavar='<redshift>',
                        type=float,
                        help='Prove redshift to auto-mask nebular emission.',
                        default=None
    )
    parser.add_argument('-vwidth',
                        metavar='<km/s>',
                        type=float,
                        help='Velocity width (km/s) around nebular lines to mask, if using -mask_neb.',
                        default=None
    )
    parser.add_argument('-slice_axis',
                        type=int,
                        help='Axis in which each pixel is a slice (KCWI=2, PCWI=1). Defaults to 2.',
                        choices=[1,2],
                        default=2
    )
    parser.add_argument('-ext',
                        metavar="<file_ext>",
                        type=str,
                        help='Extension to append to subtracted cube (.ps.fits)',
                        default='.ps.fits'
    )
    parser.add_argument('-savepsf',
                        help='Set flag to output PSF Cube)',
                        action='store_true'
    )
    parser.add_argument('-maskpsf',
                        help='Set flag to spaxels used for fitting.',
                        action='store_true'
    )
    parser.add_argument('-v', help="Verbose: display progress and info.",action="store_true")
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
    args = parser.parse_args()

    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_PSFSUB:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(parser)
    utils.output(titlestring + infostring)

    #Load from list and type if list is given
    if args.list != None:

        clist = utils.parse_cubelist(args.list)
        file_list =  utils.find_files(
            clist["ID_LIST"],
            clist["INPUT_DIRECTORY"],
            args.cube,
            clist["SEARCH_DEPTH"]
        )

    #Load list of individual cubes if that is given instead
    else:
        if os.path.isfile(args.cube):
            file_list = [args.cube]
        else:
            raise FileNotFoundError(args.cube)

    #By default, assume we are propagating variance
    usevar = True
    #If var is a file
    if os.path.isfile(args.var):
        var_file_list = [args.var]

    #If not a file and not None - assume it is a cube type
    elif args.var != None:

        var_file_list =  utils.find_files(
            clist["ID_LIST"],
            clist["INPUT_DIRECTORY"],
            args.var,
            clist["SEARCH_DEPTH"]
        )
    #If none of the above, don't use var
    else:
        usevar = False
        var_file_list = []

    #Try to parse the wavelength mask tuple
    custom_masks = []
    neb_masks = []
    if args.wmask != None:
        try:
            for pair in args.wmask.split('-'):
                w0,w1 = tuple(int(x) for x in pair.split(':'))
                custom_masks.append((w0,w1))
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)

    for i, file_in in enumerate(file_list):

        fits_in = fits.open(file_in)

        if usevar:
            var_cube, var_header = fits.getdata(var_file_list[i], header=True)
        else:
            var_cube = []

        header2d = get_header2d(fits_in[0].header)
        wcs2d = WCS(header2d)

        #Take position of source as either x,y pair...
        if args.xy != None:
            pos = tuple(int(x) for x in args.xy.split(','))

        #Or ra,dec pair and convert to x,y
        elif args.radec !=None:
            ra, dec = tuple(float(x) for x in args.radec.split(','))
            pos = wcs2d.all_world2pix(ra, dec, 0)
            pos = tuple(int(round(float(x))) for x in pos)[::-1]

        else:
            pos = None

        if args.mask_neb is not None:
            utils.output("\n\tAuto-masking Nebular Emission Lines\n")
            neb_masks = utils.get_nebmask(fits_in[0].header,
                z = args.mask_neb,
                vel_window = args.vwidth,
                mode = 'tuples'
            )    

        masks = custom_masks + neb_masks

        #Get subtracted cube and psf model
        res  = extraction.psf_sub_all(fits_in,
            pos=pos,
            reg=args.reg,
            auto=args.auto,
            fit_rad = args.rfit,
            sub_rad = args.rsub,
            wl_window = args.wlwindow,
            slice_axis = args.slice_axis,
            method = args.method,
            wmasks = masks,
            var_cube = var_cube,
            maskpsf = args.maskpsf
        )
        if usevar:
            sub_cube, psf_model, var_cube = res
        else:
            sub_cube, psf_model = res

        file_out = file_in.replace('.fits', args.ext)
        out_fits = fits.HDUList([fits.PrimaryHDU(sub_cube)])
        out_fits[0].header = fits_in[0].header
        out_fits.writeto(file_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(file_out))

        if args.savepsf:
            psf_out  = file_out.replace('.fits','.psf_model.fits')
            psf_fits = fits.HDUList([fits.PrimaryHDU(psf_model)])
            psf_fits[0].header = fits_in[0].header
            psf_fits.writeto(psf_out, overwrite=True)
            utils.output("\tSaved {0}\n".format(psf_out))

        if usevar:
            var_out  = file_out.replace('.fits','.var.fits')
            var_fits = fits.HDUList([fits.PrimaryHDU(var_cube)])
            var_fits[0].header = var_header
            var_fits.writeto(var_out, overwrite=True)
            utils.output("\tSaved {0}\n".format(var_out))

if __name__=="__main__":
    main()
