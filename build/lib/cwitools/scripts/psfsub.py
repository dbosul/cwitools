"""Subtract point sources from 3D data."""
#Standard Imports
import argparse
import os

#Third-party Imports
from astropy.io import fits
from astropy.wcs import WCS

#Local Imports
from cwitools import extraction, utils
from cwitools.coordinates import get_header2d
import cwitools

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Subtract point sources from 3D data."""
    )
    parser.add_argument(
        'cube',
        type=str,
        help='Individual cube or cube type to be subtracted.'
    )
    parser.add_argument(
        '-clist',
        type=str,
        metavar='<cube_list>',
        help='CWITools cube list'
    )
    parser.add_argument(
        '-var',
        metavar='<var_cube/type>',
        type=str,
        help="Variance cube or variance cube type."
    )
    src_group = parser.add_mutually_exclusive_group(required=False)
    src_group.add_argument(
        '-xy',
        metavar='<x.xx,y.yy>',
        type=str,
        help='Position of one source (x, y) to subtract.'
    )
    src_group.add_argument(
        '-radec',
        metavar='<dd.dd,dd.dd>',
        type=str,
        help='Position of one source (ra, dec) to subtract.'
    )
    src_group.add_argument(
        '-reg',
        metavar='<DS9RegFile>',
        type=str,
        help='DS9 region file of sources to subtract.'
    )
    src_group.add_argument(
        '-auto',
        metavar='<SNR_Thresh>',
        type=float,
        help='SNR threshold for automatic source detection',
        default=7
    )
    parser.add_argument(
        '-r_fit',
        type=float,
        metavar='<arcsec>',
        help='Radius (arcsec) used to fit the PSF (default 1)',
        default=1
    )
    parser.add_argument(
        '-r_sub',
        type=float,
        metavar='<arcsec>',
        help='Radius (arcsec) of subtraction area (default 15).',
        default=15
    )
    parser.add_argument(
        '-wl_window',
        type=int,
        metavar='<Angstrom>',
        help='Window (angstrom) used to create WL image of PSF (default 150).',
        default=150
    )
    parser.add_argument(
        '-wmask',
        metavar='<w0:w1 w2:w3 ...>',
        type=str,
        help='Wavelength range(s) to mask, presented as space-separated A:B pairs.'
    )
    parser.add_argument(
        '-mask_neb',
        metavar='<redshift>',
        type=float,
        help='Prove redshift to auto-mask nebular emission.'
    )
    parser.add_argument(
        '-vwidth',
        metavar='<km/s>',
        type=float,
        help='Velocity width (km/s) around nebular lines to mask, if using -mask_neb.',
        default=500
    )
    parser.add_argument(
        '-ext',
        metavar="<file_ext>",
        type=str,
        help='Extension to append to subtracted cube (.ps.fits)',
        default='.ps.fits'
    )
    parser.add_argument(
        '-recenter',
        help='Set flag to adjust input source coordinates to line up with source center.',
        action='store_true'
    )
    parser.add_argument(
        '-savepsf',
        help='Set flag to output PSF Cube)',
        action='store_true'
    )
    parser.add_argument(
        '-maskpsf',
        help='Set flag to spaxels used for fitting.',
        action='store_true'
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

def main(cube, clist=None, var=None, xy=None, radec=None, reg=None, auto=7,
         r_fit=1, r_sub=15, wl_window=150, wmask=None, mask_neb_z=None,
         mask_neb_dv=500, ext=".ps.fits", recenter=False, save_psf=False,
         mask_psf=False, log=None, silent=True):
    """Subtract point sources from 3D data."""

    cwitools.silent_mode = silent
    cwitools.log_file = log

    utils.output_func_summary("PSF_SUB", locals())

    use_var = (var is not None)

    #Load from list and type if list is given
    if list is not None:

        cdict = utils.parse_cubelist(clist)
        file_list = utils.find_files(
            cdict["ID_LIST"],
            cdict["INPUT_DIRECTORY"],
            cube,
            cdict["SEARCH_DEPTH"]
        )

        if use_var:
            var_file_list = utils.find_files(
                cdict["ID_LIST"],
                cdict["INPUT_DIRECTORY"],
                var,
                cdict["SEARCH_DEPTH"]
            )

    #Load list of individual cubes if that is given instead
    else:
        if os.path.isfile(cube):
            file_list = [cube]
        else:
            raise FileNotFoundError(cube)

        if use_var:
            if os.path.isfile(var):
                var_file_list = [var]
            else:
                raise FileNotFoundError(var)

    for i, file_in in enumerate(file_list):

        fits_in = fits.open(file_in)

        if use_var:
            var_cube, var_header = fits.getdata(var_file_list[i], header=True)
        else:
            var_cube = []

        header2d = get_header2d(fits_in[0].header)
        wcs2d = WCS(header2d)

        #Take position of source as either x,y pair...
        if xy is not None:
            pos = xy
        #Or ra,dec pair and convert to x,y
        elif radec is not None:
            pos = wcs2d.all_world2pix(radec[0], radec[1], 0)
        else:
            pos = None

        if mask_neb_z is not None:
            utils.output("\n\tAuto-masking Nebular Emission Lines\n")
            neb_masks = utils.get_nebmask(
                fits_in[0].header,
                z=mask_neb_z,
                vel_window=mask_neb_dv,
                mode='tuples'
            )

        res = extraction.psf_sub_all(
            fits_in,
            pos=pos,
            reg=reg,
            auto=auto,
            fit_rad=r_fit,
            sub_rad=r_sub,
            wl_window=wl_window,
            wmasks=wmask + neb_masks,
            var_cube=var_cube,
            maskpsf=mask_psf,
            recenter=recenter
        )

        if use_var:
            sub_cube, psf_model, var_cube = res
        else:
            sub_cube, psf_model = res

        file_out = file_in.replace('.fits', ext)
        out_fits = fits.HDUList([fits.PrimaryHDU(sub_cube)])
        out_fits[0].header = fits_in[0].header
        out_fits.writeto(file_out, overwrite=True)
        utils.output("\tSaved {0}\n".format(file_out))

        if save_psf:
            psf_out = file_out.replace('.fits', '.psf_model.fits')
            psf_fits = fits.HDUList([fits.PrimaryHDU(psf_model)])
            psf_fits[0].header = fits_in[0].header
            psf_fits.writeto(psf_out, overwrite=True)
            utils.output("\tSaved {0}\n".format(psf_out))

        if use_var:
            var_out = file_out.replace('.fits', '.var.fits')
            var_fits = fits.HDUList([fits.PrimaryHDU(var_cube)])
            var_fits[0].header = var_header
            var_fits.writeto(var_out, overwrite=True)
            utils.output("\tSaved {0}\n".format(var_out))


#Call using dict and argument parser if run from command-line
if __name__ == "__main__":

    arg_parser = parser_init()
    args = arg_parser.parse_args()

    #Parse wmask argument properly into list of float-tuples
    if isinstance(args.wmask, list):
        try:
            for i, wpair in enumerate(args.wmask):
                args.wmask[i] = tuple(float(x) for x in wpair.split(':'))
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)

    main(**vars(args))
