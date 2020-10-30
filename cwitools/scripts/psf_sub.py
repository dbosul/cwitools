"""Subtract point sources from 3D data."""
#Standard Imports
import argparse
import os

#Third-party Imports
from astropy.io import fits
from astropy.wcs import WCS

#Local Imports
from cwitools import extraction, utils, config
from cwitools.coordinates import get_header2d

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
        metavar='<x.xx y.yy>',
        type=float,
        nargs=2,
        help='Position of one source <x y> to subtract.'
    )
    src_group.add_argument(
        '-radec',
        metavar='<dd.dd dd.dd>',
        type=float,
        nargs=2,
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
        nargs='+',
        help='Wavelength range(s) to mask, presented as space-separated A:B pairs.'
    )
    parser.add_argument(
        '-mask_neb_z',
        metavar='<redshift>',
        type=float,
        help='Prove redshift to auto-mask nebular emission.'
    )
    parser.add_argument(
        '-mask_neb_dv',
        metavar='<km/s>',
        type=float,
        help='Velocity width (km/s) around nebular lines to mask, if using -mask_neb_z.',
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
        '-outdir',
        metavar='<file_ext>',
        type=str,
        help='The directory to save cropped files to. Default is same directory as input data.'
        )
    parser.add_argument(
        '-recenter',
        help='Set flag to adjust input source coordinates to line up with source center.',
        action='store_true'
    )
    parser.add_argument(
        '-save_psf',
        help='Set flag to output PSF Cube)',
        action='store_true'
    )
    parser.add_argument(
        '-mask_psf',
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

def psf_sub(cube, clist=None, var=None, xy=None, radec=None, reg=None, auto=7,
            r_fit=1, r_sub=15, wl_window=150, wmask=None, mask_neb_z=None,
            mask_neb_dv=500, recenter=False, save_psf=False, mask_psf=False,
            ext=".ps.fits", outdir=None, log=None, silent=None):
    """Subtract point sources from 3D data.

    Generate a surface brightness map of a 3D object.

    Args:
        cube (str): Path to the input data (FITS file) or a CWI cube type
            (e.g. 'icubes.fits') if using a CWITools .list file.
        clist (str): Path to CWITools list file, for acting on multiple cubes.
        var (str): Path to variance data FITS file or CWI cube type for variance
            data (e.g. 'vcubes.fits'), if using CWITools .list file.
        xy (float tuple): Image coordinates of source to be subtracted
        radec (float tuple): RA/DEC coordinates of source to be subtracted
        reg (str): Path to DS9 region file of sources to subtract
        auto (float): For automatic-PSF subtraction, the SNR threshold for
            source detection.
        r_fit (float): Inner radius, in arcsec, used for fitting PSF.
        r_sub (float): Outer radius, in arcsec, used to subtract PSF.
        wl_window (int): Size of white-light window (in Angstrom) to use.
            This is the window used to form a white-light image centered
            on each wavelength layer. Default: 200A.
        wmask (list): List of wavelength ranges to mask, given as a list of
            float tuples in units of Angstroms. e.g. [(4100,4200), (5100,5200)]
        mask_neb_z (float): Redshift of nebular emission to auto-mask.
        mask_neb_dv (float): Velocity width, in km/s, of nebular emission masks.
        recenter (bool): Recenter the input (x, y) using the centroid within a
            box of size recenter_box, arcseconds.
        save_psf (bool): Set to TRUE to save a FITS containing the PSF model
        mask_psf (bool): Set to TRUE to mask the PSF region used to scale the
            white-light images.
        ext (str): File extension for output files. (".ps.fits")
        outdir (str): Output directory for files. Default is the same directory as input.
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("PSF_SUB", locals())

    use_var = (var is not None)

    #Make sure output directory exists before we start
    if outdir is not None:
        if not os.path.isdir(outdir):
            raise NotADirectoryError(outdir)
        outdir = os.path.abspath(outdir)

    #Load from list and type if list is given
    if clist is not None:

        cdict = utils.parse_cubelist(clist)
        file_list = utils.find_files(
            cdict["ID_LIST"],
            cdict["DATA_DIRECTORY"],
            cube,
            cdict["SEARCH_DEPTH"]
        )

        if use_var:
            var_file_list = utils.find_files(
                cdict["ID_LIST"],
                cdict["DATA_DIRECTORY"],
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

    if mask_neb_z is not None:
        utils.output("\n\tAuto-masking Nebular Emission Lines\n")

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

        if wmask is None:
            wmask = []

        if mask_neb_z is not None:
            wmask += utils.get_nebmask(
                fits_in[0].header,
                redshift=mask_neb_z,
                vel_window=mask_neb_dv,
                mode='tuples'
            )


        res = extraction.psf_sub_all(
            fits_in,
            pos=pos,
            reg=reg,
            auto=auto,
            r_fit=r_fit,
            r_sub=r_sub,
            wl_window=wl_window,
            wmasks=wmask,
            var_cube=var_cube,
            maskpsf=mask_psf,
            recenter=recenter
        )

        if use_var:
            sub_cube, psf_model, var_cube = res
        else:
            sub_cube, psf_model = res

        if outdir is None:
            file_out = file_in.replace('.fits', ext)
        else:
            outdir = os.path.abspath(outdir)
            file_out = outdir + '/' + os.path.basename(file_in).replace('.fits', ext)

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

    config.set_temp_output_mode(log, silent)


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()

    #Parse wmask argument properly into list of float-tuples
    if isinstance(args.wmask, list):
        try:
            for i, wpair in enumerate(args.wmask):
                args.wmask[i] = tuple(float(x) for x in wpair.split(':'))
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)

    psf_sub(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
