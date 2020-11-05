"""Subtract background signal from a data cube"""

#Standard Imports
import argparse
import os

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import extraction, utils, config

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Subtract background signal from a data cube"""
    )
    parser.add_argument(
        'cube',
        type=str,
        help='Individual cube or cube type to be subtracted.',
        default=None
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
    parser.add_argument(
        '-method',
        type=str,
        help='Which method to use for subtraction. Polynomial fit or median filter.\
        (\'medfilt\' or \'polyFit\')',
        choices=['medfilt', 'polyfit', 'noiseFit', 'median'],
        default='medfilt'
        )
    parser.add_argument(
        '-poly_k',
        metavar='<poly_degree>',
        type=int,
        help='Degree of polynomial (if using polynomial sutbraction method).',
        default=1
        )
    parser.add_argument(
        '-med_window',
        metavar='<window_size_px>',
        type=int,
        help='Size of median window (if using median filtering method).',
        default=31
        )
    parser.add_argument(
        '-wmask',
        metavar='<w0:w1>',
        type=str,
        nargs='+',
        help='Wavelength range(s) to mask before processing. Specify each as a\
        tuple of the form A:B',
        default=None
        )
    parser.add_argument(
        '-mask_neb_z',
        metavar='<redshift>',
        type=float,
        help='Prove redshift to auto-mask nebular emission.',
        default=None
        )
    parser.add_argument(
        '-mask_neb_dv',
        metavar='<km/s>',
        type=float,
        help='Velocity width (km/s) around nebular lines to mask, if using -mask_neb.',
        default=500
        )
    parser.add_argument(
        '-mask_sky',
        action='store_true',
        help='Prove redshift to auto-mask nebular emission.'
        )
    parser.add_argument(
        '-mask_sky_dw',
        metavar='<Angstrom>',
        type=float,
        help='FWHM to use when masking sky lines. Default is automatically\
        determined based on instrument configuration.',
        default=None
        )
    parser.add_argument(
        '-mask_reg',
        metavar="<.reg>",
        type=str,
        help="Region file of areas to exclude when using median subraction.",
        default=None
        )
    parser.add_argument(
        '-save_model',
        help='Set flag to output background model cube (.bg.fits)',
        action='store_true'
        )
    parser.add_argument(
        '-outdir',
        metavar='<file_ext>',
        type=str,
        help='The directory to save cropped files to. Default is same directory as input data.'
        )
    parser.add_argument(
        '-ext',
        metavar='<file_ext>',
        type=str,
        help='Extension to append to input cube for output cube (.bs.fits)',
        default='.bs.fits'
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

def bg_sub(cube, clist=None, var=None, method='polyfit', poly_k=3, med_window=31,
           wmask=None, mask_neb_z=None, mask_neb_dv=None, mask_sky=False, mask_sky_dw=None,
           mask_reg=None, save_model=False, ext=".bs.fits", outdir=None, log=None, silent=None):
    """Subtract background signal from a data cube

    Args:
        cube (str): Path to the input data (FITS file) or a CWI cube type
            (e.g. 'icubes.fits') if using a CWITools .list file.
        clist (str): Path to CWITools list file, for acting on multiple cubes.
        var (str): Path to variance data FITS file or CWI cube type for variance
            data (e.g. 'vcubes.fits'), if using CWITools .list file.
        method (str): Which method to use to model background
            'polyfit': Fits polynomial to the spectrum in each spaxel (default.)
            'median': Subtract the spatial median of each wavelength layer.
            'medfilt': Model spectrum in each spaxel by median filtering it.
            'noiseFit': Model noise in each z-layer and subtract mean.
        poly_k (int): The degree of polynomial to use for background modeling.
        med_window (int): The filter window size to use if median filtering.
        wmask (list): List of wavelength ranges to exclude from model-fitting,
            provided as a list of float-like tuples e.g. [(4100,4200), (5100,5200)]
        mask_neb_z (float): Redshift of nebular emission to auto-mask.
        mask_neb_dv (float): Velocity width, in km/s, of nebular emission masks.
        mask_sky (bool): Set to TRUE to auto-mask sky emission lines.
        mask_sky_dw (float): Width of sky-line masks to use, in Angstroms.
        mask_reg (str): Path to a DS9 region file to use to exclude regions
            when using 'median' method of bg subtraction.
        save_model (bool): Set to TRUE to save a FITS containing the bg model.
        ext (str): File extension to use for masked FITS (".M.fits")
        outdir (str): Output directory for files. Default is the same directory as input.
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """
    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("BG_SUB", locals())


    #Load from list and type if list is given
    if clist is not None:

        cdict = utils.parse_cubelist(clist)
        file_list = utils.find_files(
            cdict["ID_LIST"],
            cdict["DATA_DIRECTORY"],
            cube,
            cdict["SEARCH_DEPTH"]
        )

        if var is not None:
            var_file_list = utils.find_files(
                cdict["ID_LIST"],
                cdict["DATA_DIRECTORY"],
                var,
                cdict["SEARCH_DEPTH"]
            )

    #Load individual cube if that is given instead
    else:
        if os.path.isfile(cube):
            file_list = [cube]
        else:
            raise FileNotFoundError(cube)

        if var is not None:
            if os.path.isfile(var):
                var_file_list = [var]
            else:
                raise FileNotFoundError(var)


    if wmask is None:
        wmask = []

    #Run through files to be BG-subtracted
    for i, filename in enumerate(file_list):

        fits_file = fits.open(filename)

        if var is not None:
            var_cube, var_header = fits.getdata(var_file_list[i], header=True)
        else:
            var_cube = None

        if mask_neb_z is not None:
            utils.output("\n\tAuto-masking Nebular Emission Lines\n")
            wmask += utils.get_nebmask(
                fits_file[0].header,
                redshift=mask_neb_z,
                vel_window=mask_neb_dv,
                mode='tuples'
            )
        if mask_sky:
            wmask += utils.get_skymask(
                fits_file[0].header,
                linewidth=mask_sky_dw,
                mode='tuples'
            )


        #Run background subtraction
        res = extraction.bg_sub(
            fits_file,
            method=method,
            poly_k=poly_k,
            median_window=med_window,
            wmasks=wmask,
            mask_reg=mask_reg,
            var=var_cube
        )

        if var is None:
            subtracted_cube, bg_model = res
        else:
            subtracted_cube, bg_model, var_out = res

        if outdir is None:
            file_out = filename.replace('.fits', ext)
        else:
            outdir = os.path.abspath(outdir)
            file_out = outdir + '/' + os.path.basename(filename).replace('.fits', ext)

        sub_fits = utils.match_hdu_type(fits_file, subtracted_cube, fits_file[0].header)
        sub_fits.writeto(file_out, overwrite=True)
        utils.output("\tSaved %s\n" % file_out)

        if save_model:
            model_file_out = file_out.replace('.fits', '.bg_model.fits')
            model_fits = utils.match_hdu_type(fits_file, bg_model, fits_file[0].header)
            model_fits.writeto(model_file_out, overwrite=True)
            utils.output("\tSaved %s\n" % model_file_out)

        if var is not None:
            var_file_out = file_out.replace('.fits', '.var.fits')
            var_fits_out = utils.match_hdu_type(fits_file, var_out, var_header)
            var_fits_out.writeto(var_file_out, overwrite=True)
            utils.output("\tSaved %s\n" % var_file_out)

    config.restore_output_mode()


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()

    #Parse wmask argument properly into list of float-tuples
    if isinstance(args.wmask, list):
        try:
            for j, wpair in enumerate(args.wmask):
                args.wmask[j] = tuple(float(x) for x in wpair.split(':'))
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)

    bg_sub(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
