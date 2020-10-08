"""Segment cube into 3D regions above a threshold."""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import utils, extraction, config

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Segment cube into 3D regions above a threshold."""
    )
    parser.add_argument(
        'cube',
        type=str,
        help='The input data cube.'
    )
    parser.add_argument(
        'var',
        type=str,
        help='Variance cube. Estimated if not provided.'
    )
    parser.add_argument(
        '-snr_int',
        type=float,
        help='Integrated SNR threshold. Takes priority over nmin if both provided.'
    )
    parser.add_argument(
        '-snr_min',
        type=float,
        help='The SNR threshold to use.',
        default=3.0
    )
    parser.add_argument(
        '-n_min',
        type=int,
        help='Minimum region size, in voxels.',
        default=10
    )
    parser.add_argument(
        '-include',
        type=str,
        nargs='+',
        help="Space-separated list of wavelength ranges to focus on. Each range\
        should be specified in the form A:B, in units of Angstrom. (A:B C:D etc.)"
    )
    parser.add_argument(
        '-exclude',
        type=str,
        nargs='+',
        help="List of wavelength ranges to exclude. Same format as -include arg."
    )
    parser.add_argument(
        '-include_neb_z',
        metavar='<redshift>',
        type=float,
        help='Prove redshift to auto-include common nebular emission.'
    )
    parser.add_argument(
        '-include_neb_dv',
        metavar='<km/s>',
        type=float,
        help='Velocity width (km/s) around nebular lines to include,\
        if using -include_neb_z.',
        default=2000
    )
    parser.add_argument(
        '-exclude_sky',
        action='store_true',
        help='Automatically exclude bright sky lines from segmentation.'
    )
    parser.add_argument(
        '-exclude_sky_dw',
        metavar='<Angstrom>',
        type=float,
        help='FWHM to use when excluding sky lines. Default is estimated based\
        on instrument configuration if not provided.'
    )
    parser.add_argument(
        '-fill_holes',
        help='Set to TRUE to auto-repair 3D objects by filling holes using \
        scipy.ndimage.morphology.binary_fill_holes',
        action='store_true'
    )
    parser.add_argument(
        '-ext',
        type=str,
        help="Output filename. Default, input cube with .obj.fits",
        default=".obj.fits"
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

def segment(cube, var, snr_int=None, snr_min=3.0, n_min=10, include=None, exclude=None,
            include_neb_z=None, include_neb_dv=None, exclude_sky=False, exclude_sky_dw=None,
            fill_holes=False, ext=".obj.fits", log=None, silent=None):
    """Segment cube into 3D regions above a threshold.

    Args:
        cube (str): Path to the input data FITS
        var (str): Path to the input variance FITS
        snr_int (float): Integrated SNR threshold, use instead of nmin to base
            selection on the total SNR instead of size.
        snr_min (float): Voxel-by-voxel threshold to apply
        n_min (int): The minimum size of a 3D object above snr_min, in voxels.
        include (list): List of float tuples indicating which wavelength ranges
            to include, in units of Angstrom. e.g. [(4100,4200), (4350,4400)]
        exclude (list): List of tuples indicating which wavelength ranges to
            exclude from segmentation process. Same format as 'include'
        include_neb_z (float): Redshift of nebular emission to auto-include
        include_neb_dv (float): Velocity width, in km/s, of target nebular emission.
        exclude_sky (bool): Set to TRUE to auto-mask sky emission lines.
        exclude_sky_dw (float): Width of sky-line masks to use, in Angstroms.
        fill_holes (bool): Set to TRUE to auto-fill holes in 3D objects using
            scipy.ndimage.morphology.binary_fill_holes.
        ext (str): File extension for output file
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.


    Returns:
        None

    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("SEGMENT", locals())

    fits_in = fits.open(cube)
    var_cube = fits.getdata(var)

    #Try to parse the wavelength mask tuple
    includes_all = []
    excludes_all = []
    if include is not None:
        includes_all += include
    if exclude is not None:
        excludes_all += exclude

    #Add nebular emission includes
    if include_neb_z is not None:
        includes_all += utils.get_nebmask(
            fits_in[0].header,
            redshift=include_neb_z,
            vel_window=include_neb_dv,
            mode='tuples'
        )

    #Add sky line excludes
    if exclude_sky is not None:
        excludes_all += utils.get_skymask(
            fits_in[0].header,
            linewidth=exclude_sky_dw,
            mode='tuples'
        )

    if len(includes_all) == 0:
        includes_all = None
    if len(excludes_all) == 0:
        excludes_all = None

    obj_fits = extraction.segment(
        fits_in,
        var_cube,
        snrmin=snr_min,
        nmin=n_min,
        includes=includes_all,
        excludes=excludes_all,
        fill_holes=fill_holes,
        snr_int=snr_int
    )

    out_file = cube.replace(".fits", ext)
    obj_fits.writeto(out_file, overwrite=True)
    utils.output("\tSaved %s\n" % out_file)
    config.set_temp_output_mode(log, silent)


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()

    #Handle any arguments that need extra parsing
    if args.include is not None:
        try:
            for i, pair in enumerate(args.include):
                args.include[i] = tuple(float(x) for x in pair.split(':'))
        except:
            raise ValueError("Could not parse include argument (%s)." % args.include)

    if args.exclude is not None:
        try:
            for i, pair in enumerate(args.exclude):
                args.exclude[i] = tuple(float(x) for x in pair.split(':'))
        except:
            raise ValueError("Could not parse exclude argument (%s)." % args.exclude)

    segment(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
