"""Segment cube into 3D regions above a threshold."""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import utils, extraction
import cwitools

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

def main(cube, var, snr_int=None, snr_min=3.0, n_min=10, include=None, exclude=None,
         include_neb_z=None, include_neb_dv=None, exclude_sky=False, exclude_sky_dw=None,
         fill_holes=False, ext=".obj.fits", silent=True, log=None):
    """Segment cube into 3D regions above a threshold."""

    #Set global parameters
    cwitools.silent_mode = silent
    cwitools.log_file = log

    #Give output for log file
    utils.output_func_summary("SEGMENT", locals())

    fits_in = fits.open(cube)
    var_cube = fits.getdata(var)

    #Try to parse the wavelength mask tuple
    if include is None:
        includes_all = []
    if exclude is None:
        excludes_all = []

    #Add nebular emission includes
    if include_neb_z is not None:
        includes_all += utils.get_nebmask(
            fits_in[0].header,
            z=include_neb_z,
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


#Call using dict and argument parser if run from command-line
if __name__ == "__main__":

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

    main(**vars(args))
