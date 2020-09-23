"""Generate an integrated spectrum of a 3D object."""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import utils, config, synthesis

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Generate an integrated spectrum of a 3D object."""
    )
    parser.add_argument(
        'cube',
        type=str,
        help='The input data cube.'
    )
    parser.add_argument(
        'obj',
        type=str,
        help='The input object cube.'
    )
    parser.add_argument(
        'obj_id',
        type=int,
        nargs='+',
        help='The input object ID or IDs (space-separated).'
    )
    parser.add_argument(
        '-var',
        type=str,
        help='The input variance estimate cube.'
    )
    parser.add_argument(
        '-extend_z',
        help="Set to TRUE to include full spectrum range from each object spaxel.",
        action='store_true'
    )
    parser.add_argument(
        '-no_covar',
        help="Set to TRUE to prevent variance rescaling for covariance.",
        action='store_true'
    )
    parser.add_argument(
        '-label',
        type=str,
        help='Label for output file (e.g. "LyA" or "HeII"). Default is "objXX", where XX is obj_id'
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

def obj_spec(cube, obj, obj_id, extend_z=False, var=None, no_covar=False, label=None, log=None,
             silent=None):
    """Generate an integrated spectrum of a 3D object.

    Args:
        cube (str): Path to input data cube
        obj (str): Path to FITS containing 3D object masks.
        obj_id (int or list): ID (or list of IDs) of object(s) to include when
            creating SB map.
        extend_z (bool): Set to TRUE to include full spectrum range from each object spaxel. By
            default, only the spectral range within the object mask will be summed. Default: False.
        rescale_cov (bool): Rescale the variance estimate to account for covariance. Only works when
            covariance calibration has been done (see scripts.fit_covar). Default: True.
        label (str): Custom label for output file name, which will add .<label>_spec.fits to the
            input file name. e.g. provide "LyA" for a Lyman-alpha object to get ".LyA_spec.fits"
            By default, the label will "objXX" where XX is the objID for a single ID, or the first
            ID followed by a '+' for a list of IDs.
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("OBJ_SPEC", locals())

    int_fits = fits.open(cube)
    obj_cube = fits.getdata(obj)
    var_cube = None if var is None else fits.getdata(var)

    spec_fits = synthesis.obj_spec(
        int_fits, obj_cube, obj_id,
        var_cube=var_cube,
        extend_z=extend_z,
        rescale_cov=(not no_covar)
        )


    if label is None:
        if isinstance(obj_id, int):
            label = "obj%02i" % obj_id
        elif isinstance(obj_id, list):
            label = "obj%02i+" % obj_id[0]

    out = cube.replace(".fits", ".%s_spec.fits" % label)

    spec_fits.writeto(out, overwrite=True)
    utils.output("\tSaved %s\n" % out)

    config.restore_output_mode()


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    obj_spec(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
