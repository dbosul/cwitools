"""Generate a surface brightness map of a 3D object."""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits
import numpy as np

#Local Imports
from cwitools import utils, coordinates, config

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Generate a surface brightness map of a 3D object."""
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
        '-ext',
        type=str,
        help='Output extension.',
        default=".sb.fits"
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

def main(cube, obj, obj_id, ext=".sb.fits", log=None, silent=None):
    """Generate a surface brightness map of a 3D object.

    Args:
        cube (str): Path to input data cube
        obj (str): Path to FITS containing 3D object masks.
        obj_id (int or list): ID (or list of IDs) of object(s) to include when
            creating SB map.
        ext (str): File extension for SB map output. 
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("OBJ_SB", locals())

    int_fits = fits.open(cube)
    int_cube, hdr3d = int_fits[0].data, int_fits[0].header
    obj_cube = fits.getdata(obj)

    pixel_size_as = coordinates.get_pxarea_arcsec(hdr3d)
    pixel_size_ang = hdr3d["CD3_3"]

    for o_id in obj_id:
        obj_cube[obj_cube == o_id] = -99

    int_cube[obj_cube != -99] = 0
    int_img = np.sum(int_cube, axis=0)

    int_img *= pixel_size_ang
    int_img /= pixel_size_as

    hdr2d = coordinates.get_header2d(hdr3d)
    out_fits = utils.matchHDUType(int_fits, int_img, hdr2d)

    out_filename = cube.replace(".fits", ext)
    out_fits.writeto(out_filename, overwrite=True)
    utils.output("\tSaved %s\n" % out_filename, silent=silent, log=log)

    config.restore_output_mode()

#Call using dict and argument parser if run from command-line
if __name__ == "__main__":

    arg_parser = parser_init()
    args = arg_parser.parse_args()

    main(**vars(args))
