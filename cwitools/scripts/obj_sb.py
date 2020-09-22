"""Generate a surface brightness map of a 3D object."""

#Standard Imports
import argparse

#Third-party Imports
from astropy.io import fits

#Local Imports
from cwitools import utils, config, synthesis

def parser_init():
    """Create command-line argument parser for this script."""
    parser = argparse.ArgumentParser(
        description="""Generate a surface brightness map of a 3D object."""
    )
    parser.add_argument(
        'cube',
        type=str,
        help='The input data cube FITS file.'
    )
    parser.add_argument(
        'obj',
        type=str,
        help='The input object mask cube FITS file.'
    )
    parser.add_argument(
        'obj_id',
        type=int,
        nargs='+',
        help='The input object ID or IDs (space-separated).'
    )
    parser.add_argument(
        '-redshift',
        type=float,
        help='Redshift of the emission - provide if you want to apply (1+z)^4 correction to SB.',
    )
    parser.add_argument(
        '-var',
        type=str,
        help='Variance cube FITS file.',
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
        help="Log file to save output in."
    )
    parser.add_argument(
        '-silent',
        help="Set flag to suppress standard terminal output.",
        action='store_true'
    )
    return parser

def obj_sb(cube, obj, obj_id, var=None, redshift=None, label=None, log=None, silent=None):
    """Generate a surface brightness map of a 3D object.

    Args:
        cube (str): Path to input data cube
        obj (str): Path to FITS containing 3D object masks.
        obj_id (int or list): ID (or list of IDs) of object(s) to include when
            creating SB map.
        redshift (float): Redshift of the emission - provide to apply (1+z)^4 factor for
            cosmological surface brightness dimming correction.
        label (str): Custom label for output file name, which will add .<label>_sb.fits to the
            input file name. e.g. provide "LyA" for a Lyman-alpha object to get ".LyA_sb.fits"
            By default, the label will "objXX" where XX is the objID for a single ID, or the first
            ID followed by a '+' for a list of IDs.
        log (str): Path to log file to save output to.
        silent (bool): Set to TRUE to suppress standard output.

    Returns:
        None
    """

    config.set_temp_output_mode(log, silent)
    utils.output_func_summary("OBJ_SB", locals())

    int_fits = fits.open(cube)
    obj_cube = fits.getdata(obj)
    var_cube = None if var is None else fits.getdata(var)

    res = synthesis.obj_sb(int_fits, obj_cube, obj_id, var_cube=var_cube, redshift=redshift)

    #Unpack depending on whether var was given or not
    sb_fits, sb_var_fits = (res, None) if var is None else res

    if label is None:
        if isinstance(obj_id, int):
            label = "obj%02i" % obj_id
        elif isinstance(obj_id, list):
            label = "obj%02i+" % obj_id[0]

    out = cube.replace(".fits", ".%s_sb.fits" % label)

    sb_fits.writeto(out, overwrite=True)
    utils.output("\tSaved %s\n" % out)

    if sb_var_fits is not None:
        out_var = out.replace('.fits', '.var.fits')
        sb_var_fits.writeto(out_var, overwrite=True)
        utils.output("\tSaved %s\n" % out_var)

    config.restore_output_mode()


def main():
    """Entry-point method for setup tools"""
    arg_parser = parser_init()
    args = arg_parser.parse_args()
    obj_sb(**vars(args))

#Call if run from command-line
if __name__ == "__main__":
    main()
