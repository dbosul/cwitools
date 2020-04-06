"""Generate a pseudo-Narrowband image"""
from astropy.io import fits
from cwitools import utils, coordinates
import argparse
import numpy as np
import sys
import time

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Make channel maps of an input cube around a specified emission line.')
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
        'id',
        type=int,
        help='The input object ID.'
    )
    parser.add_argument(
        '-ext',
        type=str,
        help='Output extension.',
        default=".sb.fits"
    )
    args = parser.parse_args()

    int_cube, hdr3d = fits.getdata(args.cube, header=True)
    obj_cube = fits.getdata(args.obj)

    pixel_size_as = coordinates.get_pxsize_arcsec(hdr3d)
    pixel_size_ang = hdr3d["CD3_3"]

    int_cube[obj_cube != args.id] = 0
    int_img = np.sum(int_cube, axis=0)

    int_img *= pixel_size_ang
    int_img /= pixel_size_as

    hdr2d = coordinates.get_header2d(hdr3d)
    out_fits = utils.get_fits(int_img, hdr2d)

    out_filename = args.cube.replace(".fits", args.ext)
    out_fits.writeto(out_filename, overwrite=True)
    print("Saved %s" % out_filename)

if __name__=="__main__": main()
