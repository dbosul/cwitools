"""Generate a pseudo-Narrowband image"""
from astropy.io import fits
from cwitools import utils, coordinates
import argparse
import numpy as np
import sys
import time

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Convert KCWI data to MUSE format.')
    parser.add_argument(
        'flux_cube',
        type=str,
        help='The input data cube.'
    )
    parser.add_argument(
        'var_cube',
        type=str,
        help='The input object cube.'
    )
    parser.add_argument(
        'out',
        type=str,
        help='The input object ID.'
    )
    args = parser.parse_args()

    int_in = fits.open(args.flux_cube)
    var_in = fits.open(args.var_cube)

    new_hdulist = fits.HDUList()
    new_hdulist.append(int_in[0])
    new_hdulist.append(var_in[0])
    new_hdulist.writeto(args.out, overwrite=True)
    print("Saved %s." % args.out)
    
if __name__ == "__main__": main(TBD, arg_parser=parser_init())
