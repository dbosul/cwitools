from astropy.io import fits
from cwitools import parameters

import argparse
import numpy as np
import os
import warnings


def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Apply a WCS corrections file to data.')
    parser.add_argument('wcscor',
                        type=str,
                        help='WCS correction file (see cwi_measurewcs.py)',
    )
    parser.add_argument('cubetype',
                        type=str,
                        help='Type of file to apply corrections to (e.g. icubes.fits)',
    )
    parser.add_argument('-ext',
                        type=str,
                        help='File extension for corrected files (Def: .wc.fits)',
                        default=".wc.fits"
    )
    args = parser.parse_args()

    try:
        wcs_cor_file = open(args.wcscor)

    except FileNotFoundError:
        print("Could not find WCS correction file: %s" % args.wcscor)
        exit()

    except:
        print("Something went wrong opening WCS correction file.")
        exit()

    ids = []
    cr_matrix = []
    in_dir = "."
    search_depth = 3

    for i, line in enumerate(wcs_cor_file):

        line = line.replace("\n", "")

        if "INPUT_DIRECTORY" in line:
            in_dir = line.split("=")[1].replace(" ", "")

        elif "SEARCH_DEPTH" in line:
            search_depth = int(line.split("=")[1])

        elif line[0] == ">":
            vals = line[1:].split()
            ids.append(vals[0])
            cr_cols = [float(x) for x in vals[1:]]
            cr_matrix.append(cr_cols)

        else:
            continue

    cr_matrix = np.array(cr_matrix)

    input_files = parameters.find_files(
        ids,
        in_dir,
        args.cubetype,
        depth=search_depth
    )
    print("\nCorrecting WCS Axes based on %s" % args.wcscor)
    print("-"*70)
    print("%30s %10s %10s %10s" % ("New Filename", "Ax1Cor?", "Ax2Cor?", "Ax3Cor?"))
    for i, filename in enumerate(input_files):

        in_fits = fits.open(filename)
        ax1, ax2, ax3 = "No", "No", "No"

        if 0 <= cr_matrix[i, 0] <= 360:
            in_fits[0].header["CRVAL1"] = cr_matrix[i, 0]
            in_fits[0].header["CRPIX1"] = cr_matrix[i, 3]
            ax1 = "Yes"

        else:
            warnings.warn("Invalid RA / CRVAL1. Must be 0-360 deg.")

        if -90 <= cr_matrix[i, 1] <= 90:
            in_fits[0].header["CRVAL2"] = cr_matrix[i, 1]
            in_fits[0].header["CRPIX2"] = cr_matrix[i, 4]
            ax2 = "Yes"

        else:
            warnings.warn("Invalid DEC / CRVAL2. Must be -90 to +90 deg.")

        if cr_matrix[i, 2] > 0:
            in_fits[0].header["CRVAL3"] = cr_matrix[i, 2]
            in_fits[0].header["CRPIX3"] = cr_matrix[i, 5]
            ax3 = "Yes"


        outfilename = filename.replace(".fits", args.ext)
        in_fits.writeto(outfilename, overwrite=True)
        outfilename_short = outfilename.split("/")[-1]
        print("%30s %10s %10s %10s" % (outfilename_short, ax1, ax2, ax3))

    print("-"*70)
    print("Done. New files saved in input directories.")
if __name__=="__main__":
    main()
