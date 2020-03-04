"""Create a binary mask using a DS9 region file."""
from astropy.io import fits
from cwitools import coordinates, imaging

import argparse
import os
import sys
import warnings

def main():

    parser = argparse.ArgumentParser(description="Apply a binary mask to data of the same dimensions.")
    parser.add_argument('reg',
                        type=str,
                        help='DS9 region file to convert into a mask.'
    )
    parser.add_argument('data',
                        type=str,
                        help='Data cube or image to create mask for.'
    )
    parser.add_argument('-fit',
                        help='Set flag to fit 2D Gaussians to sources',
                        action='store_true'
    )
    parser.add_argument('-fit_box',
                        type=float,
                        help='Size of box (in px) to use for fitting sources',
                        default=10
    )
    parser.add_argument('-width',
                        type=float,
                        help='Width of each source mask.',
                        default=3
    )
    parser.add_argument('-width_unit',
                        type=str,
                        help='Units of width argument (px, arcsec or sigma.)',
                        choices=['px', 'arcsec', 'sigma'],
                        default='sigma'
    )
    args = parser.parse_args()

    if os.path.isfile(args.data):
        data, header = fits.getdata(args.data, header=True)
    else:
        raise FileNotFoundError(args.data)

    #Create 2D WL image and header if 3D data given
    if max(data.shape) == 2:
        data = np.mean(data, axis=0)
        header = coordinates.get_header2d(header)

    #Get mask
    mask = imaging.get_mask(data, header, args.reg,
        fit=args.fit,
        fit_box=args.fit_box,
        mask_width=args.width,
        width_unit=args.width_unit

    )

    outFileName = args.data.replace('.fits', args.ext)
    maskedFits = fits.HDUList([fits.PrimaryHDU(mask)])
    maskedFits[0].header = header.copy()
    maskedFits.writeto(outFileName,overwrite=True)

    print("Saved %s" % outFileName)


if __name__=="__main__": main()
