"""Apply Mask: Apply a binary mask FITS image to data."""
from astropy.io import fits
import argparse
import os
import sys
import warnings

def main():

    parser = argparse.ArgumentParser(description="Apply a binary mask to data of the same dimensions.")
    parser.add_argument('mask',
                        type=str,
                        help='Binary mask to be applied.'
    )
    parser.add_argument('data',
                        type=str,
                        help='Data to be masked.'
    )
    parser.add_argument('-fill',
                        type=float,
                        help='Value used to mask data (Default: 0)',
                        default=0
    )
    parser.add_argument('-out',
                        type=str,
                        help="Output file name. Default is to add .M.fits to input data.",
                        default=None
    )
    args = parser.parse_args()

    if os.path.isfile(args.mask): mask = fits.getdata(args.mask)
    else: raise FileNotFoundError(args.mask)

    if os.path.isfile(args.data): data,header = fits.getdata(args.data,header=True)
    else: raise FileNotFoundError(args.data)

    data_masked = data.copy()

    if data.shape == mask.shape: data_masked[ mask==1 ] = args.fill

    elif mask.shape == data[0].shape:

        for zi in range(data.shape[0]):
            data_masked[zi][mask==1] = args.fill

    else:
        raise RuntimeError("Mask should be 2D (spatial) or 3D (full cube) with matching dimensions")

    if args.out == None:
        outfilename = args.data.replace('.fits', '.M.fits')
    else:
        outfilename = args.out

    maskedFits = fits.HDUList([fits.PrimaryHDU(data_masked)])
    maskedFits[0].header = header.copy()
    maskedFits.writeto(outfilename,overwrite=True)

    print("Saved %s"%outfilename)


if __name__=="__main__": main()
