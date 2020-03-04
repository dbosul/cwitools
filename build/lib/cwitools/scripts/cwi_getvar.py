from cwitools.variance import estimate_variance
from astropy.io import fits

import argparse
import os

def main():
    #Take any additional input params, if provided
    parser = argparse.ArgumentParser(description='Get estimated variance cube.')
    parser.add_argument('cube',
                        type=str,
                        metavar='path',
                        help='Input cube whose 3D variance you would like to estimate.'
    )
    parser.add_argument('-zwindow',
                        type=int,
                        metavar='int (px)',
                        help='Algorithm chops cube into z-bins and estimates 2D variance map at each bin by calculating it along z-axis. This parameter controls that bin size.',
                        default=10
    )
    parser.add_argument('-zmask',
                        type=str,
                        metavar='int tuple (px)',
                        help='Pair of z-indices (e.g. 21,29) to ignore (i.e. interpolate over) when calculating variance.',
                        default="0,0"
    )
    parser.add_argument('-fmin',
                        type=float,
                        metavar='float',
                        help='Minimum rescaling factor (default 0.9)',
                        default=0.9
    )
    parser.add_argument('-out',
                        type=str,
                        metavar='str',
                        help='Filename for output. Default is input + .var.fits',
                        default=None
    )
    args = parser.parse_args()

    #Try to load the fits file
    if os.path.isfile(args.cube): fits_in = fits.open(args.cube)
    else:
        raise FileNotFoundError("Input file not found.")

    #Try to parse the wavelength mask tuple
    try: zmask = tuple(int(x) for x in args.zmask.split(','))
    except:
        raise ValueError("Could not parse zmask argument")

    vardata = estimate_variance(fits_in,
        zwindow=args.zwindow,
        zmask=zmask,
        fmin=args.fmin
    )

    if args.out == None:
        outfilename = args.cube.replace('.fits', '.var.fits')
    else:
        outfilename = args.out

    var_fits = fits.HDUList([fits.PrimaryHDU(vardata)])
    var_fits[0].header = fits_in[0].header
    var_fits.writeto(outfilename,overwrite=True)
    print("Saved %s" % outfilename)

if __name__=="__main__": main()
