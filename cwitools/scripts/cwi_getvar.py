from cwitools.analysis import estimate_variance
from cwitools.libs.cubes import make_fits

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
    parser.add_argument('-rescale',
                        type=str,
                        metavar='bool',
                        help="Whether or not to rescale each wavelength layer to normalize variance to sigma=1 in that layer.",
                        choices=["True","False"],
                        default="True"
    )
    parser.add_argument('-sigmaclip',
                        type=float,
                        metavar='float',
                        help="Sigma-clip threshold in stddevs to apply before estimating variance. Set to 0 to skip sigma-clipping (default: 4)",
                        default=4.0
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
    parser.add_argument('-fmax',
                        type=float,
                        metavar='float',
                        help='Maximum rescaling factor (default 10)',
                        default=10
    )
    parser.add_argument('-ext',
                        type=str,
                        metavar='str',
                        help='Extension to add to output file (default .var.fits)',
                        default=".var.fits"
    )
    args = parser.parse_args()

    #Try to load the fits file
    if os.path.isfile(args.cube): fitsFile = fits.open(args.cube)
    else:
        raise FileNotFoundError("Input file not found.")

    #Try to parse the wavelength mask tuple
    try: zmask = tuple(int(x) for x in args.zmask.split(','))
    except:
        raise ValueError("Could not parse zmask argument")

    vardata = estimate_variance(fitsFile,
        zwindow=args.zwindow,
        rescale=args.rescale,
        sigmaclip=args.sigmaclip,
        zmask=zmask,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    varPath = args.cube.replace('.fits',args.ext)
    varFits = make_fits(vardata,fitsFile[0].header)
    varFits.writeto(varPath,overwrite=True)
    print("Saved %s"%varPath)

if __name__=="__main__": main()
