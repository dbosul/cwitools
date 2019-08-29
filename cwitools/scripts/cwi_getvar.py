from cwitools.analysis import estimate_variance
from cwitools.cubes import make_fits
import argparse

def main():
    #Take any additional input params, if provided
    parser = argparse.ArgumentParser(description='Get estimated variance cube.')
    parser.add_argument('cube',
                        type=str,
                        metavar='path',
                        help='Input cube whose 3D variance you would like to estimate.'
    )
    parser.add_argument('-zWindow',
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
    parser.add_argument('-fMin',
                        type=float,
                        metavar='float',
                        help='Minimum rescaling factor (default 0.9)',
                        default=0.9
    )
    parser.add_argument('-fMax',
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
    try: zmask = tuple(int(x) for x in zmask.split(','))
    except:
        raise ValuError("Could not parse zmask argument")

    vardata = estimate_variance(fitsFile,
        zWindow=args.zWindow,
        rescale=args.rescale,
        sigmaclip=args.sigmaclip,
        zmask=zmask,
        fMin=args.fMin,
        fMax=args.fMax,
    )

    varPath = cubePath.replace('.fits',fileExt)
    varFits = make_fits(vardata,fitsFile[0].header)
    varFits.writeto(varPath,overwrite=True)
    print("Saved %s"%varPath)

if __name__=="__main__": main()
