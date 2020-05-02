"""Estimate the 3D variance for a data cube"""
from cwitools.reduction import estimate_variance
from cwitools import utils
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
    parser.add_argument('-window',
                        type=int,
                        help='Size of wavelength bin, in Angstrom, for 2D layer variance estimate.',
                        default=50
    )
    parser.add_argument('-wmask',
                        type=str,
                        metavar='Wav Mask',
                        help='Wavelength range(s) to mask when fitting',
                        default=None
    )
    parser.add_argument('-sclip',
                        type=int,
                        help='Sigma-clip to apply calculating rescaling factors',
                        default=4
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
    parser.add_argument('-log',
                        type=str,
                        help="Log file to save this command in",
                        default=None
    )
    args = parser.parse_args()

    #Try to load the fits file
    if os.path.isfile(args.cube): fits_in = fits.open(args.cube)
    else:
        raise FileNotFoundError("Input file not found.")

    #Try to parse the wavelength mask tuple
    wmasks = []
    if args.wmask != None:
        try:
            w0,w1 = tuple(int(x) for x in args.wmask.split(':'))
            wmasks.append([w0,w1])
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)



    vardata = estimate_variance(fits_in,
        window=args.window,
        wmasks=wmasks,
        fmin=args.fmin,
        sclip=args.sclip
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
