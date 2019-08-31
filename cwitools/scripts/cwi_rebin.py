from astropy.io import fits
from cwitools.analysis import rebin

from astropy.io import fits
import argparse
import os
import warnings

def main():

    #Handle user input with argparse
    parser = argparse.ArgumentParser(description='Re-bin cubes by integer amounts along spatial (XY) and/or wavelength (Z) axes.')
    parser.add_argument('cube',
                        type=str,
                        help='Input cube to be binned.'
    )
    parser.add_argument('-xybin',
                        type=int,
                        help='Number of pixels to bin in X,Y axes'
    )
    parser.add_argument('-zbin',
                        type=int,
                        help='Number of pixels to bin in Z axis.'

    )
    parser.add_argument('-ext',
                        type=str,
                        help='File extension to add for binned cube (Default: .binned.fits)',
                        default=".binned.fits"
    )
    parser.add_argument('-vardata',
                        type=bool,
                        help='Set to True when binning variance data. Coefficients are squared.'
    )
    args = parser.parse_args()

    args.vardata = (args.vardata.upper() in ['T','TRUE'])

    #Load data
    if os.path.isfile(args.cube): inFits = fits.open(args.cube)
    else:
        raise FileNotFoundError("Input file not found.\nFile:%s"%args.cube)

    #Check that user has actually set the bin options
    if args.zbin==1 and args.xybin==1:
        warnings.warn("Binning 1x1x1 won't change anything! Set the bin sizes with the flags -zBin and -xyBin.")

    binnedFits = rebin(inFits,xybin=args.xybin,zbin=args.zbin,vardata=args.vardata)

    outFileName = args.cube.replace(".fits",args.ext)
    binnedFits.writeto(outFileName,overwrite=True)
    print("Saved %s"%outFileName)


if __name__ == "__main__": main()
