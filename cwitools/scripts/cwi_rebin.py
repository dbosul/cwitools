from astropy.io import fits
from cwitools.analysis import rebin

import argparse

def main():

    #Handle user input with argparse
    parser = argparse.ArgumentParser(description='Re-bin cubes by integer amounts along spatial (XY) and/or wavelength (Z) axes.')
    parser.add_argument('cube',
                        type=str,
                        help='Input cube to be binned.'
    )
    parser.add_argument('-xyBin',
                        type=int,
                        help='Number of pixels to bin in X,Y axes'
    )
    parser.add_argument('-zBin',
                        type=int,
                        help='Number of pixels to bin in Z axis.'

    )
    parser.add_argument('-fileExt',
                        type=str,
                        help='File extension to add for binned cube (Default: .binned.fits)',
                        default=".binned.fits"
    )
    parser.add_argument('-varData',
                        type=bool,
                        help='Set to True when binning variance data. Coefficients are squared.'
    )
    args = parser.parse_args()

    args.varData = (args.varData.upper() in ['T','TRUE'])

    #Load data
    try: inFits = fits.open(args.cube)
    except:
        print("Could not open input cube. Check path and try again. (Path: %s)"%args.cube)
        sys.exit()

    #Check that user has actually set the bin options
    if zBin==1 and xyBin==1:
        print("Binning 1x1x1 won't change anything! Set the bin sizes with the flags -zBin and -xyBin.")
        sys.exit()

    binnedFits = rebin(inFits,xyBin=args.xyBin,zBin=args.zBin,varData=args.varData)

    outFileName = args.cube.replace(".fits",args.fileExt)
    binnedFits.writeto(outFileName,overwrite=True)
    print("Saved %s"%outFileName)


if __name__ == "__main__": main()
