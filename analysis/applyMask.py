from astropy.io import fits
import argparse
import sys

def run(maskPath,dataPath,fileExt=".M.fits",fillValue=0):

    try: mskFITS = fits.open(maskPath)
    except:
        print("Could not load mask. Check path and try again.\nPath:%s"%maskPath)
        sys.exit()

    try: inpFITS = fits.open(dataPath)
    except:
        print("Could not load data. Check path and try again.\nPath:%s"%dataPath)
        sys.exit()

    inpFITS[0].data[mskFits[0].data==0] = fillValue

    outFileName = dataPath.replace('.fits',fileExt)

    inpFITS.writeto(outFileName,overwrite=True)

    print("Saved %s"%outFileName)

if __name__="__main__":

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
    parser.add_argument('-ext',
                        type=str,
                        help="File extension to be used for masked data. Default: .M.fits",
                        default=".M.fits"
    )
    args = parser.parse_args()
    
    run(args.mask,args.data,fillValue=args.fill,fileExt=args.ext)
