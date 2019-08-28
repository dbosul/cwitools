from cwitools import params
from cwitools.reduction import trim

import argparse

def main():

    #Handle input with argparse
    parser = argparse.ArgumentParser(description='Crop input cubes according to a CWITools parameter file.')
    parser.add_argument('paramPath',
                        type=str,
                        help='Path to CWITools parameter file.'
    )
    parser.add_argument('cubeType',
                        type=str,
                        help='The PCWI/KCWI cube type to load for each image ID. (e.g. "icubes.fits" or "ocubes.fits") Must end in .fits file extension.'
    )
    parser.add_argument('wcrop',
                        type=str,
                        help="Wavelength range, in Angstrom, to crop to (syntax 'w0,w1') (Default:0,-1)",
                        default='0,-1'
    )
    parser.add_argument('xcrop',
                        type=str,
                        help="Subrange of x-axis to crop to (syntax 'x0,x1') (Default:0,-1)"",
                        default='0,-1'
    )
    parser.add_argument('ycrop',
                        type=str,
                        help="Subrange of y-axis to crop to (syntax 'y0,y1') (Default:0,-1)"",
                        default='0,-1'
    )
    parser.add_argument('-ext',
                        type=str,
                        help='The filename extension to add to cropped cubes. Default: .c.fits'
    )
    args = parser.parse_args()

    try: x0,x1 = ( int(x) for x in args.xcrop.split(','))
    except:
        print("Could not parse -xcrop, should be comma-separated integer tuple.")
        sys.exit()

    try: y0,y1 = ( int(y) for y in args.ycrop.split(','))
    except:
        print("Could not parse -ycrop, should be comma-separated integer tuple.")
        sys.exit()

    try: w0,w1 = ( int(y) for w in args.wcrop.split(','))
    except:
        print("Could not parse -wcrop, should be comma-separated integer tuple.")
        sys.exit()

    # Check if any parameter values are missing (set to set-up mode if so)
    params = params.loadparams(paramPath)

    # Get filenames
    fileList = params.findfiles(params,cubeType)

    # Open fits objects
    fitsList = [ fits.open(x) for x in fileList ]

    # Pass to trimming function
    trimmed_fitsList = trim(fitsList,fileExt=args.ext,xcrop=(x0,x1),ycrop=(y0,y1),wcrop=(w0,w1))

    for i,trimmedFits in enumerate(trimmed_fitsList):
        outFileName = fileList[i].replace('.fits',fileExt)
        f.writeto(outFileName,overwrite=True)
        print("Saved %s"%outFileName)

if __name__=="__main__": main()
