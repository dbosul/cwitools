from cwitools.reduction import wav_crop
import argparse


def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Use RA/DEC and Wavelength reference points to adjust WCS.')


    parser.add_argument('cube',
                        type=str,
                        metavar='path',
                        help='Input cube to crop.)'
    )
    parser.add_argument('wavPair',
                        type=str,
                        metavar='float tuple',
                        help='Wavelength range (in angstrom) to crop to (e.g. 4160,4180)'
    )
    parser.add_argument('-ext',
                        type=str,
                        metavar='str',
                        help='Extension to add to cropped cube filename (default: .wcrop.fits)',
                        default=".wcrop.fits"
    )
    args = parser.parse_args()

    #Try to load the fits file
    try: fitsFile = fits.open(cubePath)
    except: print("Error: could not open '%s'\nExiting."%args.cube);sys.exit()

    #Try to parse wavelength tuple
    try: w0,w1 = (float(x) for x in wavPair.split(','))
    except:
        print("Could not parse wavelengths from input. Please check syntax (should be comma-separated tuple of floats representing upper/lower bound in wavelength for cropped cube.")
        sys.exit();

    croppedFits = wav_crop(fitsFile,w0,w1)

    #Get output name and save
    outFile = cubePath.replace('.fits',fileExt)
    croppedFits.writeto(outFile,overwrite=True)
    print("Saved %s."%outFile)

if __name__=="__main__": main()
