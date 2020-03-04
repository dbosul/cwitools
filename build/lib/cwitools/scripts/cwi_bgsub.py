from cwitools.subtraction import bg_sub
from astropy.io import fits
import argparse
import os

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Perform background subtraction on a data cube.')

    mainGroup = parser.add_argument_group(title="Main",description="Basic input")
    mainGroup.add_argument('cube',
                        type=str,
                        metavar='cube',
                        help='The cube to be subtracted.'
    )

    methodGroup = parser.add_argument_group(title="Methods",description="Parameters related to BKG Subtraction methods.")
    methodGroup.add_argument('-method',
                        type=str,
                        metavar='Method',
                        help='Which method to use for subtraction. Polynomial fit or median filter. (\'medfilt\' or \'polyFit\')',
                        choices=['medfilt','polyfit','noiseFit','median'],
                        default='medfilt'
    )
    methodGroup.add_argument('-k',
                        type=int,
                        metavar='Polynomial Degree',
                        help='Degree of polynomial (if using polynomial sutbraction method).',
                        default=1
    )
    methodGroup.add_argument('-window',
                        type=int,
                        metavar='MedFilt Window',
                        help='Size of median window (if using median filtering method).',
                        default=31
    )
    methodGroup.add_argument('-zmask',
                        type=str,
                        metavar='Wav Mask',
                        help='Z-indices to mask when fitting or median filtering (e.g. \'21,32\' or \'4140,4170\')',
                        default='0:0'
    )
    methodGroup.add_argument('-zunit',
                        type=str,
                        metavar='Wav Mask',
                        help='Unit of input for zmask. Can be Angstrom (A) or Pixels (px) (Default: A)',
                        default='A',
                        choices=['A','px']
    )
    fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")
    fileIOGroup.add_argument('-savemodel',
                        type=str,
                        metavar='Save BG Model',
                        help='Set to True to output background model cube (.bg.fits)',
                        choices = ["True","False"],
                        default = "False"
    )
    fileIOGroup.add_argument('-ext',
                        type=str,
                        metavar='File Extension',
                        help='Extension to append to input cube for output cube (.bs.fits)',
                        default='.bs.fits'
    )
    args = parser.parse_args()

    #Parse arg.save from str to bool
    savemodel = True if args.savemodel=="True" else False

    #Try to load the fits file
    if os.path.isfile(args.cube): fitsFile = fits.open(args.cube)
    else: raise FileNotFoundError(args.cube)

    #Try to parse the wavelength mask tuple
    masks = [(0,0)]
    try:
        for pair in args.zmask.split('-'):
            print(pair)
            z0,z1 = tuple(int(x) for x in pair.split(':'))
            masks.append((z0,z1))
    except: raise ValueError("Could not parse zmask argument.")


    subtracted_cube, bg_model = bg_sub(  fitsFile,
                            method=args.method,
                            poly_k=args.k,
                            median_window=args.window,
                            zmasks=masks,
                            zunit=args.zunit
    )

    outFileName = args.cube.replace('.fits',args.ext)

    subtracted_Fits = fits.HDUList([fits.PrimaryHDU(subtracted_cube)])
    subtracted_Fits[0].header  = fitsFile[0].header
    subtracted_Fits.writeto(outFileName,overwrite=True)
    print("Saved %s" % outFileName)

    if savemodel:
        outFileName2 = outFileName.replace('.fits','.bg_model.fits')
        model_Fits = fits.HDUList([fits.PrimaryHDU(bg_model)])
        model_Fits[0].header  = fitsFile[0].header
        model_Fits.writeto(outFileName2,overwrite=True)
        print("Saved %s" % outFileName2)

if __name__=="__main__": main()
