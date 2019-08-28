from .. imports libs

from astropy.io import fits
import argparse
import sys

def run(fitsList, xcrop=None, ycrop=None, wcrop=None):
    """Trims axes of each input cube according to CWITools parameter file.

    Args:
        paramPath (str): Path to CWITools parameter file.
        cubeType (str): Type of cube to work with (e.g. icubes.fits)
        fileExt (str): New file extension for output cubes (Default: .c.fits)
        xcrop (int tuple): Indices of range to crop x-axis to. Default: None.
        ycrop (int tuple): Indices of range to crop y-axis to. Default: None.
        wcrop (int tuple): Wavelength range to crop cube to. Default: None.

    Returns:
        list: List of trimmed FITS (astropy) objects

    """
    trimmedFits_List = []

    #Trim each cube
    for fitsFile in fitsList:

        data = fitsFile[0].data.copy()
        header = fitsFile[0].header.copy()

        if xcrop==None: xcrop=[0,-1]
        if ycrop==None: ycrop=[0,-1]
        if wcrop==None: zcrop=[0,-1]
        else: zcrop=getband(wcrop[0],wcrop[1],header)

        #Crop cube
        cropData = f[0].data[zcrop[0]:zcrop[1],ycrop[0]:ycrop[1],xcrop[0]:xcrop[1]].copy()
        data = cropData

        #Change RA/DEC/WAV reference pixels
        header["CRPIX1"] -= xcrop[0]
        header["CRPIX2"] -= ycrop[0]
        header["CRPIX3"] -= zcrop[0]

        #Make FITS for trimmed data and add to list
        trimmed_HDU = fits.PrimaryHDU(data)
        trimmed_HDU.header = header
        trimmed_HDUList = fits.HDUList(trimmed_HDU)
        trimmedFits_list.append(trimmed_HDUList)

    return trimmedFits_List


if __name__=="__main__":

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
    params = libs.params.loadparams(paramPath)

    # Get filenames
    fileList = libs.params.findfiles(params,cubeType)

    # Open fits objects
    fitsList = [ fits.open(x) for x in fileList ]

    # Pass to trimming function
    trimmed_fitsList = run(fitsList,fileExt=args.ext,xcrop=(x0,x1),ycrop=(y0,y1),wcrop=(w0,w1))

    for i,trimmedFits in enumerate(trimmed_fitsList):
        outFileName = fileList[i].replace('.fits',fileExt)
        f.writeto(outFileName,overwrite=True)
        print("Saved %s"%outFileName)
