import argparse
import numpy as np
import sys

from astropy.io import fits
from astropy.wcs import WCS

def run(cubePath,xyBin=1,zBin=1,varData=False,ext=".binned.fits"):

    #Load data
    try: inFits = fits.open(cubePath)
    except:
        print("Could not open input cube. Check path and try again. (Path: %s)"%cubePath)
        sys.exit()

    #Check that user has actually set the bin options
    if zBin==1 and xyBin==1:
        print("Binning 1x1x1 won't change anything! Set the bin sizes with the flags -zBin and -xyBin.")
        sys.exit()

    #Extract useful structures
    data = inFits[0].data
    head = inFits[0].header

    #Get dimensions & Wav array
    z,y,x = data.shape
    wav = libs.cubes.getWavAxis(head)

    #Get new sizes
    znew = int(w/zBin)  + 1 if zBin >1 else z
    ynew = int(y/xyBin) + 1 if xyBin>1 else y
    xnew = int(x/xyBin) + 1 if xyBin>1 else x

    #Perform wavelenght-binning first, if bin provided
    if zBin>1:

        #Get new bin size in Angstrom
        zBinSize = zBin*head["CD3_3"]

        #Create new data cube shape
        data_zBinned = np.zeros((znew,y,x))

        #Run through all input wavelength layers and add to new cube
        for zi in range(z): data_zBinned[ int(zi/zBin) ] += data[zi]

        #Normalize so that units remain as "erg/s/cm2/A"
        if varData: data_zBinned /= zBin**2
        else: data_zBinned /= zBin

        #Update central reference and pixel scales
        head["CD3_3"] *= zBin
        head["CRPIX3"] /= zBin

    else: data_zBinned = data

    #Perform spatial binning next
    if xyBin>1:

        #Get new shape
        data_xyBinned = np.zeros((wnew,ynew,xnew))

        #Run through spatial pixels and add
        for yi in range(y):
            for xi in range(x):
               data_xyBinned[:,yi/xyBin,xi/xyBin] += data_zBinned[:,yi,xi]

        #
        # No normalization needed for binning spatial pixels.
        # Units remain as 'per pixel' but pixel size changes.
        #

        #Update reference pixel
        head["CRPIX1"] /= float(xyBin)
        head["CRPIX2"] /= float(xyBin)

        #Update pixel scales
        for key in ["CD1_1","CD1_2","CD2_1","CD2_2"]: head[key] *= xyBin

    else: data_xyBinned = data_zBinned

    outFileName = cubePath.replace(".fits",ext)

    newFITS = fIO.HDUList( [ fIO.PrimaryHDU(data_xyBinned) ] )
    newFITS[0].header = head
    newFITS.writeto(outFileName,overwrite=True)

    print("Saved %s"%outFileName)

if __name__ == "__main__":

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
    parser.add_argument('-ext',
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

    run(cubePath=args.cube,xyBin=args.xyBin,zBin=args.zBin,varData=args.varData,ext=args.ext)
