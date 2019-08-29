from cwitools.analysis import psf_subtract
from cwitools.cubes import get_header2d

import argparse

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Perform PSF subtraction on a data cube.')
    mainGroup = parser.add_argument_group(title="Main",description="Basic input")
    mainGroup.add_argument('cube',
                        type=str,
                        metavar='cube',
                        help='The cube to be PSF subtracted.'
    )
    srcGroup = parser.add_mutually_exclusive_group(required=True)
    srcGroup.add_argument('-reg',
                        type=str,
                        metavar='path',
                        help='Region file of sources to subtract.',
                        default=None
    )
    srcGroup.add_argument('-pos',
                        type=str,
                        metavar='float tuple',
                        help='Position of source (x,y) to subtract.',
                        default=None
    )
    srcGroup.add_argument('-auto',
                        type=str,
                        metavar='float',
                        help='Automatically detect and subtract sources above this SNR (default: 5).',
                        default=None
    )
    methodGroup = parser.add_argument_group(title="Method",description="Parameters related to PSF subtraction methods.")
    methodGroup.add_argument('-rmin',
                        type=float,
                        metavar='Fit Radius',
                        help='Radius (arcsec) used to FIT the PSF model (default 1)',
                        default=1
    )
    methodGroup.add_argument('-rmax',
                        type=float,
                        metavar='Sub Radius',
                        help='Radius (arcsec) of subtraction area (default 3).',
                        default=1
    )
    methodGroup.add_argument('-scaleMask',
                        type=float,
                        metavar='float',
                        help='Scaling factor for PSF mask (mask radius=S*HWHM).',
                        default=1.0
    )
    methodGroup.add_argument('-wlWindow',
                        type=int,
                        metavar='PSF Window',
                        help='Window (angstrom) used to create WL image of PSF (default 150).',
                        default=150
    )
    methodGroup.add_argument('-localWindow',
                        type=int,
                        metavar='Local PSF Window',
                        help='Use this many extra layers around each wavelength layer to construct local PSF for fitting (default 0 - i.e. only fit to current layer)',
                        default=0
    )
    methodGroup.add_argument('-zMask',
                        type=str,
                        metavar='Wav Mask',
                        help='Z-indices to mask when fitting or median filtering (e.g. \'21,32\')',
                        default='0,0'
    )
    methodGroup.add_argument('-zunit',
                        type=str,
                        metavar='Wav Mask',
                        help='Unit of input for zmask. Can be Angstrom (A) or Pixels (px) (Default: A)',
                        default='A',
                        choices=['A','px']
    )
    methodGroup.add_argument('-recenter',
                        type=str,
                        metavar='Recenter',
                        help='Auto-recenter the input positions using PSF centroid',
                        choices=["True","False"],
                        default="True"
    )
    fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")
    fileIOGroup.add_argument('-var',
                        type=str,
                        metavar='varCube',
                        help='The variance cube associated with input cube - used to propagate error.',
                        default=None
    )
    fileIOGroup.add_argument('-ext',
                        type=str,
                        metavar='File Extension',
                        help='Extension to append to subtracted cube (.ps.fits)',
                        default='.ps.fits'
    )
    fileIOGroup.add_argument('-savePSF',
                        type=str,
                        metavar='Save PSFCube',
                        help='Set to True to output PSF Cube)',
                        choices=["True","False"],
                        default="False"
    )
    fileIOGroup.add_argument('-saveMask',
                        type=str,
                        metavar='Save PSFCube',
                        help='Set to True to output 2D Source Mask',
                        choices=["True","False"],
                        default="True"
    )
    args = parser.parse_args()

    #Try to load the fits file
    if os.path.isfile(args.cube): fitsFile = fits.open(args.cube)
    else:
        raise FileNotFoundError("Input file not found.\nFile:%s"%args.cube)

    #Try to parse the wavelength mask tuple
    try: z0,z1 = tuple(int(x) for x in args.zMask.split(','))
    except:
        raise ValueError("Could not parse zmask argument (%s). Should be int tuple."%args.zmask)

    #Convert boolean-like strings to actual booleans
    for x in [args.saveMask,args.savePSF]: x=(x.upper()=="TRUE")

    subCube,psfCube,mask2D = psf_subtract(fitsFile,
        reg=args.reg,
        pos=args.pos,
        auto=args.auto,
        recenter=args.recenter,
        zMask=args.zMask,
        zUnit=args.zUnit,
        wlWindow=args.wlWindow,
        localwindow=args.localWindow,
    )

    headerIn = fitsFile[0].header

    outFileName = args.cube.replace('.fits',fileExt)
    outFits = make_fits(subCube,headerIn)
    outFits.writeto(outFileName,overwrite=True)
    print("Saved {0}".format(outFileName))

    if savePSF:
        psfOut  = outFileName.replace('.fits','.psfModel.fits')
        psfFits = make_fits(psfCube,headerIn)
        psfFits.writeto(psfOut,overwrite=True)
        print("Saved {0}.".format(psfOut))

    if saveMask:
        mskOut  = outFileName.replace('.fits','.psfMask.fits')
        psfMask = make_fits(mask2D,get_header2d(headerIn))
        psfMask.writeto(mskOut,overwrite=True)
        print("Saved {0}.".format(mskOut))

if __name__=="__main__": main()
