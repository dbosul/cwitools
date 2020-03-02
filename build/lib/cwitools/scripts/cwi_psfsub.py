from astropy.io import fits
from astropy.wcs import WCS
from cwitools.analysis import subtraction
from cwitools.coordinates import get_header2d
from cwitools import parameters

import argparse
import numpy as np
import os

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Perform PSF subtraction on a data cube.')
    cubeGroup = parser.add_argument_group(
        title="Cube Input",
        description="Individual or paramfile-based input."
    )
    cubeGroup.add_argument('-cube',
                        type=str,
                        metavar='cube',
                        help='The cube to be PSF subtracted.'
    )
    cubeGroup.add_argument('-param',
                        type=str,
                        metavar='param',
                        help='CWITools parameter file.'
    )
    cubeGroup.add_argument('-cubetype',
                        type=str,
                        metavar='cubetype',
                        help='Type of input cube to work with (e.g. icubes.fits)',
    )
    srcGroup = parser.add_mutually_exclusive_group(required=True)
    srcGroup.add_argument('-xy',
                        type=str,
                        metavar='float tuple',
                        help='Position of one source (x,y) to subtract.',
                        default=None
    )
    srcGroup.add_argument('-radec',
                        type=str,
                        metavar='float tuple',
                        help='Position of one source (ra, dec) to subtract.',
                        default=None
    )
    methodGroup = parser.add_argument_group(title="Method",description="Parameters related to PSF subtraction methods.")
    methodGroup.add_argument('-rmin',
                        type=float,
                        metavar='Fit Radius',
                        help='Radius (pixels) used to FIT the PSF model (default 2)',
                        default=2
    )
    methodGroup.add_argument('-rmax',
                        type=float,
                        metavar='Sub Radius',
                        help='Radius (pixels) of subtraction area (default 15).',
                        default=15
    )
    methodGroup.add_argument('-wlwindow',
                        type=int,
                        metavar='PSF Window',
                        help='Window (angstrom) used to create WL image of PSF (default 150).',
                        default=150
    )
    methodGroup.add_argument('-wmask',
                        type=str,
                        metavar='Wav Mask',
                        help='Wavelength range(s) to mask when fitting',
                        default=None
    )
    fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")
    fileIOGroup.add_argument('-ext',
                        type=str,
                        metavar='File Extension',
                        help='Extension to append to subtracted cube (.ps.fits)',
                        default='.ps.fits'
    )
    fileIOGroup.add_argument('-savepsf',
                        help='Set flag to output PSF Cube)',
                        action='store_true'
    )
    fileIOGroup.add_argument('-v', help="Verbose: display progress and info.",action="store_true")
    args = parser.parse_args()

    #Try to load the fits file
    if args.cube != None:

        if os.path.isfile(args.cube):
            files_in = [args.cube]
        else:
            raise FileNotFoundError("Input file not found.\nFile:%s"%args.cube)

    elif args.param != None and args.cubetype != None:
        params = parameters.load_params(args.param)
        files_in = parameters.find_files(
            params["ID_LIST"],
            params["INPUT_DIRECTORY"],
            args.cubetype,
            depth=params["SEARCH_DEPTH"]
        )

    else:
        raise ValueError("Must provide either -cube as an argument OR -param and -cubetype")
    #Try to parse the wavelength mask tuple
    try:
        masks = []
        for pair in args.wmask.split('-'):
            w0,w1 = tuple(int(x) for x in pair.split(':'))
            masks.append((w0,w1))
    except:
        raise ValueError("Could not parse zmask argument (%s)." % args.wmask)

    for file_in in files_in:

        fits_in = fits.open(file_in)
        header2d = get_header2d(fits_in[0].header)
        wcs2d = WCS(header2d)

        #Take position of source as either x,y pair...
        if args.xy != None:
            pos = tuple(int(x) for x in args.xy.split(','))

        #Or ra,dec pair and convert to x,y
        else:
            ra, dec = tuple(float(x) for x in args.radec.split(','))
            pos = wcs2d.all_world2pix(ra, dec, 0)
            pos = tuple(int(round(float(x))) for x in pos)

        #Get subtracted cube and psf model
        sub_cube, psf_model = subtraction.psf_subtract_byslice(fits_in,
            pos=pos,
            fit_rad = args.rmin,
            sub_rad = args.rmax,
            wl_window = args.wlwindow,
            wmasks = masks
        )

        outFileName = file_in.replace('.fits',args.ext)
        outFits = fits.HDUList([fits.PrimaryHDU(sub_cube)])
        outFits[0].header = fits_in[0].header
        outFits.writeto(outFileName,overwrite=True)
        print("Saved {0}".format(outFileName))

        if args.savepsf:
            psfOut  = outFileName.replace('.fits','.psf_model.fits')
            psfFits = fits.HDUList([fits.PrimaryHDU(psf_model)])
            psfFits[0].header = fits_in[0].header
            psfFits.writeto(psfOut,overwrite=True)
            print("Saved {0}.".format(psfOut))


if __name__=="__main__":
    main()
