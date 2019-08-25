from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import ndimage

import argparse
import numpy as np
import sys

def run(cubePath,zWindow=10,rescale=True,sigmaclip=4,zmask=(0,0),fMin=0.9,fMax=10,fileExt=".var.fits"):

    #Try to load the fits file
    try: F = fits.open(cubePath)
    except: print("Error: could not open '%s'\nExiting."%cubePath);sys.exit()

    #Try to parse the wavelength mask tuple
    try: z0,z1 = tuple(int(x) for x in zmask.split(','))
    except: print("Could not parse zmask argument. Should be two comma-separated integers (e.g. 21,32)");sys.exit()

    #Output warning
    if z1-z0 > zWindow: print("WARNING: Your z-mask is large relative to your zWindow size - this means your variance estimate near the mask may be unreliable. There must be enough non-masked layers in each bin to get a reliable variance estimate.")


    #Parse boolean input
    rescale = True if rescale=="True" else False


    #Extract data
    D = F[0].data

    #Run sigma-clip if set
    if sigmaclip>0:
        print("Sigma-clipping...")
        D = sigma_clip(D,sigma=sigmaclip).data

    #Make first estimate by binning data
    dz = zWindow
    V = np.zeros_like(D)
    i   = 0
    a,b = (i*dz), (i+1)*dz
    while b < D.shape[0]:
        V[a:b] = np.var(D[a:b],axis=0)
        i+=1
        a,b = (i*dz), (i+1)*dz
    V[a:] = np.var(D[a:],axis=0)

    #Adjust first estimate by rescaling, if set to do so
    if rescale:
        for wi in range(len(V)):

            sig = np.sqrt(V[wi])

            useXY = sig>0

            varNorm = np.var(D[wi][useXY]/sig[useXY])

            #Normalize so that variance of layer as a whole is ~1
            #
            # Note: this assumes most of the 3D field is empty of real signal.
            # Z and XY Masks should be supplied if that is not the case
            #

            rsFactor = (1/varNorm)

            rsFactor = max(rsFactor,fMin)
            rsFactor = min(rsFactor,fMax)

            V[wi] *= rsFactor

    varPath = cubePath.replace('.fits',fileExt)
    F[0].data = V
    F.writeto(varPath,overwrite=True)
    print("Saved %s"%varPath)

if __name__=="__main__":

    #Take any additional input params, if provided
    parser = argparse.ArgumentParser(description='Get estimated variance cube.')
    parser.add_argument('cube',
                        type=str,
                        metavar='path',
                        help='Input cube whose 3D variance you would like to estimate.'
    )
    parser.add_argument('-zWindow',
                        type=int,
                        metavar='int (px)',
                        help='Algorithm chops cube into z-bins and estimates 2D variance map at each bin by calculating it along z-axis. This parameter controls that bin size.',
                        default=10
    )
    parser.add_argument('-rescale',
                        type=str,
                        metavar='bool',
                        help="Whether or not to rescale each wavelength layer to normalize variance to sigma=1 in that layer.",
                        choices=["True","False"],
                        default="True"
    )
    parser.add_argument('-sigmaclip',
                        type=float,
                        metavar='float',
                        help="Sigma-clip threshold in stddevs to apply before estimating variance. Set to 0 to skip sigma-clipping (default: 4)",
                        default=4.0
    )
    parser.add_argument('-zmask',
                        type=str,
                        metavar='int tuple (px)',
                        help='Pair of z-indices (e.g. 21,29) to ignore (i.e. interpolate over) when calculating variance.',
                        default="0,0"
    )
    parser.add_argument('-fMin',
                        type=float,
                        metavar='float',
                        help='Minimum rescaling factor (default 0.9)',
                        default=0.9
    )
    parser.add_argument('-fMax',
                        type=float,
                        metavar='float',
                        help='Maximum rescaling factor (default 10)',
                        default=10
    )
    parser.add_argument('-ext',
                        type=str,
                        metavar='str',
                        help='Extension to add to output file (default .var.fits)',
                        default=".var.fits"
    )
    args = parser.parse_args()

    run(args.cube,
        zWindow=args.zWindow,
        rescale=args.rescale,
        sigmaclip=args.sigmaclip,
        zmask=args.zmask,
        fMin=args.fMin,
        fMax=args.fMax,
        fileExt=args.ext
    )
