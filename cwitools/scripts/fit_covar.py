"""Estimate the 3D variance for a data cube"""
from astropy.io import fits
from cwitools import reduction, utils
from datetime import datetime

import argparse
import cwitools
import os
import sys

def main():
    #Take any additional input params, if provided
    parser = argparse.ArgumentParser(description='Get estimated variance cube.')
    parser.add_argument('cube',
                        type=str,
                        metavar='int_cube',
                        help='Input data cube.'
    )
    parser.add_argument('var',
                        type=str,
                        help='Input variance cube.',
                        metavar='var_cube'
    )
    parser.add_argument('-wrange',
                        type=str,
                        metavar='Wav Mask',
                        help='Wavelength range to use for extracting curve',
                        default=None
    )
    parser.add_argument('-alpha_bounds',
                        type=str,
                        help='Range of allowable alpha values, specified in the format 0.1:10',
                        default=None
    )
    parser.add_argument('-norm_bounds',
                        type=str,
                        help='Range of allowable normalization factors, specified in the format 0.1:10',
                        default=None
    )
    parser.add_argument('-thresh_bounds',
                        type=str,
                        help='Range of allowable values for the threshold, specified in the format 0.1:10',
                        default=None
    )
    parser.add_argument('-beta_bounds',
                        type=str,
                        help='Range of allowable values for the beta, specified in the format 0.1:10',
                        default=None
    )
    parser.add_argument('-mask_neb',
                        metavar='<redshift>',
                        type=float,
                        help='Prove redshift to auto-mask nebular emission.',
                        default=None
    )
    parser.add_argument('-mask_sky',
                        help="Set to auto-mask bright sky lines.",
                        action='store_true'
    )
    parser.add_argument('-obj',
                        type=str,
                        help='Object mask - use to remove 3D objects.',
                        default=None
    )
    parser.add_argument('-plot',
                        help="Set flag to display plot of fit.",
                        action='store_true'
    )
    parser.add_argument('-out',
                        type=str,
                        metavar='str',
                        help='Filename for output. Default is input + .var.fits',
                        default=None
    )
    parser.add_argument('-log',
                        metavar="<log_file>",
                        type=str,
                        help="Log file to save output in.",
                        default=None
    )
    parser.add_argument('-silent',
                        help="Set flag to suppress standard terminal output.",
                        action='store_true'
    )
    args = parser.parse_args()

    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_GETVAR:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    #Try to load the fits file
    fits_in = fits.open(args.cube)
    varcube = fits.getdata(args.var)

    if args.obj is not None:
        mskcube = fits.getdata(args.obj)
    else:
        mskcube = None

    if args.alpha_bounds is not None:
        alpha_bounds = tuple(float(x) for x in args.alpha_bounds.split(':'))
    else:
        alpha_bounds = (0.1, 10)

    if args.norm_bounds is not None:
        norm_bounds = tuple(float(x) for x in args.norm_bounds.split(':'))
    else:
        norm_bounds = (1, 1)

    if args.thresh_bounds is not None:
        thresh_bounds = tuple(float(x) for x in args.thresh_bounds.split(':'))
    else:
        thresh_bounds = (15, 60)

    if args.beta_bounds is not None:
        beta_bounds = tuple(float(x) for x in args.beta_bounds.split(':'))
    else:
        beta_bounds = (1, 5)

    if args.wrange is not None:
        try:
            wrange = tuple(float(x) for x in args.wrange.split(':'))
        except:
            raise ValueError("Could not parse wrange argument (%s)." % args.wmask)
    else:
        wrange = None

    model_bounds = [alpha_bounds, norm_bounds, thresh_bounds, beta_bounds]
    fits_out, params, bins, ratios = reduction.fit_covar_xy(
        fits_in,
        varcube,
        mask = mskcube,
        mask_sky = args.mask_sky,
        model_bounds = model_bounds,
        wrange = wrange,
        plot = args.plot,
        return_all = True
    )

    utils.output("\t%10s%10s\n" % ("BinArea", "Ratio"))
    for i in range(len(bins)):
        utils.output("\t%10i%10.3f\n" % (bins[i], ratios[i]))

    utils.output("\n\tCovariance Model Parameters:\n"
                 "\t\tAlpha\t= %5.2f (header['COV_ALPH'])\n"
                 "\t\tNorm\t= %5.2f (header['COV_NORM'])\n"
                 "\t\tThresh\t= %5.2f (header['COV_THRE'])\n"
                 "\t\tBeta\t= %5.2f (header['COV_BETA'])\n"
                 % (params[0], params[1], params[2], params[3])
    )


    fits_out.writeto(args.cube, overwrite=True)
    utils.output("\n\tUpdated header and saved to %s\n" % args.cube)


if __name__=="__main__": main()
