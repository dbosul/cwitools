"""Rebin a data cube."""
from astropy.io import fits
from cwitools.reduction import rebin
from cwitools import utils
from datetime import datetime

import argparse
import cwitools
import os
import sys
import warnings

def main():

    #Handle user input with argparse
    parser = argparse.ArgumentParser(description='Re-bin cubes by integer amounts along spatial (XY) and/or wavelength (Z) axes.')
    parser.add_argument('cube',
                        type=str,
                        help='Input cube to be binned.'
    )
    parser.add_argument('-xybin',
                        type=int,
                        help='Number of pixels to bin in X,Y axes',
                        default=1
    )
    parser.add_argument('-zbin',
                        type=int,
                        help='Number of pixels to bin in Z axis.',
                        default=1
    )
    parser.add_argument('-out',
                        type=str,
                        help='File extension to add for binned cube (Default: .binned.fits)',
                        default=".binned.fits"
    )
    parser.add_argument('-vardata',
                        action='store_true'
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

    #Get command that was issues
    argv_string = " ".join(sys.argv)
    cmd_string = "python " + argv_string + "\n"

    #Give output summarizing mode
    timestamp = datetime.now()
    infostring = """\n{0}\n{1}\n\tCWI_REBIN:\n
\t\tCUBE = {2}
\t\tXYBIN = {3}
\t\tZBIN = {4}
\t\tOUT = {5}
\t\tVARDATA = {6}
\t\tLOG = {7}
\t\tSILENT = {8}\n\n""".format(timestamp, cmd_string, args.cube, args.xybin,
args.zbin, args.out, args.vardata, args.log, args.silent)
    utils.output(infostring)

    #Load data
    if os.path.isfile(args.cube): inFits = fits.open(args.cube)
    else:
        raise FileNotFoundError("Input file not found.\nFile:%s"%args.cube)

    #Check that user has actually set the bin options
    if args.zbin==1 and args.xybin==1:
        warnings.warn("Binning 1x1x1 won't change anything!")

    binnedFits = rebin(
        inFits,
        xybin=args.xybin,
        zbin=args.zbin,
        vardata=args.vardata
    )


    if args.out == None:
        ext = ".rebin_%i_%i.fits" % (args.xybin, args.zbin)
        outfilename = args.cube.replace(".fits", ext)
    else:
        outfilename = args.out

    binnedFits.writeto(outfilename,overwrite=True)
    utils.output("\tSaved %s" % outfilename)


if __name__ == "__main__": main()
