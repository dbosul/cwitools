"""Subtract background signal from a data cube"""

from astropy.io import fits
from cwitools import extraction, utils
from datetime import datetime

import argparse
import cwitools
import os
import sys

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Perform background subtraction on a data cube.')

    mainGroup = parser.add_argument_group(title="Main",description="Basic input")
    mainGroup.add_argument('cube',
                        type=str,
                        help='Individual cube or cube type to be subtracted.',
                        default=None
    )
    mainGroup.add_argument('-list',
                        type=str,
                        metavar='<cube_list>',
                        help='CWITools cube list'
    )
    mainGroup.add_argument('-var',
                        metavar='<var_cube/type>',
                        type=str,
                        help="Variance cube or variance cube type."
    )
    methodGroup = parser.add_argument_group(title="Methods",description="Parameters related to BKG Subtraction methods.")
    methodGroup.add_argument('-method',
                        type=str,
                        help='Which method to use for subtraction. Polynomial fit or median filter. (\'medfilt\' or \'polyFit\')',
                        choices=['medfilt','polyfit','noiseFit','median'],
                        default='medfilt'
    )
    methodGroup.add_argument('-k',
                        metavar='<poly_degree>',
                        type=int,
                        help='Degree of polynomial (if using polynomial sutbraction method).',
                        default=1
    )
    methodGroup.add_argument('-window',
                        metavar='<window_size_px>',
                        type=int,
                        help='Size of median window (if using median filtering method).',
                        default=31
    )
    methodGroup.add_argument('-wmask',
                        metavar='<w0:w1,w2:w3,...>',
                        type=str,
                        help='Wavelength range(s) to mask when fitting or filtering',
                        default='0:0'
    )
    fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")
    fileIOGroup.add_argument('-savemodel',
                        help='Set flag to output background model cube (.bg.fits)',
                        action='store_true'
    )
    fileIOGroup.add_argument('-ext',
                        metavar='<file_ext>',
                        type=str,
                        help='Extension to append to input cube for output cube (.bs.fits)',
                        default='.bs.fits'
    )
    fileIOGroup.add_argument('-log',
                        metavar='<log_file>',
                        type=str,
                        help="Log file to save output in",
                        default=None
    )
    fileIOGroup.add_argument('-silent',
                        help='Set flag to mute standard output.',
                        action='store_true'
    )
    args = parser.parse_args()

    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Get command that was issued
    argv_string = " ".join(sys.argv)
    cmd_string = "python " + argv_string + "\n"

    #Summarize script usage
    timestamp = datetime.now()

    infostring = """\n{11}\n{12}\n\tCWI_BGSUB:\n
\t\tCUBE = {0}
\t\tLIST = {1}
\t\tVAR = {2}
\t\tMETHOD = {3}
\t\tK = {4}
\t\tWINDOW = {5}
\t\tWMASK = {6}
\t\tSAVEMODEL = {7}
\t\tEXT = {8}
\t\tLOG = {9}
\t\tSILENT = {10}\n\n""".format(args.cube, args.list, args.var, args.method,
    args.k, args.window, args.wmask, args.savemodel, args.ext, args.log,
    args.silent, timestamp, cmd_string)

    #Output info string
    utils.output(infostring)


    #Load from list and type if list is given
    if args.list != None:

        clist = utils.parse_cubelist(args.list)
        file_list =  utils.find_files(
            clist["ID_LIST"],
            clist["INPUT_DIRECTORY"],
            args.cube,
            clist["SEARCH_DEPTH"]
        )

    #Load list of individual cubes if that is given instead
    else:
        if os.path.isfile(args.cube):
            file_list = [args.cube]
        else:
            raise FileNotFoundError(x)

    #By default, assume we are propagating variance
    usevar = True
    #If var is a file
    if args.var == None:
        usevar = False
        var_file_list = []

    elif os.path.isfile(args.var) and args.list == None:
        var_file_list = [args.var]

    #If not a file and not None - assume it is a cube type
    elif args.list != None:

        var_file_list =  utils.find_files(
            clist["ID_LIST"],
            clist["INPUT_DIRECTORY"],
            args.var,
            clist["SEARCH_DEPTH"]
        )

    #If none of the above, don't use var
    else:
        raise SyntaxError("Variance input not understood. Check usage and try again.")

    #Try to parse the wavelength mask tuple
    masks = []
    if args.wmask != None:
        try:
            for pair in args.wmask.split('-'):
                w0, w1 = tuple(int(x) for x in pair.split(':'))
                masks.append((w0, w1))
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)

    for i, filename in enumerate(file_list):

        fits_file = fits.open(filename)

        subtracted_cube, bg_model, var = extraction.bg_sub(fits_file,
                                method=args.method,
                                poly_k=args.k,
                                median_window=args.window,
                                wmasks=masks
        )

        outfile = args.cube.replace('.fits',args.ext)

        sub_fits = fits.HDUList([fits.PrimaryHDU(subtracted_cube)])
        sub_fits[0].header  = fits_file[0].header
        sub_fits.writeto(outfile,overwrite=True)
        utils.output("\tSaved %s\n" % outfile)

        if args.savemodel:
            model_out = outfile.replace('.fits','.bg_model.fits')
            model_fits = fits.HDUList([fits.PrimaryHDU(bg_model)])
            model_fits[0].header  = fits_file[0].header
            model_fits.writeto(model_out, overwrite=True)
            utils.output("\tSaved %s\n" % model_out)

        if usevar:
            var_fits_in = fits.open(var_file_list[i])
            var_in = var_fits_in[0].data
            print(var_in.shape, var.shape)
            varfileout = outfile.replace('.fits','.var.fits')
            var_fits_out = fits.HDUList([fits.PrimaryHDU(var + var_in)])
            var_fits_out[0].header  = var_fits_in[0].header
            var_fits_out.writeto(varfileout,overwrite=True)
            utils.output("\tSaved %s\n" % varfileout)

if __name__=="__main__": main()
