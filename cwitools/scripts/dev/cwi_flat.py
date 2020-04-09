from cwitools.libs import params

from astropy.io import fits
from astropy.modeling import models, fitting
from scipy.signal import find_peaks

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def flatfield(cube, peak_height=0.02, peak_widths=(2, 10), plot=False, k=3):

    w, y, s = cube.shape

    #Create in-slice average profile of cube
    xprof = np.sum(cube, axis=(0,2))
    xrange = np.arange(cube.shape[1])
    wrange = np.arange(cube.shape[0])

    #Get peaks and widths of PSFs in xprof
    xprof_norm = (xprof-np.median(xprof))/np.max(xprof)
    peaks, peak_properties = find_peaks(xprof_norm, height=peak_height, width=peak_widths, rel_height=0.9)
    widths = peak_properties['widths']

    #Create boolean index of which pixels are background
    usex = np.ones_like(xrange, dtype=bool)
    for i, p in enumerate(peaks): usex[ np.abs(xrange - p) <= widths[i] ] = 0

    if np.count_nonzero(usex) < 10:
        print("Warning: not enough background pixels in slice to fit to.")

    if plot:

        fig, ax = plt.subplots(1, 1)
        ax.plot(xprof_norm, 'k-')
        for i, p in enumerate(peaks):
            ax.fill_between( [p - widths[i], p+widths[i]], [1, 1], [0, 0], facecolor='r')
        fig.show()
        plt.waitforbuttonpress()
        plt.close()

    #Run through
    bgmodel_init = models.Polynomial1D(degree=k)
    bg_fitter = fitting.LinearLSQFitter()
    for si in range(s):

        wprof_i = np.mean(cube[:, usex, si], axis=1)

        bgmodel_fit = bg_fitter(bgmodel_init, wrange, wprof_i )

        bgmodel = bgmodel_fit(wrange)

        for xi in range(cube.shape[1]):
            cube[:, xi, si] -= bgmodel

        if plot:

            fig, ax = plt.subplots(1, 1)
            ax.plot(wrange, wprof_i, 'k.')
            # ax.plot(xrange[usex], xprof_i[usex], 'kx')
            ax.plot(wrange, bgmodel_fit(wrange), 'r-')
            ax.set_title("Slice %i" % si)
            fig.show()
            plt.waitforbuttonpress()
            plt.close()

    return cube

def main():

    #Handle input with argparse
    parser = argparse.ArgumentParser(description="""
    TBD
    """)
    parser.add_argument('-cube',
                        type=str,
                        help='Cube to be cropped (for working on a single cube).',
                        default=None
    )
    parser.add_argument('-params',
                        type=str,
                        help='CWITools parameter file (for working on a list of input cubes).',
                        default=None
    )
    parser.add_argument('-cubetype',
                        type=str,
                        help='The cube type to load (e.g. icubes.fits) if working with a parameter file.',
                        default=None

    )
    parser.add_argument('-ext',
                        type=str,
                        help='The filename extension to add to modified cubes. Default: .f.fits',
                        default=".f.fits"
    )
    args = parser.parse_args()

    #Make list out of single cube if working in that mode
    if args.cube!=None and args.params==None and args.cubetype==None:

        if os.path.isfile(args.cube): fileList = [args.cube]
        else:
            raise FileNotFoundError("Input file not found. \nFile:%s"%args.cube)

    #Load list from parameter files if working in that mode
    elif args.cube==None and args.params!=None and args.cubetype!=None:

        # Check if any parameter values are missing (set to set-up mode if so)
        if os.path.isfile(args.params): parameters = params.loadparams(args.params)
        else:
            raise FileNotFoundError("Parameter file not found.\nFile:%s"%args.params)

        # Get filenames
        fileList = params.findfiles(parameters,args.cubetype)

    #Make sure usage is understood if some odd mix
    else:
        raise SyntaxError("""
        Usage should be one of the following modes:\
        \n\nUse -cube argument to specify one input cube to crop\
        \nOR\
        \nUse -params AND -cubetype flag together to load cubes from parameter file.
        """)

    # Open fits objects
    for fileName in fileList:

        fitsFile = fits.open(fileName)

        # Pass to trimming function
        fitsFile[0].data = flatfield(fitsFile[0].data)

        outFileName = fileName.replace('.fits',args.ext)
        fitsFile.writeto(outFileName,overwrite=True)
        print("Saved %s"%outFileName)

if __name__=="__main__": main()
