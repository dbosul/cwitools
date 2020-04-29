


from astropy.io import fits
from cwitools import utils
from scipy.stats import sigmaclip, linregress
from tqdm import tqdm

import argparse
import numpy as np
import os
import sys

debug=0
if debug:
    from cwitools import coordinates
    import matplotlib #DEBUG
    import matplotlib.pyplot as plt #DEBUG
    matplotlib.use('TkAgg') #DEBUG

def slicecorr(fits_in):

    hdu = utils.extractHDU(fits_in)
    data, header = hdu.data, hdu.header

    slice_axis = np.nanargmin(data.shape)
    nslices = data.shape[slice_axis]

    #Run through slices
    for i in tqdm(range(nslices)):

        if slice_axis == 1:
            slice_2d = data[:, i, :]
        elif slice_axis == 2:
            slice_2d = data[:, :, i]
        else:
            raise RuntimeError("Shortest axis should be slice axis.")

        if debug:
            if i < 10 or i > 13: continue
            wav_ax = coordinates.get_wav_axis(header)

        xdomain = np.arange(slice_2d.shape[1])

        #Run through wavelength layers
        for wi in range(slice_2d.shape[0]):

            if debug:
                if wav_ax[wi] < 4220 or wav_ax[wi] > 4240: continue

            xprof = slice_2d[wi]
            clipped, lower, upper = sigmaclip(xprof, low=2, high=2)
            usex = (xprof >= lower) & (xprof <= upper)

            #m, c, p, r, sig = linregress(xdomain[usex], xprof[usex])
            #bg_model = m * xdomain + c
            #bg_model = np.poly1d(np.polyfit(xdomain[usex], xprof[usex], 2))(xdomain)
            bg_model = np.median(xprof[usex])

            if debug:
                fig, axes = plt.subplots(2, 1)
                ax, ax2 = axes
                ax.plot(xdomain, xprof, 'k.-')
                ax.plot(xdomain[usex], xprof[usex], 'rs')
                ax.plot(xdomain, bg_model, 'g-')
                ax2.plot(xdomain, xprof - bg_model, 'k.-')
                fig.show()
                plt.waitforbuttonpress()#input("")
                plt.close()

            if slice_axis == 1:
                fits_in[0].data[wi, i, :] -= bg_model
            else:
                fits_in[0].data[wi, :, i] -= bg_model

    return fits_in

def main():

    #Handle input with argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('clist',
                        type=str,
                        help='The input id list.'
    )
    parser.add_argument('ctype',
                        type=str,
                        help='The input cube type.'
    )
    parser.add_argument('-ext',
                        type=str,
                        help='The filename extension to add to modified cubes. Default: .f.fits',
                        default=".sc.fits"
    )
    args = parser.parse_args()


    clist = utils.parse_cubelist(args.clist)
    file_list = utils.find_files(
        clist["ID_LIST"],
        clist["INPUT_DIRECTORY"],
        args.ctype,
        clist["SEARCH_DEPTH"]
    )
    for file_in in file_list:
        fits_in = fits.open(file_in)
        fits_corrected = slicecorr(fits_in)
        out_filename = file_in.replace('.fits', args.ext)
        fits_corrected.writeto(out_filename, overwrite=True)
        print("Saved %s" % out_filename)

if __name__=="__main__": main()
