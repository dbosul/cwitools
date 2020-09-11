"""Estimate the 3D variance for a data cube"""
from cwitools import coordinates, utils
from astropy.io import fits
from astropy.modeling import models, fitting
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage import measure
from tqdm import tqdm

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def scale_variance(data, var, nmin=50, snrmin=3, plot=True):

    snr_range = (-5, 5)
    snr_nbins = 100

    data = np.nan_to_num(data, nan=0)
    var = np.nan_to_num(var, nan=np.inf)



    scale_factor = 1
    scale_factor_change = 1.0
    std_fit = 99
    n = 0
    utils.output("\t%10s %15s %15s %15s\n" % ("iter", "scale_f", "std-dev", "1/std-dev"))
    while abs(std_fit - 1) >= 0.001:

        n += 1
        snr = data / np.sqrt(var)
        #Adjust SNR dist. using latest scale factor
        snr_scaled = snr * scale_factor

        #Segment into regions
        vox_msk = np.abs(snr_scaled) > snrmin
        vox_lab, num_reg = measure.label(vox_msk, return_num=True)

        #Measure sizes of regions above (in absolute terms) snr min
        reg_props = measure.regionprops_table(vox_lab, properties=['area', 'label'])
        large_regions = reg_props['area'] > nmin

        # Create object mask to exclude these regions
        obj_mask = np.zeros_like(data, dtype=bool)
        for label in reg_props['label'][large_regions]:
            obj_mask[vox_lab == label] = 1


        #Get SNR distribution of non-masked regions
        counts, edges = np.histogram(
            snr_scaled[~obj_mask],
            range=[-3, 3],
            bins=50
        )

        #Fit Gaussian model
        centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(edges.size - 1)])
        noisefitter = fitting.LevMarLSQFitter()
        noisemodel0 = models.Gaussian1D(amplitude=counts.max(), mean=0, stddev=1)
        noisemodel1 = noisefitter(noisemodel0, centers, counts)
        std_fit1 = noisemodel1.stddev.value
        fit_cens = np.abs(centers) > 0.5 * std_fit1
        noisemodel2 = noisefitter(noisemodel0, centers[fit_cens], counts[fit_cens])
        std_fit = noisemodel2.stddev.value

        if plot:
            counts_all, edges_all = np.histogram(
                snr_scaled[~obj_mask],
                range=[-3, 3],
                bins=50
            )
            fig, ax  = plt.subplots(1, 1, figsize=(12,12))
            ax.plot( centers, counts_all, 'k.--', alpha=0.5)
            ax.plot( centers, counts, 'k.--')
            ax.plot( centers[fit_cens], counts[fit_cens], 'kx')
            ax.plot( centers, noisemodel2(centers), 'r-')
            fig.show()
            input("")#plt.waitforbuttonpress()
            plt.close()

        new_scale_factor = 1 / std_fit
        utils.output("\t%10i %15.5f %15.5f %15.5f\n" % (n, scale_factor, std_fit, 1 / std_fit))

        scale_factor *= new_scale_factor


    return 1 / scale_factor**2


def main():
    #Take any additional input params, if provided
    parser = argparse.ArgumentParser(description='Get estimated variance cube.')
    parser.add_argument('data',
                        type=str,
                        help='Data cube.'
    )
    parser.add_argument('var',
                        type=str,
                        help='Variance cube.'
    )
    parser.add_argument('-snr_min',
                        type=float,
                        help='SNR Threshold for detection.',
                        default=2
    )
    parser.add_argument('-n_min',
                        type=int,
                        help='Minimum size of detection',
                        default=100
    )
    parser.add_argument('-wrange',
                        type=str,
                        metavar='Wav Mask',
                        help='Wavelength range(s) to mask when fitting',
                        default=None
    )
    parser.add_argument('-plot',
                        action='store_true',
                        help="Display diagnostic plots."
    )
    parser.add_argument('-out',
                        type=str,
                        metavar='str',
                        help='Filename for output. Default is input + .scaled.fits',
                        default=None
    )
    args = parser.parse_args()

    #Try to load the fits file
    if os.path.isfile(args.data): data_fits = fits.open(args.data)
    else: raise FileNotFoundError("Input file not found.")


    #Try to load the fits file
    if os.path.isfile(args.var):
        var_fits = fits.open(args.var)
    else:
        raise FileNotFoundError("Variance file not found.")

    data = data_fits[0].data
    var = var_fits[0].data

    wav_axis = coordinates.get_wav_axis(data_fits[0].header)

    #Try to parse the wavelength mask tuple
    if args.wrange != None:
        try:
            w0,w1 = tuple(int(x) for x in args.wrange.split(':'))
        except:
            raise ValueError("Could not parse wmask argument (%s)." % args.wmask)
    else:
        w0, w1 = wav_axis[0], wav_axis[-1]


    zmask = (wav_axis >= w0) & (wav_axis <= w1)
    data_scale = data[zmask]
    var_scale = var[zmask]

    scale_factor = scale_variance(data_scale, var_scale,
        nmin=args.n_min,
        snrmin=args.snr_min,
        plot = args.plot
    )
    #scale_factor = scale_variance(data_scale, var_scale,
    #    nmin=args.n_min,
    #    snr_min=args.snr_min
    #)

    if args.out == None:
        outfilename = args.var.replace('.fits', '.scaled.fits')
    else:
        outfilename = args.out

    utils.output("Std-dev of Noise SNR Distribution = %.3f\n" % np.sqrt(scale_factor))
    utils.output("Variance Scaled by %.3f to assert Standard Normal Distribution\n" % scale_factor)
    var_fits[0].data *= scale_factor
    var_fits.writeto(outfilename, overwrite=True)
    utils.output("Saved %s\n" % outfilename)

if __name__ == "__main__": main(TBD, arg_parser=parser_init())
