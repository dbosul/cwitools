"""Estimate the 3D variance for a data cube"""
from cwitools import coordinates
from astropy.io import fits
from astropy.modeling import models, fitting
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage import measure
from tqdm import tqdm

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def scale_variance2(data, var, nmax=100, snrmin=2):

    snr_range = (-5, 5)
    snr_nbins = 100

    data = np.nan_to_num(data, nan=0)
    var = np.nan_to_num(var, nan=np.inf)

    snr = data / np.sqrt(var)
    obj_mask = np.zeros_like(data, dtype=bool)

    vox_msk = snr <= -3#snrmin
    vox_lab, num_reg = measure.label(vox_msk, return_num=True)
    for label in range(1, num_reg):
        reg = vox_lab == label
        n = np.count_nonzero(reg)
        if n >= nmax:
            print(n, nmax, label)
            vox_lab[reg] = 0
            obj_mask[reg] = 1

    nnew = np.count_nonzero(obj_mask)

    connect = generate_binary_structure(3, 3)
    while nnew > 0:
        obj_mask_exp = binary_dilation(obj_mask, structure=connect)
        obj_mask_new = (obj_mask_exp) & (snr <= -1)
        nnew = np.count_nonzero(obj_mask_new) - np.count_nonzero(obj_mask)
        obj_mask = obj_mask_new.copy()

    obj_mask[snr > 1] = 1

    #Get SNR distribution of non-masked region
    counts, edges = np.histogram(snr[~obj_mask],
        range=[-4, 1],
        bins=50
    )
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(edges.size - 1)]
    noisefitter = fitting.LevMarLSQFitter()
    noisemodel0 = models.Gaussian1D(amplitude=counts.max(), mean=0, stddev=1)
    noisemodel1 = noisefitter(noisemodel0, centers, counts)

    scale_factor = noisemodel1.stddev.value**2
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(snr.flatten(),
        range=snr_range,
        bins=snr_nbins,
        facecolor='k'
    )
    ax.hist(snr[~obj_mask].flatten(),
        range=snr_range,
        bins=snr_nbins,
        facecolor='r'
    )
    centers_all = np.linspace(-5, 5, 100)
    ax.plot(centers_all, noisemodel1(centers_all), 'b-')
    fig.show()
    input("")#plt.waitforbuttonpress()
    plt.close()

    return scale_factor

def scale_variance(data, var, nmin=100, snr_min=2):


    snr_range = (-5, 5)
    snr_nbins = 100

    #Initialize scale factor as 1
    scale_factor = 2
    scale_factor_change = 1.0

    data = np.nan_to_num(data, nan=0)
    var = np.nan_to_num(var, nan=np.inf)

    while scale_factor_change > 0.001:

        #Get initial SNR cube


        #Get labelled mask of regions above or below snr_min
        snr = data / np.sqrt(var*scale_factor)
        vox_mask = (np.abs(snr) >= snr_min)
        vox_lab = measure.label(vox_mask)

        #Reject regions below a size of n_min
        obj_mask = np.zeros_like(data, dtype=bool)
        labels_unique = np.unique(vox_lab[vox_lab > 0])
        for label in tqdm(labels_unique):
            region = vox_lab == label
            if np.count_nonzero(region) >= nmin:
                obj_mask[region] = 1

        #Get SNR distribution of non-masked region
        counts, edges = np.histogram(snr[~obj_mask],
            range=snr_range,
            bins=snr_nbins
        )
        centers = [(edges[i] + edges[i+1]) / 2 for i in range(edges.size - 1)]

        noisefitter = fitting.LevMarLSQFitter()
        noisemodel0 = models.Gaussian1D(amplitude=counts.max(), mean=0, stddev=1)
        noisemodel1 = noisefitter(noisemodel0, centers, counts)


        scale_factor_new =  scale_factor * noisemodel1.stddev.value**2
        scale_factor_change = abs(scale_factor_new - scale_factor)
        scale_factor = scale_factor_new

        print(scale_factor, scale_factor_change)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.hist(snr.flatten(),
            range=snr_range,
            bins=snr_nbins,
            facecolor='k'
        )
        ax.hist(snr[~obj_mask].flatten(),
            range=snr_range,
            bins=snr_nbins,
            facecolor='r'
        )
        ax.plot(centers, noisemodel1(centers), 'b-')
        fig.show()
        plt.waitforbuttonpress()
        plt.close()

    return scale_factor

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

    scale_factor = scale_variance2(data_scale, var_scale,
        nmax=args.n_min,
        snrmin=args.snr_min
    )
    #scale_factor = scale_variance(data_scale, var_scale,
    #    nmin=args.n_min,
    #    snr_min=args.snr_min
    #)
    print(scale_factor)
    if args.out == None:
        outfilename = args.var.replace('.fits', '.scaled.fits')
    else:
        outfilename = args.out

    var_fits[0].data *= scale_factor
    var_fits.writeto(outfilename, overwrite=True)
    print("Saved %s" % outfilename)

if __name__=="__main__": main()
