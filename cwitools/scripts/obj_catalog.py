"""Automatically associate 3D objects with known emission lines."""
from astropy.io import fits
from cwitools import utils, coordinates, extraction, synthesis
from datetime import datetime
from matplotlib import gridspec,colors
from matplotlib.backends.backend_pdf import PdfPages
from skimage import morphology

import argparse
import cwitools
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

import matplotlib
matplotlib.use('TkAgg')

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Make channel maps of an input cube around a specified emission line.')
    parser.add_argument(
        'int',
        type=str,
        help='The input intensity cube.'
    )
    parser.add_argument(
        'obj',
        type=str,
        help='The input object cube.'
    )
    parser.add_argument(
        '-z',
        type=float,
        help='The redshift of the emission. (Default:0)',
        default=0
    )
    parser.add_argument(
        '-zla',
        type=float,
        help='The redshift of LyA emission. (Default:0)',
        default=None
    )
    parser.add_argument(
        '-out',
        type=str,
        help='Name for output PDF with object plots.',
        default="obj_catalog.pdf"
    )
    parser.add_argument(
        '-log',
        metavar="<log_file>",
        type=str,
        help="Log file to save output in.",
        default=None
    )
    parser.add_argument(
        '-silent',
        help="Set flag to suppress standard terminal output.",
        action='store_true'
    )
    args = parser.parse_args()

    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Give output summarizing mode
    cmd = utils.get_cmd(sys.argv)
    titlestring = """\n{0}\n{1}\n\tCWI_OBJ_CATALOG:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(args)
    utils.output(titlestring + infostring)

    int_fits = fits.open(args.int)
    obj_fits = fits.open(args.obj)
    obj = obj_fits[0].data
    obj_ids = np.unique(obj[obj > 0])

    wav_axis = coordinates.get_wav_axis(int_fits[0].header)
    neblines = utils.get_neblines(wav_low=wav_axis[0], wav_high=wav_axis[-1], z=args.z)
    neblines_wavs = neblines['WAV']
    neblines_labels = neblines['ION']

    if args.zla is not None:
        wla = 1215.7 * (1 + args.zla)
    else:
        wla = 0
    #pdfout = PdfPages(args.out)

    with PdfPages(args.out) as pdfout:

        bin_cube = obj > 0
        sb_all = synthesis.obj_sb(int_fits, bin_cube, 1)[0].data
        spec_all = synthesis.obj_spec(int_fits, bin_cube, 1, limit_z=True)
        wav_all, flux_all = spec_all.data['wav'], spec_all.data['flux']

        fig = plt.figure(figsize=(32, 5))
        gs = gridspec.GridSpec(
            ncols=2,
            nrows=1,
            figure=fig,
            width_ratios=[1, 5]
        )

        sb_ax = fig.add_subplot(gs[0, 0])
        spc_ax = fig.add_subplot(gs[0, 1])

        for item in neblines:
            spc_ax.plot([item['WAV']]*2, [0, 2], 'r--')
            spc_ax.text(item['WAV'] + 2, 0.8, item['ION'], color = 'r', rotation = 90 )

        if wla > 0:
            spc_ax.plot([wla]*2, [0, 2], 'b--')
            spc_ax.text(wla + 2, 0.8, "LyA(z)", color = 'b', rotation = 90 )
        sb_ax.set_aspect("equal")
        sb_ax.pcolor(sb_all, norm=colors.LogNorm(vmin=0.01, vmax=1))
        sb_ax.contour(sb_all > 0, levels=[0.5], colors='k')
        spc_ax.step(wav_all, flux_all, 'k-')

        spc_ax.set_xlim([wav_all[0], wav_all[-1]])
        spc_ax.set_ylim([0.01, 2.0])
        spc_ax.set_yscale('log')
        spc_ax.tick_params(labelsize=16)
        spc_ax.set_xlabel(r"$\mathrm{\lambda~[\AA]}$", fontsize=24)
        spc_ax.set_ylabel(r"$\mathrm{10^{-16} F_{\lambda}~[erg~s^{-1}cm^{-2}\AA^{-1}]}$", fontsize=24)
        sb_ax.set_ylabel("ALL", fontsize=24)
        sb_ax.set_xticks([])
        sb_ax.set_yticks([])
        fig.show()
        pdfout.savefig(fig, orientation='landscape')
        #input("")#plt.waitforbuttonpress()
        plt.close()

        for obj_id in obj_ids:

            sb_map = synthesis.obj_sb(int_fits, obj, obj_id)[0].data
            spec = synthesis.obj_spec(int_fits, obj, obj_id, limit_z=True)

            N_obj = np.count_nonzero(obj == obj_id)
            wav, flux = spec.data['wav'], spec.data['flux']

            z_inds = np.where(flux > 0)[0]
            z0, z1 = z_inds[0], z_inds[-1]
            w0, w1 = wav[z0], wav[z1]

            w0 -= 50
            w1 += 50
            usewav = (wav >= w0) & (wav <= w1)
            wav = wav[usewav]
            flux = flux[usewav]

            fig = plt.figure(figsize=(32, 5))
            gs = gridspec.GridSpec(
                ncols=2,
                nrows=1,
                figure=fig,
                width_ratios=[1, 5]
            )

            sb_ax = fig.add_subplot(gs[0, 0])
            spc_ax = fig.add_subplot(gs[0, 1])

            for item in neblines:
                spc_ax.plot([item['WAV']]*2, [0, 2], 'r--')
                spc_ax.text(item['WAV'] + 2, 0.8, item['ION'], color = 'r', rotation = 90 )

            sb_ax.set_aspect("equal")
            sb_ax.pcolor(sb_map, norm=colors.LogNorm(vmin=0.01, vmax=1))
            sb_ax.contour(sb_map > 0, levels=[0.5], colors='k')
            spc_ax.step(wav, flux, 'k-')

            spc_ax.set_xlim([wav[0], wav[-1]])
            spc_ax.set_ylim([0.001, 2.0])
            spc_ax.set_yscale('log')
            spc_ax.tick_params(labelsize=16)
            spc_ax.set_xlabel(r"$\mathrm{\lambda~[\AA]}$", fontsize=24)
            spc_ax.set_ylabel(r"$\mathrm{10^{-16} F_{\lambda}~[erg~s^{-1}cm^{-2}\AA^{-1}]}$", fontsize=24)
            sb_ax.set_ylabel("%i (N=%i)" %(obj_id, N_obj), fontsize=24)
            sb_ax.set_xticks([])
            sb_ax.set_yticks([])
            fig.show()
            pdfout.savefig(fig, orientation='landscape')
            #plt.waitforbuttonpress()
            plt.close()


if __name__=="__main__": main()
