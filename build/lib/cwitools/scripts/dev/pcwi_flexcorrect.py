from astropy.io import fits
from scipy.signal import correlate

import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

#INPUT
in_dir = sys.argv[1]
img_ref = sys.argv[2]
plot = 1
thresh = 0.15
filetype = "int.fits"
imgtypes = ["object"]
donotfix = ["bias"]

ref_img = fits.getdata(img_ref).astype(float)
ref_xprof = np.sum(ref_img, axis=0)
ref_xprof -= np.median(ref_xprof[-200:])
ref_xprof /= np.max(ref_xprof)
ref_xprof[ref_xprof < thresh] = 0
ref_xprof[ref_xprof >= thresh] = 1

ref_yprof = np.sum(ref_img, axis=1)
ref_yprof -= np.median(ref_yprof)
ref_yprof[ref_yprof < 0] = 0
ref_yprof /= ref_yprof.max()
ref_yprof[ref_yprof > thresh] = 1
dx = 6
iters = 2
for iter in range(iters):
    for i in range(ref_xprof.size):
        if ref_xprof[i] == 0:
            if np.mean(ref_xprof[i-dx:i]) > 0 and np.mean(ref_xprof[i:i+dx+1]) > 0:
                ref_xprof[i] = 1


if plot:
    template_fig, template_axes = plt.subplots(2, 1, figsize=(8,8))
    template_ax1, template_ax2 = template_axes

    template_ax1.plot(ref_xprof, 'k-')
    template_ax2.plot(ref_yprof, 'k-')

    template_fig.show()
    input("")#plt.waitforbuttonpress()
    plt.close()

slice_template = ref_xprof.copy()
nasmask_template = ref_yprof.copy()

#Load files
in_files = sorted(glob.glob("%s/*%s" % (in_dir, filetype)))
print("#%19s %10s %15s %15s %10s %10s" % ("FILE", "FM4POS", "RA", "DEC", "DX", "DY"))

for file in in_files:

    in_fits = fits.open(file)
    head = in_fits[0].header
    data = in_fits[0].data

    datasec = head["DSEC1"][1:-1].split(',')
    xdatasec, ydatasec = datasec[0], datasec[1]
    x0, x1 = tuple(int(x) for x in xdatasec.split(':'))
    y0, y1 = tuple(int(y) for y in ydatasec.split(':'))

    imtype = head["IMGTYPE"]

    if imtype in donotfix: continue

    ## HORIZONTAL PROFILE CORRECTION
    hprof = np.sum(data, axis=0, dtype=float)
    Nh = hprof.size
    hprof -= np.median(hprof[Nh-100:Nh-1])
    hprof[hprof < 0] = 0
    hprof /= hprof.max()

    hcorr = correlate(slice_template, hprof, mode='same')
    hcorr_max = np.nanargmax(hcorr)
    hcenter = Nh/2
    dx = int(round(hcorr_max - hcenter))

    hprof_corr = hprof.copy()
    hprof_corr[x0:x1] = np.roll(hprof[x0:x1], dx)

    # VERTICAL PROFILE CORRECTION
    vprof = np.sum(data, axis=1, dtype=float)
    vprof -= np.median(vprof)
    vprof[vprof < 0] = 0
    vprof /= vprof.max()

    vcorr = correlate(nasmask_template, vprof, mode='same')
    vcorr_max = np.nanargmax(vcorr)
    vcenter = vprof.size/2
    dy = int(round(vcorr_max - vcenter))

    if np.max(hcorr) < 100: dx = -999
    print("%20s %10i %15.6f %15.6f %10i %10i" % (file, head["FM4POS"], head["RA"], head["DEC"], dx, dy))

    vprof_corr = vprof.copy()
    vprof_corr[y0:y1] = np.roll(vprof[y0:y1], dy)


    #IMAGE CORRECTION
    data_corr = data.copy()
    data_corr[:, x0:x1] = np.roll(data[:, x0:x1], dx, axis=1)
    data_corr[y0:y1, :] = np.roll(data[y0:y1, :], dy, axis=0)

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(10,10))
        ax1, ax2, ax3, ax4 = axes.flatten()
        ax1.set_title("%s %s %i" %(file,head["IMGTYPE"], head["EXPTIME"]))
        ax1.plot(hprof, 'k-')
        ax1.plot(hprof_corr, 'r-')
        ax1.plot(slice_template, 'r-')
        ax1.plot([hcorr_max]*2, [0,1], 'b--')
        ax1.plot([hcenter]*2, [0,1], 'k-')
        ax3.plot(hcorr, 'b-')
        ax3.plot([hcorr_max]*2, [0,hcorr.max()], 'b--')
        ax3.plot([hcenter]*2, [0,hcorr.max()], 'k-')
        ax3.set_xlim(ax1.get_xlim())

        ax2.plot(vprof, 'k-')
        ax2.plot(vprof_corr, 'r-')
        ax2.plot(nasmask_template, 'r-')
        ax2.plot([vcorr_max]*2, [0,1], 'b--')
        ax2.plot([vcenter]*2, [0,1], 'k-')
        ax4.plot(vcorr, 'b-')
        ax4.plot([vcorr_max]*2, [0,vcorr.max()], 'b--')
        ax4.plot([vcenter]*2, [0,vcorr.max()], 'k-')
        ax4.set_xlim(ax2.get_xlim())
        fig.show()
        plt.waitforbuttonpress()
        plt.close()
