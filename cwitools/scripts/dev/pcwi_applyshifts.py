from astropy.io import fits
from scipy.signal import correlate

import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

shifts_tab_in = sys.argv[1]
shifts_tab = np.genfromtxt(shifts_tab_in,
                dtype=None,
                encoding='ascii',
                names=True
)

for i, file in enumerate(shifts_tab['FILE']):

    in_fits = fits.open(file)
    head = in_fits[0].header
    data = in_fits[0].data

    dx = shifts_tab["DX"][i]
    dy = shifts_tab["DY"][i]

    datasec = head["DSEC1"][1:-1].split(',')
    xdatasec, ydatasec = datasec[0], datasec[1]
    x0, x1 = tuple(int(x) for x in xdatasec.split(':'))
    y0, y1 = tuple(int(y) for y in ydatasec.split(':'))

    #IMAGE CORRECTION
    data_corr = data.copy()
    data_corr[:, x0:x1] = np.roll(data[:, x0:x1], dx, axis=1)
    #data_corr[y0:y1, :] = np.roll(data[y0:y1, :], dy, axis=0)

    in_fits[0].data = data_corr
    out_file = file.replace(".fits", "_corr.fits")
    in_fits.writeto(out_file, overwrite=True)
    print("Saved %s" % out_file)
