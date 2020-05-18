from astropy.io import fits
from astropy import units as u
from cwitools import coordinates, utils
from datetime import datetime
from glob import glob
from scipy.ndimage import gaussian_filter
from scipy.stats import sigmaclip, linregress
from skimage import measure

import argparse
import cwitools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_proc(proc_file):
    """Load a kcwi.proc file as a dictionary"""

    # Define the columns (space-separated) as they appear in kcwi.proc files
    # Note that imno appears twice - this is not a typo
    main_cols = ["imno", "binx", "biny", "amps", "read", "gain", "ssm", "ifu",
    "grat", "filt", "cwav", "jdobs", "expt", "type", "imno", "ra", "dec", "pa",
    "airm", "obj"]

    integer_cols = ["imno", "binx", "biny", "read"]
    float_cols = ["gain", "cwav", "jdobs", "expt", "ra", "dec", "pa", "airm"]

    #Define dictionary to contain all basic info
    proc = {x:[] for x in main_cols}

    #Add additional calibration keys that are not main columns
    calib_keys = ["masterbias", "masterflat", "geomcbar", "geomarc",
    "mastersky", "masterstd"]

    for ckey in calib_keys:
        proc[ckey] = []

    #Run through all lines in a file
    for i, line in enumerate(open(proc_file)):

        line = line.replace("\n", "")
        row = line.split()

        #Ignore # COMBAK: ommented line
        if line[0] == "#":
            continue

        #Parse image info
        elif 19 <= len(row) <= 20:
            #Parse the main columns
            for i, item in enumerate(row):

                #Exclude second mention of "imno", at index 14
                if i == 14:
                    continue

                if main_cols[i] in integer_cols:
                    item = int(item)
                elif main_cols[i] in float_cols:
                    item = float(item)

                proc[main_cols[i]].append(item)

            #Some rows have no object, add None for these
            if len(row) == 19:
                proc["obj"].append(None)

            #Initialize the extra info (master sky etc.) as NoneType
            for ckey in calib_keys:
                proc[ckey].append(None)

        #Calibration assignment following image info
        elif "=" in line:

            #Update most recent entry from NoneType to new value
            key, val = line.split("=")
            if key in calib_keys:
                proc[key][-1] = val

    return proc

def get_smsk(intf_fits, flat_fits, smooth=1, slice_thresh=0.5, psf_sclip=2,
blob_sclip=1.5, blob_nmin=50):
    """Get 'smsk' (for kcwi_stage5sky) from intf and flat image."""

    intf_hdu = utils.extractHDU(intf_fits)
    intf, intf_hdr = intf_hdu.data, intf_hdu.header
    flat = utils.extractHDU(flat_fits).data

    #Remove NaNs and smooth if requested
    intf = np.nan_to_num(intf, nan=0, posinf=0, neginf=0)
    if smooth is not None:
        intf = gaussian_filter(intf, smooth)

    #Get 1D slice mask from flat image
    slice_prof = np.median(flat, axis=0)
    slice_prof /= slice_prof.max() #Normalize
    slice_prof = slice_prof > slice_thresh #Threshold -> binary mask

    #Expand 1D mask to to 2D image
    slice_msk2d = np.zeros_like(intf.T)
    slice_msk2d[slice_prof] = 1
    slice_msk2d = slice_msk2d.T

    #Get labelled 2D mask of slices to iterate over
    slice_msk2d = measure.label(slice_msk2d) #Label slices

    #Create blank masks for psfs (i.e. traces) and nebulae (i.e. blobs)
    psf_msk2d_T = np.zeros_like(slice_msk2d.T, dtype=bool)
    blob_msk2d_T = np.zeros_like(psf_msk2d_T)

    # Run through slices and build 2D mask for each
    for s in range(slice_msk2d.max()):

        # Get mask and isolate intf data within slice
        s_msk = slice_msk2d == s + 1

        # Get bounds of this slice in 2D (y, x)
        yinds, xinds = np.where(s_msk == 1)
        x0, x1 = xinds.min(), xinds.max()
        y0, y1 = yinds.min(), yinds.max()
        x0 += 5
        x1 -= 5

        # Extract copy of 2D slice
        slice_2d = intf[y0 : y1 + 1, x0 : x1 + 1].copy()

        #
        # STEP 1: PSF MASK
        #

        # Collapse along y (wavelength) axis to get 1D x-profile
        s_xprof = np.sum(slice_2d, axis=0)

        #Sigma-clip to mask sources
        clipped, low, high = sigmaclip(s_xprof, low=psf_sclip, high=psf_sclip)
        s_psf_msk = (s_xprof < low) | (s_xprof > high)

        #Model residuals to flatten
        x_dom = np.arange(x0, x1 + 1)
        m_x, c0, r, p, err = linregress(x_dom[~s_psf_msk], s_xprof[~s_psf_msk])
        line = m_x * x_dom + c0

        #Sigma-clip again with background flattened
        s_xprof -= line
        clipped, low, high = sigmaclip(s_xprof, low=psf_sclip, high=psf_sclip)
        s_psf_msk = (s_xprof < low) | (s_xprof > high)

        #Create mask with full x-axis dimension, fill in 1D psf mask
        m_xprof = np.zeros_like(intf[0], dtype=bool)
        m_xprof[x0 : x1 + 1] = s_psf_msk

        #Fill in to 2D psf mask
        psf_msk2d_T[m_xprof] = 1

        #
        # STEP 2: BLOB/NEBULAR MASK
        #

        # Basic sigma-clipping across x direction, if enough pixels left
        n_unmasked = np.count_nonzero(~s_psf_msk)

        # Require enough unmasks columns/pixels to define median
        if n_unmasked >= 5:

            # Create 2D slice with sources masked
            slice_2d_masked = slice_2d.copy().T
            slice_2d_masked[s_psf_msk] = np.nan
            slice_2d_masked = slice_2d_masked.T

            # Get median spectrum and standard error on mean spectrum
            yprof_mean = np.nanmedian(slice_2d_masked, axis=1)
            yprof_std = np.nanstd(slice_2d_masked, axis=1)

            # Run through each column (i.e. x index) and apply threshold
            for xi in range(slice_2d_masked.shape[1]):

                # Ignore masked columns
                if s_psf_msk[xi]: continue

                # Extract spectrum / column
                spec1d = slice_2d_masked[:, xi]

                # Threshold as per user seting
                spec_msk = spec1d > yprof_mean + blob_sclip * yprof_std
                blob_msk2d_T[x0 + xi] = spec_msk

    # Transpose to get back to original shape
    psf_msk2d = psf_msk2d_T.T
    blob_msk2d = blob_msk2d_T.T

    # Segment blob mask
    blob_msk2d_final = np.zeros_like(blob_msk2d)
    neb_regs = measure.label(blob_msk2d)

    # Measure properties of regions and get those with area > blob_nmin
    regprops = measure.regionprops_table(neb_regs, properties=['label', 'area'])
    good_regs = regprops['area'] >= blob_nmin

    # Fill large regions back into final blob mask
    for region_label in regprops['label'][good_regs]:
        blob_msk2d_final[neb_regs == region_label] = 1

    # Create final sky mask
    smsk = (psf_msk2d | blob_msk2d_final).astype(int)

    # Get fits-like object to return
    smsk_fits = utils.matchHDUType(intf_fits, smsk, intf_hdr)

    return smsk_fits

def main():

    parser = argparse.ArgumentParser(description='Experimental sky subtraction.')
    parser.add_argument('proc_file',
                        type=str,
                        help='The input id list.'
    )
    parser.add_argument('-smooth',
                        type=float,
                        metavar="<float>",
                        help='Std-deviation of a Gaussian smoothing kernel to apply before calculating mask. Useful to detect blob-like features, but avoid over-smoothing. Default 1.0.',
                        default=1
    )
    parser.add_argument('-slice_thresh',
                        type=float,
                        metavar="<float>",
                        help='Relative intensity threshold to apply to define slice mask. Usually should not need to be changed. Default 0.5.',
                        default=0.5
    )
    parser.add_argument('-psf_sclip',
                        type=float,
                        metavar="<float>",
                        help='Sigma-clipping to apply across x-axis within a slice to reject sources. Default 2.0',
                        default=2.5
    )
    parser.add_argument('-blob_sclip',
                        type=float,
                        metavar="<float>",
                        help='Low threshold to apply to define blob contours in slices. Default 1.5. See -blob_nmin also.',
                        default=1.5
    )
    parser.add_argument('-blob_nmin',
                        type=float,
                        metavar="<float>",
                        help='Minimum size of a blob above -blob_sclip to be masked. Default 50 pixels.',
                        default=50
    )
    parser.add_argument('-imno',
                        type=int,
                        metavar="<int>",
                        help='Provide image number to get smsk only for one image',
                        default=None
    )
    parser.add_argument('-log',
                        type=str,
                        help='Log file to store output in.',
                        default=None
    )
    parser.add_argument('-silent',
                        help='Set flag to suppress standard output',
                        action='store_true'
    )
    args = parser.parse_args()

    cmd = utils.get_cmd(sys.argv)

    #Set global parameters
    cwitools.silent_mode = args.silent
    cwitools.log_file = args.log

    #Give output summarizing mode
    titlestring = """\n{0}\n{1}\n\tKCWI_SMSK:""".format(datetime.now(), cmd)
    infostring = utils.get_arg_string(parser)
    utils.output(titlestring + infostring)

    #Open proc file and run through lines
    proc = load_proc(args.proc_file)

    n_imgs = len(proc["imno"])

    #Get base filename for intf file s
    redux_dir = os.path.abspath(args.proc_file).replace("kcwi.proc", "")

    #Run through images
    for n in range(n_imgs):

        imno, object = proc["imno"][n], proc["obj"][n]

        if object is None:
            continue
        elif args.imno is not None and imno != args.imno:
            continue

        # Find intf file associated with this imno
        glob_str = redux_dir + "*{0:05d}_intf.fits".format(imno)

        intf_files = glob(glob_str)
        if len(intf_files) == 0:
            utils.output("WARNING: no _intf.fits file found for ID {0} ({1})\n".format(imno, object))
            continue

        intf_file = intf_files[0]

        #Load intf and the intf associated with the master flat file
        intf_fits = fits.open(intf_file)
        flat_fits = fits.open(proc["masterflat"][n].replace("mflat", "intf"))

        smsk_fits = get_smsk(intf_fits, flat_fits,
            smooth = args.smooth,
            slice_thresh = args.slice_thresh,
            psf_sclip = args.psf_sclip,
            blob_sclip = args.blob_sclip,
            blob_nmin = args.blob_nmin
        )

        smsk_file = intf_file.replace("intf", "smsk")
        smsk_fits.writeto(smsk_file, overwrite=True)
        utils.output("Saved {0} - {1}\n".format(smsk_file, proc["obj"][n]))



if __name__=="__main__": main()
