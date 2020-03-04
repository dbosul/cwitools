"""Tools for extended data reduction."""
from cwitools import coordinates
from cwitools import modeling
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.interpolate import interp1d
from scipy.ndimage.filters import convolve
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import correlate
from scipy.stats import sigmaclip
from shapely.geometry import box, Polygon
from tqdm import tqdm

import argparse
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import warnings

if sys.platform == 'linux': matplotlib.use('TkAgg')

def align_crpix3(fits_list, xmargin=2, ymargin=2):
    """Get relative offsets in wavelength axis by cross-correlating sky spectra.

    Args:
        fits_list (Astropy.io.fits.HDUList list): List of sky cube FITS objects.
        xmargin (int): Margin to use along FITS axis 1 when summing spatially to
            create spectra. e.g. xmargin = 2 - exclude the edge 2 pixels left
            and right from contributing to the spectrum.
        ymargin (int): Margin to use along fits axis 2 when creating spevtrum.

    Returns:
        crpix3_corr (list): List of corrected CRPIX3 values.

    """
    #Extract wavelength axes and normalized sky spectra from each fits
    N = len(fits_list)
    wavs, spcs, crval3s, crpix3s = [], [], [], []
    for i, sky_fits in enumerate(fits_list):

        sky_data, sky_hdr = sky_fits[0].data, sky_fits[0].header
        sky_data = np.nan_to_num(sky_data, nan=0, posinf=0, neginf=0)

        wav = coordinates.get_wav_axis(sky_hdr)

        sky = np.sum(sky_data[:, ymargin:-ymargin, xmargin:-xmargin], axis=(1, 2))
        sky /= np.max(sky)

        spcs.append(sky)
        wavs.append(wav)
        crval3s.append(sky_hdr["CRVAL3"])
        crpix3s.append(sky_hdr["CRPIX3"])

    #Create common wavelength axis to interpolate sky spectra onto
    w0, w1 = np.min(wavs), np.max(wavs)
    dw_min = np.min([x[1] - x[0] for x in wavs])
    Nw = int((w1 - w0) / dw_min) + 1
    wav_common = np.linspace(w0, w1, Nw)

    #Interpolate (linearly) spectra onto common wavelength axis
    spc_interps = [interp1d(wavs[i], spcs[i])(wav_common) for i in range(N)]

    #Cross-correlate interpolated spectra to look for shifts between them
    corrs = []
    for i, spc_int in enumerate(spc_interps):
        corr_ij = correlate(spc_interps[0], spc_int, mode='full')
        corrs.append(np.nanargmax(corr_ij))

    #Subtract first self-correlation (reference point)
    corrs = corrs[0] -  np.array(corrs)

    #Create new
    crpix3s_corr = [crpix3s[i] + c for i, c in enumerate(corrs)]

    #Return corrections to CRPIX3 values
    return crpix3s


def get_crpix12(fits_in, crval1, crval2, box_size=10, plot=False, iters=3, std_max=4):
    """Measure the position of a known source to get crpix1 and crpix2.

    Args:
        fits_in (Astropy.io.fits.HDUList): The input data cube as a fits object
        crval1 (float): The RA/CRVAL1 of the known source
        crval2 (float): The DEC/CRVAL2 of the known source
        crpix12_guess (int tuple): The estimated x,y location of the source.
            If none provided, the existing WCS will be used to estimate x,y.
        box_size (float): The size of the box (in arcsec) to use for measuring.

    Returns:
        crval1 (float): The axis 1 centroid of the source
        crval2 (float): The axis 2 centroid of the source

    """

    # Convention here is that cube dimensions are (w, y, x)
    # For KCWI - x is the across-slice axis, for PCWI it is y

    #Load input
    cube = fits_in[0].data.copy()
    header3d = fits_in[0].header

    #Create 2D WCS and get pixel sizes in arcseconds
    header2d = coordinates.get_header2d(header3d)
    wcs2d = WCS(header2d)
    pixel_scales = proj_plane_pixel_scales(wcs2d)
    y_scale = (pixel_scales[1] * u.deg).to(u.arcsec).value
    x_scale = (pixel_scales[0] * u.deg).to(u.arcsec).value

    #Get initial estimate of source position
    crpix1, crpix2 = wcs2d.all_world2pix(crval1, crval2, 0)

    #Limit cube to good wavelength range and clean cube
    wavgood0, wavgood1 = header3d["WAVGOOD0"], header3d["WAVGOOD1"]
    wav_axis = coordinates.get_wav_axis(header3d)
    use_wav = (wav_axis > wavgood0) & (wav_axis < wavgood1)
    cube[~use_wav] = 0
    cube = np.nan_to_num(cube, nan=0, posinf=0, neginf=0)

    #Create WL image
    wl_img = np.sum(cube, axis=0)
    wl_img -= np.median(wl_img)

    #Extract box and measure centroid
    box_size_x = box_size / x_scale
    box_size_y = box_size / y_scale

    #Get bounds of box - limited by image bounds.
    x0 = max(0, int(crpix1 - box_size_x / 2))
    x1 = min(cube.shape[2] - 1, int(crpix1 + box_size_x / 2 + 1))

    y0 = max(0, int(crpix2 - box_size_y / 2))
    y1 = min(cube.shape[1] - 1, int(crpix2 + box_size_y / 2 + 1))

    #Create data structures for fitting
    x_domain = np.arange(x0, x1)
    y_domain = np.arange(y0, y1)

    x_profile = np.sum(wl_img[y0:y1, x0:x1], axis=0)
    y_profile = np.sum(wl_img[y0:y1, x0:x1], axis=1)

    x_profile /= np.max(x_profile)
    y_profile /= np.max(y_profile)
    #Determine bounds for gaussian profile fit
    x_gauss_bounds = [
        (0, 10),
        (x0, x1),
        (0, std_max / x_scale)
    ]
    y_gauss_bounds = [
        (0, 10),
        (y0, y1),
        (0, std_max / y_scale)
    ]

    #Run differential evolution fit on each profile
    x_fit = modeling.fit_de(
        modeling.gauss1d,
        x_gauss_bounds,
        x_domain,
        x_profile
    )
    y_fit = modeling.fit_de(
        modeling.gauss1d,
        y_gauss_bounds,
        y_domain,
        y_profile
    )

    x_center, y_center = x_fit.x[1], y_fit.x[1]

    #Fit Gaussian to each profile
    if plot:

        x_profile_model = modeling.gauss1d(x_fit.x, x_domain)
        y_profile_model = modeling.gauss1d(y_fit.x, y_domain)

        fig, axes = plt.subplots(2, 2, figsize=(8,8))
        TL, TR = axes[0, :]
        BL, BR = axes[1, :]
        TL.set_title("Full Image")
        TL.pcolor(wl_img, vmin=0, vmax=wl_img.max())
        TL.plot( [x0, x0], [y0, y1], 'w-')
        TL.plot( [x0, x1], [y1, y1], 'w-')
        TL.plot( [x1, x1], [y1, y0], 'w-')
        TL.plot( [x1, x0], [y0, y0], 'w-')
        TL.plot( x_center + 0.5, y_center + 0.5, 'rx')
        TL.set_aspect(y_scale/x_scale)

        TR.set_title("%.1f x %.1f Arcsec Box" % (box_size, box_size))
        TR.pcolor(wl_img[y0:y1, x0:x1], vmin=0, vmax=wl_img.max())
        TR.plot( x_center + 0.5 - x0, y_center + 0.5 - y0, 'rx')
        TR.set_aspect(y_scale/x_scale)

        BL.set_title("X Profile Fit")
        BL.plot(x_domain, x_profile, 'k.-', label="Data")
        BL.plot(x_domain, x_profile_model, 'r--', label="Model")
        BL.plot( [x_center]*2, [0,1], 'r--')
        BL.legend()

        BR.set_title("Y Profile Fit")
        BR.plot(y_domain, y_profile, 'k.-', label="Data")
        BR.plot(y_domain, y_profile_model, 'r--', label="Model")
        BR.plot( [y_center]*2, [0,1], 'r--')
        BR.legend()

        for ax in fig.axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.show()
        plt.waitforbuttonpress()
        plt.close()

    #Return
    return x_center + 1, y_center + 1

def rebin(inputfits, xybin=1, zbin=1, vardata=False):
    """Re-bin a data cube along the spatial (x,y) and wavelength (z) axes.

    Args:
        inputfits (astropy FITS object): Input FITS to be rebinned.
        xybin (int): Integer binning factor for x,y axes. (Def: 1)
        zbin (int): Integer binning factor for z axis. (Def: 1)
        vardata (bool): Set to TRUE if rebinning variance data. (Def: True)
        fileExt (str): File extension for output (Def: .binned.fits)

    Returns:
        astropy.io.fits.HDUList: The re-binned cube with updated WCS/Header.

    Examples:

        Bin a cube by 4 pixels along the wavelength (z) axis:

        >>> from astropy.io import fits
        >>> from cwitools import rebin
        >>> myfits = fits.open("mydata.fits")
        >>> binned_fits = rebin(myfits, zbin = 4)
        >>> binned_fits.writeto("mydata_binned.fits")


    """


    #Extract useful structures
    data = inputfits[0].data.copy()
    head = inputfits[0].header.copy()

    #Get dimensions & Wav array
    z, y, x = data.shape
    wav = coordinates.get_wav_axis(head)

    #Get new sizes
    znew = int(z // zbin)
    ynew = int(y // xybin)
    xnew = int(x // xybin)

    #Perform wavelenght-binning first, if bin provided
    if zbin > 1:

        #Get new bin size in Angstrom
        zbinSize = zbin * head["CD3_3"]

        #Create new data cube shape
        data_zbinned = np.zeros((znew, y, x))

        #Run through all input wavelength layers and add to new cube
        for zi in range(znew * zbin):
            data_zbinned[int(zi // zbin)] += data[zi]

        #Normalize so that units remain as "erg/s/cm2/A"
        if vardata: data_zbinned /= zbin**2
        else: data_zbinned /= zbin

        #Update central reference and pixel scales
        head["CD3_3"] *= zbin
        head["CRPIX3"] /= zbin

    else:

        data_zbinned = data

    #Perform spatial binning next
    if xybin > 1:

        #Get new shape
        data_xybinned = np.zeros((znew, ynew, xnew))

        #Run through spatial pixels and add
        for yi in range(ynew * xybin):
            for xi in range(xnew * xybin):
                xindex = int(xi // xybin)
                yindex = int(yi // xybin)
                data_xybinned[:, yindex, xindex] += data_zbinned[:, yi, xi]

        #
        # No normalization needed for binning spatial pixels.
        # Units remain as 'per pixel' but pixel size changes.
        #

        #Update reference pixel
        head["CRPIX1"] /= float(xybin)
        head["CRPIX2"] /= float(xybin)

        #Update pixel scales
        for key in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]: head[key] *= xybin

    else: data_xybinned = data_zbinned

    binnedFits = fits.HDUList([fits.PrimaryHDU(data_xybinned)])
    binnedFits[0].header = head

    return binnedFits

def crop(inputfits, xcrop=None, ycrop=None, wcrop=None, auto=False, autopad=1,
plot=False):
    """Crops an input data cube (FITS).

    Args:
        inputfits (astropy.io.fits.HDUList): FITS file to be trimmed.
        xcrop (int tuple): Indices of range to crop x-axis to. Default: None.
        ycrop (int tuple): Indices of range to crop y-axis to. Default: None.
        wcrop (int tuple): Wavelength range (A) to crop cube to. Default: None.
        auto (boolean): Set to True to automatically determine all crop params.

    Returns:
        astropy.io.fits.HDUList: Trimmed FITS object with updated header.

    Examples:

        The parameter wcrop (wavelength crop) is in Angstrom, so to crop a
        data cube to the wavelength range 4200-4400A ,the usage would be:

        >>> from astropy.io import fits
        >>> from cwitools.reduction import crop
        >>> myfits = fits.open("mydata.fits")
        >>> myfits_cropped = crop(myfits,wcrop=(4200,4400))

        Crop ranges for the x/y axes are given in image coordinates (px).
        They can be given either as straight-forward indices:

        >>> crop(myfits, xcrop=(10,60))

        Or using negative numbers to count backwards from the last index:

        >>> crop(myfits, ycrop=(10,-10))

    """
    trimmedFits_List = []

    data = inputfits[0].data.copy()
    header = inputfits[0].header.copy()

    data[np.isnan(data)] = 0
    xprof = np.max(data, axis=(0, 1))
    yprof = np.max(data, axis=(0, 2))
    zprof = np.max(data, axis=(1, 2))


    if auto:

        w0, w1 = header["WAVGOOD0"], header["WAVGOOD1"]
        zcrop = z0, z1 = coordinates.get_indices(w0, w1, header)

        xbad = xprof <= 0
        ybad = yprof <= 0

        x0 = xbad.tolist().index(False) - 1 + autopad
        x1 = len(xbad) - xbad[::-1].tolist().index(False) - 1 - autopad

        xcrop = [x0, x1]

        y0 = ybad.tolist().index(False) - 1 + autopad
        y1 = len(ybad) - ybad[::-1].tolist().index(False) - 1 - autopad

        ycrop = [y0, y1]

        print("AutoCrop Parameters:")
        print("\tx-crop: %02i:%02i" % (x0, x1))
        print("\ty-crop: %02i:%02i" % (y0, y1))
        print("\tz-crop: %i:%i (%i:%i A)" % (z0, z1, w0, w1))


    else:

        if xcrop==None: xcrop=[0,-1]
        if ycrop==None: ycrop=[0,-1]
        if wcrop==None: zcrop=[0,-1]
        else: zcrop = coordinates.get_indices(wcrop[0],wcrop[1],header)

    if plot:

        x0, x1 = xcrop
        y0, y1 = ycrop
        z0, z1 = zcrop

        xprof_clean = np.max(data[z0:z1, y0:y1, :], axis=(0, 1))
        yprof_clean = np.max(data[z0:z1, :, x0:x1], axis=(0, 2))
        zprof_clean = np.max(data[:, y0:y1, x0:x1], axis=(1, 2))

        fig, axes = plt.subplots(3, 1, figsize=(8, 8))
        xax, yax, wax = axes
        xax.step(xprof_clean, 'k-')
        xax.set_xlabel("X (Axis 2)", fontsize=14)
        xax.plot([x0, x0], [xprof.min(), xprof.max()], 'r-' )
        xax.plot([x1, x1], [xprof.min(), xprof.max()], 'r-' )

        yax.step(yprof_clean, 'k-')
        yax.set_xlabel("Y (Axis 1)", fontsize=14)
        yax.plot([y0, y0], [yprof.min(), yprof.max()], 'r-' )
        yax.plot([y1, y1], [yprof.min(), yprof.max()], 'r-' )

        wax.step(zprof_clean, 'k-')
        wax.plot([z0, z0], [zprof.min(), zprof.max()], 'r-' )
        wax.plot([z1, z1], [zprof.min(), zprof.max()], 'r-' )
        wax.set_xlabel("Z (Axis 0)", fontsize=14)
        fig.tight_layout()
        fig.show()
        input("")
        plt.close()

    #Crop cube
    cropData = data[zcrop[0]:zcrop[1],ycrop[0]:ycrop[1],xcrop[0]:xcrop[1]]

    #Change RA/DEC/WAV reference pixels
    header["CRPIX1"] -= xcrop[0]
    header["CRPIX2"] -= ycrop[0]
    header["CRPIX3"] -= zcrop[0]

    #Make FITS for trimmed data and add to list
    trimmedFits = fits.HDUList([fits.PrimaryHDU(cropData)])
    trimmedFits[0].header = header

    return trimmedFits

def rotate(wcs, theta):
    """Rotate WCS coordinates to new orientation given by theta.

    Analog to ``astropy.wcs.WCS.rotateCD``, which is deprecated since
    version 1.3 (see https://github.com/astropy/astropy/issues/5175).

    Args:
        wcs (astropy.wcs.WCS): The input WCS to be rotated
        theta (float): The rotation angle, in degrees.

    Returns:
        astropy.wcs.WCS: The rotated WCS

    """
    theta = np.deg2rad(theta)
    sinq = np.sin(theta)
    cosq = np.cos(theta)
    mrot = np.array([[cosq, -sinq],
                     [sinq, cosq]])

    if wcs.wcs.has_cd():    # CD matrix
        newcd = np.dot(mrot, wcs.wcs.cd)
        wcs.wcs.cd = newcd
        wcs.wcs.set()
        return wcs
    elif wcs.wcs.has_pc():      # PC matrix + CDELT
        newpc = np.dot(mrot, wcs.wcs.get_pc())
        wcs.wcs.pc = newpc
        wcs.wcs.set()
        return wcs
    else:
        raise TypeError("Unsupported wcs type (need CD or PC matrix)")


def coadd(fitsList, pa=0, pxthresh=0.5, expthresh=0.1, vardata=False, verbose=False):
    """Coadd a list of fits images into a master frame.

    Args:

        fitslist (lists): List of FITS (Astropy HDUList) objects to coadd
        pxthresh (float): Minimum fractional pixel overlap.
            This is the overlap between an input pixel and a pixel in the
            output frame. If a given pixel from an input frame covers less
            than this fraction of an output pixel, its contribution will be
            rejected.
        expthresh (float): Minimum exposure time, as fraction of maximum.
            If an area in the coadd has a stacked exposure time less than
            this fraction of the maximum overlapping exposure time, it will be
            trimmed from the coadd. Default: 0.1.
        pa (float): The desired position-angle of the output data.
        vardata (bool): Set to TRUE when coadding variance.
        verbose (bool): Show progress bars and file names.

    Returns:

        astropy.io.fits.HDUList: The stacked FITS with new header.


    Raises:

        RuntimeError: If wavelength scales of input are not equal.

    Examples:

        Basic example of coadding three cubes in your current directory:

        >>> from cwitools import reduction
        >>> myfiles = ["cube1.fits","cube2.fits","cube3.fits"]
        >>> coadded_fits = reduction.coadd(myfiles)
        >>> coadded_fits.writeto("coadd.fits")

        More advanced example, using glob to find files:

        >>> from glob import glob
        >>> from cwitools import reduction
        >>> myfiles = glob.glob("/home/user1/data/target1/*icubes.fits")
        >>> coadded_fits = reduction.coadd(myfiles)
        >>> coadded_fits.writeto("/home/user1/data/target1/coadd.fits")

    """

    #DEBUG PLOTTING MODE
    plot=False

    #
    # STAGE 0: PREPARATION
    #

    # Extract basic header info
    hdrList    = [ f[0].header for f in fitsList ]
    wcsList    = [ WCS(h) for h in hdrList ]
    pxScales   = np.array([ proj_plane_pixel_scales(wcs) for wcs in wcsList ])

    # Get 2D headers, WCS and on-sky footprints
    h2DList    = [ coordinates.get_header2d(h) for h in hdrList]
    w2DList    = [ WCS(h) for h in h2DList ]
    footPrints = np.array([ w.calc_footprint() for w in w2DList ])

    # Exposure times
    expTimes = []
    for i,hdr in enumerate(hdrList):
        if "TELAPSE" in hdr: expTimes.append(hdr["TELAPSE"])
        else: expTimes.append(hdr["EXPTIME"])

    # Extract into useful data structures
    xScales,yScales,wScales = ( pxScales[:,i] for i in range(3) )
    pxAreas = [ (xScales[i]*yScales[i]) for i in range(len(xScales)) ]
    # Determine coadd scales
    coadd_xyScale = np.min(np.abs(pxScales[:,:2]))
    coadd_wScale  = np.min(np.abs(pxScales[:,2]))


    #
    # STAGE 1: WAVELENGTH ALIGNMENT
    #
    if verbose: print("Aligning wavelength axes...")
    # Check that the scale (Ang/px) of each input image is the same
    if len(set(wScales))!=1:

        raise RuntimeError("ERROR: Wavelength axes must be equal in scale for current version of code.")

    else:

        # Get common wavelength scale
        cd33 = hdrList[0]["CD3_3"]

        # Get lower and upper wavelengths for each cube
        wav0s = [ h["CRVAL3"] - (h["CRPIX3"]-1)*cd33 for h in hdrList ]
        wav1s = [ wav0s[i] + h["NAXIS3"]*cd33 for i,h in enumerate(hdrList) ]

        # Get new wavelength axis
        wNew = np.arange(min(wav0s)-cd33, max(wav1s)+cd33,cd33)

        # Adjust each cube to be on new wavelenght axis
        for i,f in enumerate(fitsList):

            # Pad the end of the cube with zeros to reach same length as wNew
            f[0].data = np.pad( f[0].data, ( (0, len(wNew)-f[0].header["NAXIS3"]), (0,0) , (0,0) ) , mode='constant' )

            # Get the wavelength offset between this cube and wNew
            dw = (wav0s[i] - wNew[0])/cd33

            # Split the wavelength difference into an integer and sub-pixel shift
            intShift = int(dw)
            spxShift = dw - intShift

            # Perform integer shift with np.roll
            f[0].data = np.roll(f[0].data,intShift,axis=0)

            # Create convolution matrix for subpixel shift (in effect; linear interpolation)
            K = np.array([ spxShift, 1-spxShift ])

            # Shift data along axis by convolving with K
            if vardata: K = K**2

            f[0].data = np.apply_along_axis(lambda m: np.convolve(m, K, mode='same'), axis=0, arr=f[0].data)

            f[0].header["NAXIS3"] = len(wNew)
            f[0].header["CRVAL3"] = wNew[0]
            f[0].header["CRPIX3"] = 1

    #
    # Stage 2 - SPATIAL ALIGNMENT
    #
    if verbose: print("Mapping pixels from input-->sky-->output frames.")

    #Take first header as template for coadd header
    hdr0 = h2DList[0]

    #Get 2D WCS
    wcs0 = WCS(hdr0)

    #Get plate-scales
    dx0,dy0 = proj_plane_pixel_scales(wcs0)

    #Make aspect ratio in terms of plate scales 1:1
    if   dx0>dy0: wcs0.wcs.cd[:,0] /= dx0/dy0
    elif dy0>dx0: wcs0.wcs.cd[:,1] /= dy0/dx0
    else: pass

    #Set coadd canvas to desired orientation

    #Try to load orientation from header
    pa0 = None
    for rotKey in ["ROTPA","ROTPOSN"]:
        if rotKey in hdr0:
            pa0=hdr0[rotKey]
            break

    #If no value was found, set to desired PA so that no rotation takes place
    if pa0==None:
        warnings.warn("No header key for PA (ROTPA or ROTPOSN) found in first input file. Cannot guarantee output PA.")
        pa0 = pa

    #Rotate WCS to the input pa
    wcs0 = rotate(wcs0,pa0-pa)

    #Set new WCS - we will use it later to create the canvas
    wcs0.wcs.set()

    # We don't know which corner is which for an arbitrary rotation, so map each vertex to the coadd space
    x0,y0 = 0,0
    x1,y1 = 0,0
    for fp in footPrints:
        ras,decs = fp[:,0],fp[:,1]
        xs,ys = wcs0.all_world2pix(ras,decs,0)

        xMin,yMin = np.min(xs),np.min(ys)
        xMax,yMax = np.max(xs),np.max(ys)

        if xMin<x0: x0=xMin
        if yMin<y0: y0=yMin

        if xMax>x1: x1=xMax
        if yMax>y1: y1=yMax

    #These upper and lower x-y bounds to shift the canvas
    dx = int(round((x1-x0)+1))
    dy = int(round((y1-y0)+1))

    #
    ra0,dec0 = wcs0.all_pix2world(x0,y0,0)
    ra1,dec1 = wcs0.all_pix2world(x1,y1,0)

    #Set the lower corner of the WCS and create a canvas
    wcs0.wcs.crpix[0] = 1
    wcs0.wcs.crval[0] = ra0
    wcs0.wcs.crpix[1] = 1
    wcs0.wcs.crval[1] = dec0
    wcs0.wcs.set()

    hdr0 = wcs0.to_header()

    #
    # Now that WCS has been figured out - make header and regenerate WCS
    #
    coaddHdr = hdrList[0].copy()

    coaddHdr["NAXIS1"] = dx
    coaddHdr["NAXIS2"] = dy
    coaddHdr["NAXIS3"] = len(wNew)

    coaddHdr["CRPIX1"] = hdr0["CRPIX1"]
    coaddHdr["CRPIX2"] = hdr0["CRPIX2"]
    coaddHdr["CRPIX3"] = 1

    coaddHdr["CRVAL1"] = hdr0["CRVAL1"]
    coaddHdr["CRVAL2"] = hdr0["CRVAL2"]
    coaddHdr["CRVAL3"] = wNew[0]

    coaddHdr["CD1_1"]  = wcs0.wcs.cd[0,0]
    coaddHdr["CD1_2"]  = wcs0.wcs.cd[0,1]
    coaddHdr["CD2_1"]  = wcs0.wcs.cd[1,0]
    coaddHdr["CD2_2"]  = wcs0.wcs.cd[1,1]

    coaddHdr2D = coordinates.get_header2d(coaddHdr)
    coaddWCS   = WCS(coaddHdr2D)
    coaddFP = coaddWCS.calc_footprint()


    #Get scales and pixel size of new canvas
    coadd_dX,coadd_dY = proj_plane_pixel_scales(coaddWCS)
    coadd_pxArea = (coadd_dX*coadd_dY)

    # Create data structures to store coadded cube and corresponding exposure time mask
    coaddData = np.zeros((len(wNew),coaddHdr["NAXIS2"],coaddHdr["NAXIS1"]))
    coaddExp  = np.zeros_like(coaddData)

    W,Y,X = coaddData.shape

    if plot:
        fig1,ax = plt.subplots(1,1)
        for fp in footPrints:
            ax.plot( -fp[0:2,0],fp[0:2,1],'k-')
            ax.plot( -fp[1:3,0],fp[1:3,1],'k-')
            ax.plot( -fp[2:4,0],fp[2:4,1],'k-')
            ax.plot( [ -fp[3,0], -fp[0,0] ] , [ fp[3,1], fp[0,1] ],'k-')
        for fp in [coaddFP]:
            ax.plot( -fp[0:2,0],fp[0:2,1],'r-')
            ax.plot( -fp[1:3,0],fp[1:3,1],'r-')
            ax.plot( -fp[2:4,0],fp[2:4,1],'r-')
            ax.plot( [ -fp[3,0], -fp[0,0] ] , [ fp[3,1], fp[0,1] ],'r-')

        fig1.show()
        plt.waitforbuttonpress()

        plt.close()

        plt.ion()

        grid_width  = 2
        grid_height = 2
        gs = gridspec.GridSpec(grid_height,grid_width)

        fig2 = plt.figure(figsize=(12,12))
        inAx  = fig2.add_subplot(gs[ :1, : ])
        skyAx = fig2.add_subplot(gs[ 1:, :1 ])
        imgAx = fig2.add_subplot(gs[ 1:, 1: ])

    if verbose: pbar = tqdm(total=np.sum([x[0].data[0].size for x in fitsList]))

    # Run through each input frame
    for i,f in enumerate(fitsList):

        #Get shape of current cube
        w,y,x = f[0].data.shape

        # Create intermediate frame to build up coadd contributions pixel-by-pixel
        buildFrame = np.zeros_like(coaddData)

        # Fract frame stores a coverage fraction for each coadd pixel
        fractFrame = np.zeros_like(coaddData)

        # Get wavelength coverage of this FITS
        wavIndices = np.ones(len(wNew),dtype=bool)
        wavIndices[wNew < wav0s[i]] = 0
        wavIndices[wNew > wav1s[i]] = 0

        # Convert to a flux-like unit if the input data is in counts
        if "electrons" in f[0].header["BUNIT"]:

            # Scale data to be in counts per unit time
            if vardata: f[0].data /= expTimes[i]**2
            else: f[0].data /= expTimes[i]

            f[0].header["BUNIT"] = "electrons/sec"

        if plot:
            inAx.clear()
            skyAx.clear()
            imgAx.clear()
            inAx.set_title("Input Frame Coordinates")
            skyAx.set_title("Sky Coordinates")
            imgAx.set_title("Coadd Coordinates")
            imgAx.set_xlabel("X")
            imgAx.set_ylabel("Y")
            skyAx.set_xlabel("RA (hh.hh)")
            skyAx.set_ylabel("DEC (dd.dd)")
            xU,yU = x,y
            inAx.plot( [0,xU], [0,0], 'k-')
            inAx.plot( [xU,xU], [0,yU], 'k-')
            inAx.plot( [xU,0], [yU,yU], 'k-')
            inAx.plot( [0,0], [yU,0], 'k-')
            inAx.set_xlim( [-5,xU+5] )
            inAx.set_ylim( [-5,yU+5] )
            #inAx.plot(qXin,qYin,'ro')
            inAx.set_xlabel("X")
            inAx.set_ylabel("Y")
            xU,yU = X,Y
            imgAx.plot( [0,xU], [0,0], 'r-')
            imgAx.plot( [xU,xU], [0,yU], 'r-')
            imgAx.plot( [xU,0], [yU,yU], 'r-')
            imgAx.plot( [0,0], [yU,0], 'r-')
            imgAx.set_xlim( [-0.5,xU+1] )
            imgAx.set_ylim( [-0.5,yU+1] )
            for fp in footPrints[i:i+1]:
                skyAx.plot( -fp[0:2,0],fp[0:2,1],'k-')
                skyAx.plot( -fp[1:3,0],fp[1:3,1],'k-')
                skyAx.plot( -fp[2:4,0],fp[2:4,1],'k-')
                skyAx.plot( [ -fp[3,0], -fp[0,0] ] , [ fp[3,1], fp[0,1] ],'k-')
            for fp in [coaddFP]:
                skyAx.plot( -fp[0:2,0],fp[0:2,1],'r-')
                skyAx.plot( -fp[1:3,0],fp[1:3,1],'r-')
                skyAx.plot( -fp[2:4,0],fp[2:4,1],'r-')
                skyAx.plot( [ -fp[3,0], -fp[0,0] ] , [ fp[3,1], fp[0,1] ],'r-')


            #skyAx.set_xlim([ra0+0.001,ra1-0.001])
            skyAx.set_ylim([dec0-0.001,dec1+0.001])

        # Loop through spatial pixels in this input frame
        for yj in range(y):

            for xk in range(x):


                # Define BL, TL, TR, BR corners of pixel as coordinates
                inPixVertices =  np.array([ [xk-0.5,yj-0.5], [xk-0.5,yj+0.5], [xk+0.5,yj+0.5], [xk+0.5,yj-0.5] ])

                # Convert these vertices to RA/DEC positions
                inPixRADEC = w2DList[i].all_pix2world(inPixVertices,0)

                # Convert the RA/DEC vertex values into coadd frame coordinates
                inPixCoadd = coaddWCS.all_world2pix(inPixRADEC,0)

                #Create polygon object for projection of this input pixel onto coadd grid
                pixIN = Polygon( inPixCoadd )


                #Get bounding pixels on coadd grid
                xP0,yP0,xP1,yP1 = (int(x) for x in list(pixIN.bounds))


                if plot:
                    inAx.plot( inPixVertices[:,0], inPixVertices[:,1],'kx')
                    skyAx.plot(-inPixRADEC[:,0],inPixRADEC[:,1],'kx')
                    imgAx.plot(inPixCoadd[:,0],inPixCoadd[:,1],'kx')

                #Get bounds of pixel in coadd image
                xP0,yP0,xP1,yP1 = (int(round(x)) for x in list(pixIN.exterior.bounds))

                # Upper bounds need to be increased to include full pixel
                xP1+=1
                yP1+=1


                # Run through pixels on coadd grid and add input data
                for xC in range(xP0,xP1):
                    for yC in range(yP0,yP1):

                        try:
                            # Define BL, TL, TR, BR corners of pixel as coordinates
                            cPixVertices =  np.array( [ [xC-0.5,yC-0.5], [xC-0.5,yC+0.5], [xC+0.5,yC+0.5], [xC+0.5,yC-0.5] ]   )

                            # Create Polygon object and store in array
                            pixCA = box( xC-0.5, yC-0.5, xC+0.5, yC+0.5 )

                            # Calculation fractional overlap between input/coadd pixels
                            overlap = pixIN.intersection(pixCA).area/pixIN.area

                            # Add fraction to fraction frame
                            fractFrame[wavIndices, yC, xC] += overlap

                            if vardata: overlap=overlap**2

                            # Add data to build frame
                            # Wavelength axis has been padded with zeros already
                            buildFrame[wavIndices, yC, xC] += overlap*f[0].data[wavIndices, yj, xk]

                        except: continue


                if verbose: pbar.update(1)
        if plot:
            fig2.canvas.draw()
            plt.waitforbuttonpress()

        #Calculate ratio of coadd pixel area to input pixel area
        pxAreaRatio = coadd_pxArea/pxAreas[i]

        # Max value in fractFrame should be pxAreaRatio - it's the biggest fraction of an input pixel that can add to one coadd pixel
        # We want to use this map now to create a flatFrame - where the values represent a covering fraction for each pixel
        flatFrame = fractFrame/pxAreaRatio

        #Replace zero-values with inf values to avoid division by zero when flat correcting
        flatFrame[flatFrame==0] = np.inf

        #Perform flat field correction for pixels that are not fully covered
        buildFrame /= flatFrame

        #Zero any pixels below user-set pixel threshold, and set flat value to inf
        buildFrame[flatFrame<pxthresh] = 0
        flatFrame[flatFrame<pxthresh] = np.inf

        # Create 3D mask of non-zero voxels from this frame
        M = flatFrame<np.inf

        # Add weight*data to coadd (numerator of weighted mean with exptime as weight)
        if vardata: coaddData += (expTimes[i]**2)*buildFrame
        else: coaddData += expTimes[i]*buildFrame

        #Add to exposure mask
        coaddExp += expTimes[i]*M
        coaddExp2D = np.sum(coaddExp,axis=0)

    if verbose:
        pbar.close()
        print("Trimming coadded canvas.")

    if plot: plt.close()

    # Create 1D exposure time profiles
    expSpec = np.mean(coaddExp,axis=(1,2))
    expXMap = np.mean(coaddExp,axis=(0,1))
    expYMap = np.mean(coaddExp,axis=(0,2))

    # Normalize the profiles
    expSpec/=np.max(expSpec)
    expXMap/=np.max(expXMap)
    expYMap/=np.max(expYMap)

    # Convert 0s to 1s in exposure time cube
    ee = coaddExp.flatten()
    ee[ee==0] = 1
    coaddExp = np.reshape( ee, coaddData.shape )

    # Divide by sum of weights (or square of sum)

    if vardata: coaddData /= coaddExp**2
    else: coaddData /= coaddExp

    # Create FITS object
    coaddHDU = fits.PrimaryHDU(coaddData)
    coaddFITS = fits.HDUList([coaddHDU])
    coaddFITS[0].header = coaddHdr

    #Exposure time threshold, relative to maximum exposure time, below which to crop.
    useW = expSpec>expthresh
    useX = expXMap>expthresh
    useY = expYMap>expthresh

    #Trim the data
    coaddFITS[0].data = coaddFITS[0].data[useW]
    coaddFITS[0].data = coaddFITS[0].data[:,useY]
    coaddFITS[0].data = coaddFITS[0].data[:,:,useX]

    #Get 'bottom/left/blue corner of cropped data
    W0 = np.argmax(useW)
    X0 = np.argmax(useX)
    Y0 = np.argmax(useY)

    #Update the WCS to account for trimmed pixels
    coaddFITS[0].header["CRPIX3"] -= W0
    coaddFITS[0].header["CRPIX2"] -= Y0
    coaddFITS[0].header["CRPIX1"] -= X0

    #Create FITS for variance data if we are propagating that
    return coaddFITS
