"""Reduction tools directly related to cube cropping, coadding, etc."""

#Standard Imports
import warnings

#Third-party Imports
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.stats import sigmaclip, mode
from skimage import morphology
from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon as shapely_polygon
from tqdm import tqdm

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

#Local Imports
from cwitools import reduction, coordinates, utils, synthesis, extraction

def slice_corr(fits_in, mask_reg=None):
    """Perform slice-by-slice median correction for scattered light.

    Args:
        fits_in (HDU or HDUList): The input data cube

    Returns:
        HDU or HDUList (same type as input): The corrected data

    """
    hdu = utils.extract_hdu(fits_in)
    data = hdu.data.copy()

    instrument = utils.get_instrument(hdu.header)
    if instrument == "PCWI":
        slice_axis = 1
    elif instrument == "KCWI":
        slice_axis = 2
    else:
        raise ValueError("Unrecognized instrument")

    slice_axis = np.nanargmin(data.shape)
    nslices = data.shape[slice_axis]

    if mask_reg is not None:
        msk2d = extraction.reg2mask(fits_in, mask_reg)[0].data > 0
    else:
        msk2d = np.zeros_like(data[0], dtype=bool)

    #Run through slices
    for slice_i in tqdm(range(nslices)):

        if slice_axis == 1:
            slice_2d = data[:, slice_i, :]
            msk1d = msk2d[slice_i, :]
        elif slice_axis == 2:
            slice_2d = data[:, :, slice_i]
            msk1d = msk2d[:, slice_i]
        else:
            raise RuntimeError("Shortest axis should be slice axis.")

        #Shrink mask if needed to obtain measurement
        while np.count_nonzero(msk1d == 0) < 5:
            msk1d = morphology.binary_erosion(msk1d)


        #Run through wavelength layers
        for layer_j in range(slice_2d.shape[0]):

            xprof = slice_2d[layer_j].copy()[msk1d == 0]

            _, lower, upper = sigmaclip(xprof, low=2, high=2)
            usex = (xprof >= lower) & (xprof <= upper)

            slice_2d[layer_j] -= np.median(xprof[usex])

    return fits_in


def rebin(inputfits, bin_xy=1, bin_z=1, vardata=False):
    """Re-bin a data cube along the spatial (x,y) and wavelength (z) axes.

    Args:
        inputfits (astropy FITS object): Input FITS to be rebinned.
        bin_xy (int): Integer binning factor for x,y axes. (Def: 1)
        bin_z (int): Integer binning factor for z axis. (Def: 1)
        vardata (bool): Set to TRUE if rebinning variance data. (Def: True)
        fileExt (str): File extension for output (Def: .binned.fits)

    Returns:
        astropy.io.fits.HDUList: The re-binned cube with updated WCS/Header.

    """


    #Extract useful structures
    data = inputfits[0].data.copy()
    head = inputfits[0].header.copy()

    #Get dimensions & Wav array
    n_z, n_y, n_x = data.shape

    #Get new sizes
    n_z_new = int(n_z // bin_z)
    n_y_new = int(n_y // bin_xy)
    n_x_new = int(n_x // bin_xy)

    #Perform wavelenght-binning first, if bin provided
    if bin_z > 1:

        #Create new data cube shape
        data_zbinned = np.zeros((n_z_new, n_y, n_x))

        #Run through all input wavelength layers and add to new cube
        for z_i in range(n_z_new * bin_z):
            data_zbinned[int(z_i // bin_z)] += data[z_i]

        #Normalize so that units remain as "erg/s/cm2/A"
        if vardata:
            data_zbinned /= bin_z**2
        else:
            data_zbinned /= bin_z

        #Update central reference and pixel scales
        head["CD3_3"] *= bin_z
        head["CRPIX3"] /= bin_z

    else:

        data_zbinned = data

    #Perform spatial binning next
    if bin_xy > 1:

        #Get new shape
        data_xybinned = np.zeros((n_z_new, n_y_new, n_x_new))

        #Run through spatial pixels and add
        for y_i in range(n_y_new * bin_xy):
            for x_i in range(n_x_new * bin_xy):
                xindex = int(x_i // bin_xy)
                yindex = int(y_i // bin_xy)
                data_xybinned[:, yindex, xindex] += data_zbinned[:, y_i, x_i]

        #
        # No normalization needed for binning spatial pixels.
        # Units remain as 'per pixel' but pixel size changes.
        #

        #Update reference pixel
        head["CRPIX1"] /= float(bin_xy)
        head["CRPIX2"] /= float(bin_xy)

        #Update pixel scales
        for key in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]:
            head[key] *= bin_xy

    else: data_xybinned = data_zbinned

    binned_fits = fits.HDUList([fits.PrimaryHDU(data_xybinned)])
    binned_fits[0].header = head

    return binned_fits

def get_crop_params(fits_in, plot=False):
    """Get optimized crop parameters for crop().

    Input can be ~astropy.io.fits.HDUList, ~astropy.io.fits.PrimaryHDU or
    ~astropy.io.fits.ImageHDU. If HDUList given, PrimaryHDU will be used.

    Returned objects will be of same type as input.

    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        plot (bool): Set to True to show diagnostic plots.

    Returns:
        crop0 (int tuple): Crop indices for NumPy axis 0 (FITS axis 3).
        crop1 (int tuple): Crop indices for NumPy axis 1 (FITS axis 2).
        crop2 (int tuple): Crop indices for NumPy axis 2 (FITS axis 1).

    """

    hdu = utils.extract_hdu(fits_in)
    data = hdu.data.copy()
    header = hdu.header.copy().copy()

    # instrument
    inst = utils.get_instrument(header)
    hdu_2d, _ = synthesis.whitelight(hdu, mask_sky=True)

    #Extract data so that axes are [wav, in-slice, across-slice] = [w, y, x]
    if inst == 'KCWI':
        wl_img = hdu_2d.data
    elif inst == 'PCWI':
        wl_img = hdu_2d.data.T
        data = np.transpose(data, (0, 2, 1))
    else:
        raise ValueError('Instrument not recognized.')

    #Number of in-slice pixels, Number of slice pixels
    npix, nslices = wl_img.shape

    #Get three views of data, maximum taken along each axis
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

    #Get z-axis crop
    z_0, z_1 = coordinates.get_indices(header["WAVGOOD0"], header["WAVGOOD1"], header)
    zcrop = [z_0, z_1]
    zprof = np.sum(data, axis=(1, 2))
    wav_axis = coordinates.get_wav_axis(header)
    wcrop = [wav_axis[z_0], wav_axis[z_1]]

    #Get profile of in-slice pixels (long axis) and across-slice pixels (short axis)
    inslice_prof = np.sum(data, axis=(0, 2))
    xslice_prof = np.sum(data, axis=(0, 1))

    #Get upper and lower limit on cross-slice pixels
    xslice_mask = xslice_prof == 0
    xslice_crop = [
        int(xslice_mask.tolist().index(False)),
        int(len(xslice_mask) - xslice_mask[::-1].tolist().index(False) - 1)
    ]

    #Get upper and lower limit on in-slice pixels
    inslice_mask = inslice_prof == 0
    inslice_crop = [
        int(np.round(inslice_mask.tolist().index(False))),
        int(len(inslice_mask) - inslice_mask[::-1].tolist().index(False) - 1)
    ]

    utils.output("\tAutoCrop Parameters:\n")
    utils.output("\t\tx-slice-crop: %02i:%02i\n" % (xslice_crop[0], xslice_crop[1]))
    utils.output("\t\tin-slice-crop: %02i:%02i\n" % (inslice_crop[0], inslice_crop[1]))
    utils.output("\t\tz-crop: %i:%i\n" % (z_0, z_1))

    if plot:
        print(inslice_crop, xslice_crop)
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        xax, yax, wax = axes
        xax.step(inslice_prof, 'k-', linewidth=2)
        xax.step(
            range(inslice_crop[0], inslice_crop[1] + 1),
            inslice_prof[inslice_crop[0]:inslice_crop[1] + 1],
            'b-',
            linewidth=2
        )

        lim = xax.get_ylim()
        xax.set_xlabel("In-slice Profile", fontsize=18)
        xax.plot([inslice_crop[0], inslice_crop[0]], [inslice_prof.min(), inslice_prof.max()], 'r-')
        xax.plot([inslice_crop[1], inslice_crop[1]], [inslice_prof.min(), inslice_prof.max()], 'r-')
        xax.plot([0, len(inslice_prof)], [0, 0], 'k--')
        xax.set_ylim(lim)

        yax.step(xslice_prof, 'k-', linewidth=2)
        yax.step(
            range(xslice_crop[0], xslice_crop[1] + 1),
            xslice_prof[xslice_crop[0]:xslice_crop[1] + 1],
            'b-',
            linewidth=2
        )
        lim = yax.get_ylim()
        yax.set_xlabel("Across-slice Profile (Axis 1)", fontsize=18)
        yax.plot([xslice_crop[0], xslice_crop[0]], [xslice_prof.min(), xslice_prof.max()], 'r-')
        yax.plot([xslice_crop[1], xslice_crop[1]], [xslice_prof.min(), xslice_prof.max()], 'r-')
        yax.plot([0, len(xslice_prof)], [0, 0], 'k--')
        yax.set_ylim(lim)

        wax.step(zprof, 'k-', linewidth=2)
        wax.step(range(z_0, z_1), zprof[z_0:z_1], 'b-', linewidth=2)
        lim = wax.get_ylim()
        wax.plot([z_0, z_0], [zprof.min(), zprof.max()], 'r-')
        wax.plot([z_1 - 1, z_1 - 1], [zprof.min(), zprof.max()], 'r-')
        wax.set_xlabel("Z (Axis 0)", fontsize=18)
        wax.set_ylim(lim)

        for ax_j in fig.axes:
            ax_j.set_yticks([])
            ax_j.tick_params(labelsize=16)
        fig.tight_layout()
        plt.show()

    if inst == 'KCWI':
        return zcrop, inslice_crop, xslice_crop
    else:
        return zcrop, xslice_crop, inslice_crop



def crop(fits_in, wcrop=None, ycrop=None, xcrop=None):
    """Crops an input data cube (FITS).

    The best crop parameters can be determined using get_crop_params. The
    arguments xcrop/ycrop 'auto' here to trim empty rows and columns, and wcrop
    can be set to 'auto' to trim to the WAVGOOD range.
    Args:
        fits_in (astropy HDU / HDUList): Input HDU/HDUList with 3D data.
        wcrop (int tuple): Wavelength range (Angstrom) to crop z-axis (axis 0)
            to. Use 'auto' to automatically trim to "WAVGOOD" range.
        ycrop (int tuple): Range to crop y-axis (axis 1) to. Use 'auto' to
            trim empty rows. See get_crop_params for more complete method.
        xcrop (int tuple): Range to crop x-axis (axis 2) to. Use 'auto' to
            trim empty rows. See get_crop_params for more complete method.

    Returns:
        HDU / HDUList*: Trimmed FITS object with updated header.
        *Return type matches type of fits_in argument.

    Examples:

        The parameter wcrop (wavelength crop) is in Angstrom, so to crop a
        data cube to the wavelength range 4200-4400A,the usage would be:

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

    #Extract info
    hdu = utils.extract_hdu(fits_in)
    data = hdu.data.copy()
    header = hdu.header.copy().copy()

    #Get profiles of each axis
    data[np.isnan(data)] = 0

    #Allow any axis to have simple automatic mode.
    if 'auto' in [xcrop, ycrop, wcrop]:
        x_auto, y_auto, w_auto = get_crop_params(fits_in)
        if xcrop == 'auto':
            xcrop = x_auto
        if ycrop == 'auto':
            ycrop = y_auto
        if wcrop == 'auto':
            wcrop = w_auto

    #If crop is not set, use entire axis
    if xcrop is None:
        xcrop = [0, data.shape[2]]

    if ycrop is None:
        ycrop = [0, data.shape[1]]

    if wcrop is None:
        zcrop = [0, data.shape[0]]
    else:
        zcrop = coordinates.get_indices(wcrop[0], wcrop[1], header)

    #Crop cube
    crop_data = data[zcrop[0]:zcrop[1], ycrop[0]:ycrop[1], xcrop[0]:xcrop[1]]

    #Change RA/DEC/WAV reference pixels
    header["CRPIX1"] -= xcrop[0]
    header["CRPIX2"] -= ycrop[0]
    header["CRPIX3"] -= zcrop[0]

    trimmed_hdu = utils.match_hdu_type(fits_in, crop_data, header)

    return trimmed_hdu

def coadd(cube_list, cube_type=None, masks_in=None, var_in=None, pos_ang=None, px_thresh=0.5,
          exp_thresh=0.1, verbose=False, plot=0, drizzle=0):
    """Coadd a list of fits images into a master frame.

    Args:

        cube_list (lists): Input files to be added, specified on one of 3 ways:
            a) A path to a CWITools .list file (must also provide -cube_type)
            b) A Python list of file paths
            c) A Python list of HDU/HDUList objects
        cube_type (str): If cube_list is given as a CWITools .list file, use
            this to specify the main cube type to coadd (e.g. icubes.fits)
        masks_in (list or str): Specification of 3D PCWI/KCWI pipeline masks to
            load and apply to data. Can be given in three ways:
              a) As a list of HDU-like objects (HDU or HDUList)
              b) As a list of file paths
              c) As a cube type (e.g. "mcubes.fits") - this option only works
                 if cube_list is given as a CWITools .list file.
        var_in (list or str): Specification of 3D PCWI/KCWI variance cubes to
            load and use for propagating error. Same rules apply as masks_in.
        px_thresh (float): Minimum fractional pixel overlap.
            This is the overlap between an input pixel and a pixel in the
            output frame. If a given pixel from an input frame covers less
            than this fraction of an output pixel, its contribution will be
            rejected.
        exp_thresh (float): Minimum exposure time, as fraction of maximum.
            If an area in the coadd has a stacked exposure time less than
            this fraction of the maximum overlapping exposure time, it will be
            trimmed from the coadd. Default: 0.1.
        pa (float): The desired position-angle of the output data.
        verbose (bool): Show progress bars and file names.
        drizzle (float): The drizzle factor to use, as a fraction of pixels size.
            E.g. 0.2 will shrink input pixels by 20%.


    Returns:
        astropy.io.fits.HDUList: The stacked FITS with new header.


    Raises:
        RuntimeError: If wavelength scales of input are not equal.

    """
    #
    # DETERMINE USE-CASE
    #

    # Scenario 1 - User provided a .list file
    if isinstance(cube_list, str) and ".list" in cube_list:

        if cube_type is None:
            raise SyntaxError("cube_type must be provided if coadding with a CWITools .list file")

        # Load cube list as dict and overwrite cube_list with list of paths
        int_clist = utils.parse_cubelist(cube_list)
        cube_list = utils.find_files(
            int_clist["ID_LIST"],
            int_clist["DATA_DIRECTORY"],
            cube_type,
            depth=int_clist["SEARCH_DEPTH"]
            )
        int_hdus = [utils.extract_hdu(x) for x in cube_list]

        # Load masks if mask cube type given
        if isinstance(masks_in, str):
            mask_list = utils.find_files(
                int_clist["ID_LIST"],
                int_clist["DATA_DIRECTORY"],
                masks_in,
                depth=int_clist["SEARCH_DEPTH"]
                )
            mask_hdus = [utils.extract_hdu(x) for x in mask_list]
        else:
            mask_hdus = None

        # Load variance if cube type given
        if isinstance(var_in, str):
            var_list = utils.find_files(
                int_clist["ID_LIST"],
                int_clist["DATA_DIRECTORY"],
                var_in,
                depth=int_clist["SEARCH_DEPTH"]
                )
            var_hdus = [utils.extract_hdu(x) for x in var_list]
        else:
            var_hdus = None

    # Scenario 2 - User provides a list of filenames or HDU-like objects
    # Masks and variance in this scenario must also be provided athis way
    elif isinstance(cube_list, list):
        int_hdus = [utils.extract_hdu(x) for x in cube_list]

        if isinstance(masks_in, list):
            mask_hdus = [utils.extract_hdu(x) for x in masks_in]
        else:
            mask_hdus = None

        if isinstance(var_in, list):
            var_hdus = [utils.extract_hdu(x) for x in var_in]
        else:
            var_hdus = None

    else:
        raise SyntaxError("Something is wrong with the given input types for cube_list and/or\
        cube_type. Check and try again.")

    #
    # At this point, we have lists of 3D HDUs for int [msk, var]
    # Next step - prepare some data structures and variables

    drz_f = (1 - drizzle) / 2.0 # Fractional margin to add for drizzle factor
    usemask = mask_hdus is not None #Boolean flags for masking and error prop
    usevar = var_hdus is not None

    footprints = [] #On-sky footprints of each HDU
    wav0s = [] #Lower wavelength limits
    wav1s = [] #Upper wavelength limits
    pas = [] #Position angles
    wscales = [] #Wavelength scales/sampling rates

    #First pass through data to build up the above lists and meta-data
    for i, int_hdu in enumerate(int_hdus):

        #3D Header
        header3d = int_hdu.header

        #On-sky footprint from 2D WCS (from 2D header)
        footprint = WCS(coordinates.get_header2d(header3d)).calc_footprint()

        #Wavelength limits
        wav0 = header3d["CRVAL3"] - (header3d["CRPIX3"] - 1) * header3d["CD3_3"]
        wav1 = wav0 + header3d["NAXIS3"] * header3d["CD3_3"]

        #Position Angle
        if "ROTPA" in header3d:
            pos_ang_i = header3d["ROTPA"]
        elif "ROTPOSN" in header3d:
            pos_ang_i = header3d["ROTPOSN"]
        else:
            warnings.warn("No header key for PA (ROTPA or ROTPOSN) found.")
            pos_ang_i = 0

        # Replace masked voxels with NaN values
        if usemask:
            msk_data = mask_hdus[i].data
            bin_mask = (msk_data == 1) & (msk_data >= 8)
            int_hdu.data[bin_mask] = np.nan

        #Add all of the above to lists
        wscales.append(header3d["CD3_3"])
        footprints.append(footprint)
        wav0s.append(wav0)
        wav1s.append(wav1)
        pas.append(pos_ang_i)

    #If user does not provide a PA, set to mode of input to minimize rotations
    if pos_ang is None:
        pos_ang = mode(np.array(pas)).mode[0]

    # Check that the scale (Ang/px) of each input image is the same
    if len(set(wscales)) != 1:
        raise RuntimeError("ERROR: Wavelength axes must be equal in scale.")


    #
    # STAGE 1: WAVELENGTH ALIGNMENT
    #
    if verbose:
        utils.output("\tAligning wavelength axes...\n")

    # Get common wavelength scale and new wavelength axis
    cd33_0 = wscales[0]
    wav_new = np.arange(min(wav0s) - cd33_0, max(wav1s) + cd33_0, cd33_0)

    # Adjust each cube to be on new wavelength axis
    for i, int_hdu in enumerate(int_hdus):

        # Pad the end of the cube with zeros to reach same length as wav_new
        n_pad_w = len(wav_new) - int_hdu.header["NAXIS3"]

        # Get the wavelength offset between this cube and wav_new
        wav_shift_i = (wav0s[i] - wav_new[0]) / cd33_0

        # Split the wavelength difference into an integer and sub-pixel shift
        int_shift = int(wav_shift_i)
        subpx_shift = wav_shift_i - int_shift

        # Create convolution matrix for subpixel shift
        shift_kernel = np.array([subpx_shift, 1 - subpx_shift])

        #
        # Apply changes
        #

        # 1 - Pad data along z-axis to same length as wav_new
        int_hdu.data = np.pad(
            int_hdu.data,
            ((0, n_pad_w), (0, 0), (0, 0)),
            mode='constant'
            )

        # 2 - Perform integer shift with np.roll
        int_hdu.data = np.roll(int_hdu.data, int_shift, axis=0)

        # 3 - Shift data along axis by convolving with K
        int_hdu.data = np.apply_along_axis(
            lambda m, sk=shift_kernel: np.convolve(m, sk, mode='same'),
            axis=0,
            arr=int_hdu.data
            )

        # 4 - Update header's WCS info for axis 3
        int_hdu.header["NAXIS3"] = len(wav_new)
        int_hdu.header["CRVAL3"] = wav_new[0]
        int_hdu.header["CRPIX3"] = 1

        # Apply steps 1-4 to variance (square the kernel for convolution)
        if usevar:
            var_hdus[i].data = np.pad(
                var_hdus[i].data,
                ((0, n_pad_w), (0, 0), (0, 0)),
                mode='constant'
                )
            # 2 - Perform integer shift with np.roll
            var_hdus[i].data = np.roll(var_hdus[i].data, int_shift, axis=0)

            # 3 - Shift data along axis by convolving with K
            var_hdus[i].data = np.apply_along_axis(
                lambda m, sk=shift_kernel: np.convolve(m, sk**2, mode='same'),
                axis=0,
                arr=var_hdus[i].data
                )

            # 4 - Update header's WCS info for axis 3
            var_hdus[i].header["NAXIS3"] = len(wav_new)
            var_hdus[i].header["CRVAL3"] = wav_new[0]
            var_hdus[i].header["CRPIX3"] = 1

    #
    # Stage 2 - SPATIAL ALIGNMENT
    #
    utils.output("\tMapping pixels from input --> sky --> output frames.\n")

    #Take first 2D header as template for 2D coadd header and get 2D WCS
    hdr0 = coordinates.get_header2d(int_hdus[0].header)
    wcs0 = WCS(hdr0)

    #Get plate-scales and then update WCS to have 1:1 aspect ratio
    dx_0, dy_0 = proj_plane_pixel_scales(wcs0)
    if dx_0 > dy_0:
        wcs0.wcs.cd[:, 0] /= dx_0 / dy_0
    else:
        wcs0.wcs.cd[:, 1] /= dy_0 / dx_0

    #Rotate WCS to the input pa
    wcs0 = reduction.wcs.rotate(wcs0, pas[i] - pos_ang)

    #Set new WCS - we will use it later to create the canvas
    wcs0.wcs.set()

    # We don't know which corner is which for an arbitrary rotation
    # So, map each vertex to the coadd space
    x_0, y_0 = 0, 0
    x_1, y_1 = 0, 0
    for f_p in footprints:
        ras, decs = f_p[:, 0], f_p[:, 1]
        x_all, y_all = wcs0.all_world2pix(ras, decs, 0)
        x_0 = min(np.min(x_all), x_0)
        y_0 = min(np.min(y_all), y_0)
        x_1 = max(np.max(x_all), x_1)
        y_1 = max(np.max(y_all), y_1)

    #Get required size of the canvas in x, y
    coadd_size_x = int(round((x_1 - x_0) + 1))
    coadd_size_y = int(round((y_1 - y_0) + 1))

    #Get RA/DEC of lower-left corner - to establish WCS reference point
    ra0, dec0 = wcs0.all_pix2world(x_0, y_0, 0)

    #Set the lower corner of the WCS and create a canvas
    wcs0.wcs.crpix[0] = 1
    wcs0.wcs.crval[0] = ra0
    wcs0.wcs.crpix[1] = 1
    wcs0.wcs.crval[1] = dec0
    wcs0.wcs.set()

    #Convert back from WCS to 2D header
    hdr0 = wcs0.to_header()

    #Get size of 3D canvas in wav
    coadd_size_w = len(wav_new)

    #
    # Now that spatial WCS has been figured out - regenerate 3D WCS/Header
    # Do this by copying an input 3D header and updating the necessary fields
    #
    coadd_hdr = int_hdus[0].header.copy()

    # Size of each axis
    coadd_hdr["NAXIS1"] = coadd_size_x
    coadd_hdr["NAXIS2"] = coadd_size_y
    coadd_hdr["NAXIS3"] = coadd_size_w

    # Reference image coordinate
    coadd_hdr["CRPIX1"] = hdr0["CRPIX1"]
    coadd_hdr["CRPIX2"] = hdr0["CRPIX2"]
    coadd_hdr["CRPIX3"] = 1

    # World coordinate position at reference image coordinate
    coadd_hdr["CRVAL1"] = hdr0["CRVAL1"]
    coadd_hdr["CRVAL2"] = hdr0["CRVAL2"]
    coadd_hdr["CRVAL3"] = wav_new[0]

    # Change along axes
    coadd_hdr["CD1_1"] = wcs0.wcs.cd[0, 0]
    coadd_hdr["CD1_2"] = wcs0.wcs.cd[0, 1]
    coadd_hdr["CD2_1"] = wcs0.wcs.cd[1, 0]
    coadd_hdr["CD2_2"] = wcs0.wcs.cd[1, 1]

    # Re-generate a WCS object from this coadd header and get on-sky footprint
    coadd_hdr2d = coordinates.get_header2d(coadd_hdr)
    coadd_wcs = WCS(coadd_hdr2d)
    coadd_fp = coadd_wcs.calc_footprint()

    #Get scales and pixel size of new canvas
    coadd_px_area = coordinates.get_pxarea_arcsec(coadd_hdr2d)

    # Create data structures to store coadded cube and corresponding exposure time mask
    coadd_data = np.zeros((coadd_size_w, coadd_size_y, coadd_size_x))
    coadd_exp = np.zeros_like(coadd_data)

    if usevar:
        coadd_var = np.zeros_like(coadd_data)

    if plot:

        fig1, axis = plt.subplots(1, 1)
        for f_p in footprints:
            axis.plot(-f_p[0:2, 0], f_p[0:2, 1], 'k-')
            axis.plot(-f_p[1:3, 0], f_p[1:3, 1], 'k-')
            axis.plot(-f_p[2:4, 0], f_p[2:4, 1], 'k-')
            axis.plot([-f_p[3, 0], -f_p[0, 0]], [f_p[3, 1], f_p[0, 1]], 'k-')
        for f_p in [coadd_fp]:
            axis.plot(-f_p[0:2, 0], f_p[0:2, 1], 'r-')
            axis.plot(-f_p[1:3, 0], f_p[1:3, 1], 'r-')
            axis.plot(-f_p[2:4, 0], f_p[2:4, 1], 'r-')
            axis.plot([-f_p[3, 0], -f_p[0, 0]], [f_p[3, 1], f_p[0, 1]], 'r-')

        fig1.show()
        plt.waitforbuttonpress()
        plt.close()
        plt.ion()

        grid = gridspec.GridSpec(2, 2)
        fig2 = plt.figure(figsize=(12, 12))
        input_ax = fig2.add_subplot(grid[:1, :])
        sky_ax = fig2.add_subplot(grid[1:, :1])
        coadd_ax = fig2.add_subplot(grid[1:, 1:])

    if verbose:
        pbar = tqdm(total=np.sum([x.data[0].size for x in int_hdus]))

    # Run through each input frame
    for i, int_hdu in enumerate(int_hdus):

        header_i = int_hdu.header
        header2d_i = coordinates.get_header2d(header_i)
        wcs2d_i = WCS(header2d_i)
        px_area_i = coordinates.get_pxarea_arcsec(header_i)

        if "TELAPSE" in header_i:
            t_exp_i = header_i["TELAPSE"]
        elif "EXPTIME" in header_i:
            t_exp_i = header_i["EXPTIME"]
        else:
            warnings.warn("No exposure time (TELAPSE/EXPTIME) keyword found in header. Skipping.")
            continue

        # Create intermediate frame to build up coadd contributions pixel-by-pixel
        build_frame = np.zeros_like(coadd_data)

        #Build frame for variance
        if usevar:
            var_build_frame = np.zeros_like(coadd_data)

        # Fract frame stores a coverage fraction for each coadd pixel
        fract_frame = np.zeros_like(coadd_data)

        # Get wavelength coverage of this FITS as binary mask
        wavmask_i = np.ones(len(wav_new), dtype=bool)
        wavmask_i[wav_new < wav0s[i]] = 0
        wavmask_i[wav_new > wav1s[i]] = 0

        # Convert to a flux-like unit if the input data is in counts
        if "electrons" in int_hdu.header["BUNIT"]:
            int_hdu.data /= t_exp_i
            if usevar:
                var_hdus[i].data /= t_exp_i**2 #Propagate error

        if plot:
            input_ax.clear()
            sky_ax.clear()
            coadd_ax.clear()
            input_ax.set_title("Input Frame Coordinates")
            sky_ax.set_title("Sky Coordinates")
            coadd_ax.set_title("Coadd Coordinates")
            coadd_ax.set_xlabel("X")
            coadd_ax.set_ylabel("Y")
            sky_ax.set_xlabel("RA (hh.hh)")
            sky_ax.set_ylabel("DEC (dd.dd)")
            y_u, x_u = int_hdu.data.shape[1:]
            input_ax.plot([0, x_u], [0, 0], 'k-')
            input_ax.plot([x_u, x_u], [0, y_u], 'k-')
            input_ax.plot([x_u, 0], [y_u, y_u], 'k-')
            input_ax.plot([0, 0], [y_u, 0], 'k-')
            input_ax.set_xlim([-5, x_u+5])
            input_ax.set_ylim([-5, y_u+5])
            #input_ax.plot(qXin,qYin,'ro')
            input_ax.set_xlabel("X")
            input_ax.set_ylabel("Y")
            y_u, x_u = coadd_data.shape[1:]
            coadd_ax.plot([0, x_u], [0, 0], 'r-')
            coadd_ax.plot([x_u, x_u], [0, y_u], 'r-')
            coadd_ax.plot([x_u, 0], [y_u, y_u], 'r-')
            coadd_ax.plot([0, 0], [y_u, 0], 'r-')
            coadd_ax.set_xlim([-0.5, x_u+1])
            coadd_ax.set_ylim([-0.5, y_u+1])
            for f_p in footprints[i:i+1]:
                sky_ax.plot(-f_p[0:2, 0], f_p[0:2, 1], 'k-')
                sky_ax.plot(-f_p[1:3, 0], f_p[1:3, 1], 'k-')
                sky_ax.plot(-f_p[2:4, 0], f_p[2:4, 1], 'k-')
                sky_ax.plot([-f_p[3, 0], -f_p[0, 0]], [f_p[3, 1], f_p[0, 1]], 'k-')
            for f_p in [coadd_fp]:
                sky_ax.plot(-f_p[0:2, 0], f_p[0:2, 1], 'r-')
                sky_ax.plot(-f_p[1:3, 0], f_p[1:3, 1], 'r-')
                sky_ax.plot(-f_p[2:4, 0], f_p[2:4, 1], 'r-')
                sky_ax.plot([-f_p[3, 0], -f_p[0, 0]], [f_p[3, 1], f_p[0, 1]], 'r-')

        # Loop through spatial pixels in this input frame
        for y_j in range(int_hdu.data.shape[1]):
            for x_k in range(int_hdu.data.shape[2]):

                #Get binary mask of good wavelength indices at this x,y position
                msk_jk = wavmask_i & ~np.isnan(int_hdu.data[:, y_j, x_k])

                # Define BL, TL, TR, BR corners of pixel as coordinates
                pix_verts = np.array([
                    [x_k - 0.5 + drz_f, y_j - 0.5 + drz_f],
                    [x_k - 0.5 + drz_f, y_j + 0.5 - drz_f],
                    [x_k + 0.5 - drz_f, y_j + 0.5 - drz_f],
                    [x_k + 0.5 - drz_f, y_j - 0.5 + drz_f]
                    ])

                # Convert these vertices to RA/DEC positions
                pix_verts_radec = wcs2d_i.all_pix2world(pix_verts, 0)

                # Convert the RA/DEC vertex values into coadd frame coordinates
                pix_verts_coadd = coadd_wcs.all_world2pix(pix_verts_radec, 0)

                #Create polygon object for projection onto coadd grid
                pix_projection = shapely_polygon(pix_verts_coadd)

                if plot:
                    input_ax.plot(pix_verts[:, 0], pix_verts[:, 1], 'kx')
                    sky_ax.plot(-pix_verts_radec[:, 0], pix_verts_radec[:, 1], 'kx')
                    coadd_ax.plot(pix_verts_coadd[:, 0], pix_verts_coadd[:, 1], 'kx')

                #Get bounds of pixel projection in coadd frame
                proj_bounds = list(pix_projection.exterior.bounds)

                # xb0 is x-bound-lower, yb1 is y-bound-upper, etc.
                xb0, yb0, xb1, yb1 = (int(round(pib)) for pib in proj_bounds)

                # Upper bounds need to be increased to include full pixel
                xb1 += 1
                yb1 += 1

                # Loop through relevant pixels in output/coadd frame ('coadd' denoted '_c')
                for x_c in range(xb0, xb1):
                    for y_c in range(yb0, yb1):

                        # Create shapely_polygon object for this coadd pixel
                        coadd_pixel = shapely_box(x_c - 0.5, y_c - 0.5, x_c + 0.5, y_c + 0.5)

                        # Get overlap between input pixel and this coadd pixel
                        overlap = pix_projection.intersection(coadd_pixel).area

                        # Convert to fraction of total input pixel area
                        overlap /= pix_projection.area

                        # Extract spectrum
                        spc_in = int_hdu.data[msk_jk, y_j, x_k]

                        # Extract variance
                        if usevar:
                            var_jk = var_hdus[i].data[msk_jk, y_j, x_k].copy()
                            var_jk *= (overlap**2)

                        #Update all relevant frames
                        try:
                            fract_frame[msk_jk, y_c, x_c] += overlap
                            build_frame[msk_jk, y_c, x_c] += overlap * spc_in
                            if usevar:
                                var_build_frame[msk_jk, y_c, x_c] += var_jk

                        except IndexError:
                            continue
                            #TODO - Fix the issue of edge pixels being trimmed
                            #Need to expand coadd canvas to include all pixels.
                if verbose:
                    pbar.update(1)

        if plot:
            fig2.canvas.draw()
            plt.waitforbuttonpress()

        #Calculate ratio of coadd pixel area to input pixel area
        px_area_ratio = coadd_px_area / px_area_i

        # Max value in fract_frame should be px_area_ratio; it's the biggest
        # fraction of an input pixel that can add to one coadd pixel
        # We want to use this map now to create a flat_frame - where the
        # values represent a covering fraction for each pixel
        # i.e. 0.1 = 10% of pixel area covered by input frames
        flat_frame = fract_frame / px_area_ratio

        #Replace zero-values with inf values to avoid division by zero
        flat_frame[flat_frame == 0] = np.inf

        # Perform flat field correction for pixels that are not fully covered
        build_frame /= flat_frame

        # Zero any pixels below user-set pixel threshold. Set flat value to inf
        build_frame[flat_frame < px_thresh] = 0

        # Propagate variance on previous two steps
        if usevar:
            var_build_frame /= (flat_frame)**2
            var_build_frame[flat_frame < px_thresh] = np.inf

        #Replace values < px_thresh with inf also
        flat_frame[flat_frame < px_thresh] = np.inf

        #Add to exposure mask
        coadd_exp += t_exp_i * (flat_frame < np.inf)

        # Add weight * data to coadd
        coadd_data += t_exp_i * build_frame

        # Propagate error on the above step
        if usevar:
            coadd_var += (t_exp_i**2) * var_build_frame

    if verbose:
        pbar.close()

    if plot:
        plt.close()

    utils.output("\tTrimming coadded canvas.\n")

    # Create 1D exposure time profiles
    exp_zprof = np.mean(coadd_exp, axis=(1, 2))
    exp_xprof = np.mean(coadd_exp, axis=(0, 1))
    exp_yprof = np.mean(coadd_exp, axis=(0, 2))

    # Normalize the profiles
    exp_zprof /= np.max(exp_zprof)
    exp_xprof /= np.max(exp_xprof)
    exp_yprof /= np.max(exp_yprof)

    # Convert 0s to +1NF in exposure time cube
    coadd_exp[coadd_exp == 0] = np.inf

    # Divide by total exposure time, or square for variance
    coadd_data /= coadd_exp
    if usevar:
        coadd_var /= coadd_exp**2

    #Exposure time threshold, relative to maximum exposure time, below which to crop.
    use_z = exp_zprof > exp_thresh
    use_x = exp_xprof > exp_thresh
    use_y = exp_yprof > exp_thresh

    #Trim the data
    coadd_data = coadd_data[use_z]
    coadd_data = coadd_data[:, use_y]
    coadd_data = coadd_data[:, :, use_x]

    #Trim variance
    if usevar:
        coadd_var = coadd_var[use_z]
        coadd_var = coadd_var[:, use_y]
        coadd_var = coadd_var[:, :, use_x]

    #Update the WCS to account for trimmed pixels
    coadd_hdr["CRPIX3"] -= np.argmax(use_z)
    coadd_hdr["CRPIX2"] -= np.argmax(use_y)
    coadd_hdr["CRPIX1"] -= np.argmax(use_x)

    # Create FITS object matching the input type (i.e. HDU or HDUList)
    coadd_fits = utils.match_hdu_type(int_hdus[0], coadd_data, coadd_hdr)

    if usevar:
        coadd_var_fits = utils.match_hdu_type(var_hdus[0], coadd_var, coadd_hdr)
        return coadd_fits, coadd_var_fits

    return coadd_fits
