"""CWITools Data Analysis Functions"""
from cwitools.libs import cubes

from astropy.nddata import Cutout2D
from astropy.convolution import Gaussian1DKernel,Box1DKernel,Gaussian2DKernel,convolve
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.modeling import models, fitting
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import SqrtStretch
from astropy.stats import sigma_clip
from photutils import DAOStarFinder
from scipy import ndimage
from scipy.signal import medfilt
from scipy.stats import tstd
from scipy.ndimage.filters import generic_filter
from scipy.stats import sigmaclip
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pyregion
import sys
import warnings

def get_momentmaps(fits, obj_cube=[], obj_ids=[], method="basic"):

    cube, hdr = fits[0].data, fits[0].header

    #Object cube not provided
    if obj_cube == []:
        if obj_ids == []:
            raise ValueError("obj_ids provided without obj_cube.")
        else:
            obj_mask = np.ones_like(cube) #By default, use entire cube

    #Object cube provided
    else:
        if obj_ids==[]:
            obj_mask = np.ones_like(cube)
        else:
            obj_mask = np.zeros_like(obj_cube)
            for obj_id in obj_ids: obj_mask[obj_cube == obj_id] = 1

    #Get 2D mask of object spaxels
    msk_2d = np.max(obj_mask, axis=0)
    msk_1d = np.max(obj_mask, axis=(1, 2))
    wav_obj = wav[msk_1d > 0]

    #Set non-object voxels to zero
    cube[obj_mask == 0] = 0

    #Create canvas for both first and zeroth (f,z) moments
    m1_map = np.zeros_like(msk_2d, dtype=float)
    m2_map = np.zeros_like(m1_map)

    #Initialize guess for first moment as center of z-mask
    try: mu1_guess = wav_obj[int(len(wav_obj)/2)]
    except: pass

    #Only perform calculation if object has any valid spaxels
    if np.count_nonzero(msk_2d)>0:

        #Calculate first moment
        m1_num = np.zeros_like(m1_map)
        m1_den = np.zeros_like(m1_map)

        for i in range(m1_map.shape[0]):
            for j in range(m1_map.shape[1]):

                if msk_2d[i,j]:

                    spc_ij = cube[msk_1d > 0, i, j] #Get 1D spectrum at (i,j) within z-mask

                    if args.method == 'closing-window':
                        m1_ij, m2_ij = libs.science.closing_window_moments(wav_obj, spc_ij, mu1_init=mu1_guess)
                    elif args.method == 'basic':
                        m1_ij, m2_ij = libs.science.basic_moments(wav_obj, spc_ij, pos_thresh=False)
                    elif args.method == 'positive':
                        m1_ij, m2_ij = libs.science.basic_moments(wav_obj, spc_ij, pos_thresh=True)

                    if np.isnan(m1_ij) or np.isnan(m2_ij) or m1_ij==-1:
                        msk_2d[i, j] = 0
                        continue

                    else:
                        m1_map[i,j] = m1_ij #Fill in to maps if valid
                        m2_map[i,j] = m2_ij



        #Calculate integrated spectrum

        spec_1d = np.sum(cube,axis=(1,2))
        spec_1d -= np.median(spec_1d)
        spec_1d = spec_1d[msk_1d>0]
        m1_ref = np.sum(wav_obj*spec_1d)/np.sum(spec_1d)

        #print("%30s  %10.3f %10.3f"%(args.cube.split('/')[0], m1_ref, disp_global_kms))
        #Convert moments to velocity space
        m1_map = 3e5*(m1_map - m1_ref)/m1_ref
        m2_map = 3e5*np.sqrt(m2_map)/m1_ref

    else: m1_ref = 0

    #Fill in empty or bad spaxels with fill value if selected
    if args.filltype == 'value':

        m1_map[msk_2d == 0] = args.fillvalue
        m2_map[msk_2d == 0] = args.fillvalue

    #Use NaNs if requested
    elif args.filltype == 'nan':

        m1_map[msk_2d == 0] = np.nan
        m2_map[msk_2d == 0] = np.nan

#Return a pseudo-Narrowband image (either SB units or SNR)
def pseudo_nb(fits_in, center, bandwidth, wlsub=True, pos=None, cwing=20,
            fitRad=2, subRad=None, maskPSF=True, smooth=None, reg=None, var=[]):

    """Create a pseudo-Narrow-Band (pNB) image from a data cube.

    Args:
        fits_in (astropy.io.fits.HDUList): The input FITS file.
        center (float): The central wavelength of the narrow-band in Angstrom.
        bandwidth (float): The width of the narrow-band image in Angstorm.
        wlsub (bool): TRUE = subtract continuum emission.
            If a source is provided with the pos argument, it will be used to
            scale the white-light image for subtraction. Otherwise the entire
            image will be used to find a scaling factor.
        pos (float tuple): The location the main continuum source in x,y.
        fitRad (float): The radius around the source to use for fitting.
        subRad (float): The radius around the soruce to use when subtracting.
        maskPSF (bool): TRUE = mask source after white-light subtraction.
        reg (string): Path to DS9 region file of continuum sources to mask.
        smooth (float): FWHM of a Gaussian kernel to use for smoothing the NB.


    Returns:
        numpy.ndarray: A white-light image with the given center/width
        numpy.ndarray: The PSF model which was subtracted from the data
        numpy.ndarray: The mask used to cover foreground sources
        numpy.ndarray: The pseudo-narrowband image, in surface-brightness units.

    """

    cube,header = fits_in[0].data, fits_in[0].header

    if var==[]: usevar=False
    else: usevar=True

    #Prep: get some useful structures numbers
    hdr2D = cubes.get_header2d(header)
    wcs2D = WCS(hdr2D) #Astropy world-coord sys
    pxScls = (getPxScales(wcs2D)*u.degree).to(u.arcsec)
    pxArea = pxScls[0]*pxScls[1] #Pixel size in arcsec2
    wavbin  = header["CD3_3"] #Spectral plate scale in angstrom/px
    halfwidth = bandwidth/2.0

    xQ, yQ = pos

    #Create plain narrow-band
    A,B = cubes.get_indices(center-halfwidth, center+halfwidth, header)

    #If requested NB is not in range of cube, return none type
    if B<=0 or A>=cube.shape[0]-1: return np.zeros_like(cube[0])
    #If it is clipped on left, adjust size
    elif A<0: A = 0
    #Clipped on right, adjust size
    elif B>cube.shape[0]-1: B=-1

    #Get indices of WL image
    a,b = cubes.get_indices(center-bandwidth-cwing,center+bandwidth+cwing,header)

    #Get WL image in SB units
    WL = np.sum(cube[a:A],axis=0) + np.sum(cube[B:b],axis=0)
    WL *= wavbin/pxArea

    #Create narrowband image
    NB = np.sum(cube[A:B], axis=0)
    NB *= wavbin/pxArea

    if usevar:

        #White-light variance image
        WL_var = np.sum(var[a:A],axis=0) + np.sum(var[B:b],axis=0)
        WL_var *= (wavbin/pxArea)**2

        #NB variance image
        NB_var = np.sum(var[A:B], axis=0)
        NB_var *= (wavbin/pxArea)**2

    #fMask is the 2D mask of pixels used for fitting (by default the whole NB)
    fMask = np.ones_like(cube[0],dtype=bool)
    sMask = np.ones_like(cube[1],dtype=bool)

    #Now build up source mask and model if reg file given
    if reg != None:

        box_rad = 3
        fit_rad = 2
        psf_fitter = fitting.LevMarLSQFitter()

        reg_wcs = pyregion.open(reg)
        reg_img = reg_wcs.as_imagecoord(header=hdr2D)

        hdu2D = fits.PrimaryHDU(NB)
        hdu2D.header = hdr2D

        src_model = np.zeros_like(WL)
        src_mask = np.zeros_like(WL) #Mask of continuum (foreground) sources

        qso_mask = np.zeros_like(WL) #Keep QSO mask separate till the end

        yy, xx = np.indices(WL.shape)

        for i, src in enumerate(reg_wcs):

            x, y, rad = reg_img[i].coord_list
            x -= 1 #Adjust to 0-indexing
            y -= 1

            #Get distance mesh
            rr = np.sqrt( (xx-x)**2 + (yy-y)**2 )

            #Get distance from QSO
            dist = np.sqrt( (x-xQ)**2 + (y-yQ)**2)

            #Cast to integers for indexing
            y = int(round(y))
            x = int(round(x))

            #If outside FOV, ignore
            if np.min(rr) > 1: continue

            elif rad > 6: src_mask[ rr < rad ] = 1

            #Otherwise, process:
            else:

                #Extract box around source

                boxL, boxR = max(0, y-box_rad), min(WL.shape[0]-1, y+box_rad+1)
                boxB, boxT = max(0, x-box_rad), min(WL.shape[1]-1, x+box_rad+1)
                box = WL[boxL:boxR, boxB:boxT]

                #Get indices/axes of box
                box_xx, box_yy = np.indices(box.shape, dtype=float)
                box_xx -= (boxT-boxB)/2.0
                box_yy -= (boxR-boxL)/2.0

                #Set bounds on model
                fit_bounds = {'amplitude':(0, 5*np.max(box)),
                                 'x_mean':(-fit_rad, fit_rad),
                                 'y_mean':(-fit_rad, fit_rad),
                                 'x_stddev':(1.5, 4),
                                 'y_stddev':(1.5, 4)
                                }

                #make initial guess
                model_guess = models.Gaussian2D(amplitude=np.max(box),
                                              x_mean = 0,
                                              y_mean = 0,
                                              x_stddev = 2,
                                              y_stddev = 2,
                                              bounds = fit_bounds

                )

                #Fit model to data
                model_fit = psf_fitter(model_guess, box_yy, box_xx, box)

                #Adjust center of PSF back to global coords
                model_fit.x_mean += x
                model_fit.y_mean += y

                #Create elliptical source mask from fitted PSF
                mask_model = models.Ellipse2D(x_0 = model_fit.x_mean,
                                        y_0 = model_fit.y_mean,
                                        theta = model_fit.theta,
                                        a = 2*model_fit.x_stddev,
                                        b = 2*model_fit.y_stddev
                )

                #If this is the central source, add to QSO mask
                if dist < 1:

                    #Set fitting mask for later
                    fMask = rr<=fitRad

                    #If subtraction radius given, update sMask
                    if subRad!=None: sMask = rr<=subRad


                    #Add to QSO mask
                    qso_mask += mask_model(xx, yy)

                else:

                    #Add to source model and mask
                    src_model += model_fit(xx, yy)
                    src_mask  += mask_model(xx, yy)

        src_mask[src_mask > 0] = 1
        src_mask[qso_mask > 0] = 2 #2 indicates QSO or central source

    else:
        src_model = np.zeros_like(NB)
        src_mask = np.zeros_like(NB)


    #Median subtract both
    # NB -= np.median(NB[ src_mask == 0 ])
    # WL -= np.median(WL[ src_mask == 0 ])

    # # Vertical median correction
    # for xi in range(NB.shape[1]):
    #     NB[:, xi] -= np.median( NB[src_mask[:, xi]==0, xi] )
    #     WL[:, xi] -= np.median( WL[src_mask[:, xi]==0, xi] )
    #
    # # # Horizontal median correction
    for yi in range(NB.shape[0]):
        NB[yi, :] -= np.median( sigmaclip(NB[yi, src_mask[yi, :]==0 ], low=3, high=3).clipped )
        WL[yi, :] -= np.median( sigmaclip(WL[yi, src_mask[yi, :]==0 ], low=3, high=3).clipped)



    #If smoothing requested
    if smooth!=None:
        NB = smooth2d(NB, smooth, ktype='gaussian')
        WL = smooth2d(WL, smooth, ktype='gaussian')
        # NB_sub = smooth2d(NB_sub, smooth, ktype='gaussian')
        # NB_sub_var = smooth2d(NB_sub_var, smooth, ktype='gaussian', var=True)
        NB_var =  smooth2d(NB_var, smooth, ktype='gaussian', var=True)



    scalingFactors = NB[fMask]/WL[fMask]
    scalingFactors_Clipped = sigmaclip(scalingFactors,high=2.5,low=2.5)
    scalingFactors_Mean = np.median(scalingFactors)
    S = scalingFactors_Mean

    sMask[WL<=0] = 1 #Do not subtract negative values
    sMask[src_mask == 1] = 0 #Do not subtract over other sources
    #Subtract WL image

    NB_sub = NB.copy()
    NB_sub[sMask] -=  S*WL[sMask]
    NB_sub[src_mask == 1] = 0
    NB_sub[fMask == 1] = 0

    NB_sub_var = NB_var.copy()
    NB_sub_var[sMask] += (S**2)*WL_var[sMask]

    NB_sub = noisefit_bgsubtract(NB_sub, src_mask)


    NB_sub = noisefit_bgsubtract(NB_sub, src_mask)
    NB_sub[src_mask == 1] = 0

    #Scale variance to match noise in NB image
    snr = NB_sub/np.sqrt(NB_sub_var)
    n, edges = np.histogram(snr, range=(-5,5), bins=40)
    centers = np.array([ (edges[i]+edges[i+1])/2.0 for i in range(edges.size-1)])
    fitter = fitting.SimplexLSQFitter()
    fit_inds = centers <= 1
    snrmodel0 = models.Gaussian1D(amplitude=n.max(), mean=0, stddev=1)
    snrmodel1 = fitter(snrmodel0, centers[fit_inds], n[fit_inds])


    NB_sub_var *= (snrmodel1.stddev.value**2)

    print(snrmodel1.stddev.value**2)
    if 0:
        snr2 = NB_sub/np.sqrt(NB_sub_var)
        n2, edges2 = np.histogram(snr2, range=(-5,5), bins=40)
        centers2 = np.array([ (edges2[i]+edges2[i+1])/2.0 for i in range(edges2.size-1)])
        snrmodel2 = models.Gaussian1D(amplitude=n2.max(), mean=0, stddev=1)
        plt.figure()
        plt.plot(centers, n, 'k.')
        plt.plot(centers, snrmodel0(centers), 'g-')
        plt.plot(centers, snrmodel1(centers), 'r-')
        plt.plot(centers2, n2, 'b.')
        plt.plot(centers, snrmodel2(centers), 'b-')
        plt.show()



    #Return SB map
    return WL, NB, NB_sub, NB_sub_var, src_mask

#Function to smooth along wavelength axis
def smooth3d(cube,scale,axes=(0,1,2),ktype='gaussian',var=False):
    """Smooth along all/any axes of a data cube with a box or gaussian kernel.

    Args:
        cube (numpy.ndarray): The input datacube.
        scale (float): The smoothing scale.
            For a gaussian kernel, this is full-width at half-maximum (FWHM)
            For a box kernel, this is the width of the box.
        axes (int tuple): The axes to smooth along.
        ktype (str): The kernel type ('gaussian' or 'box')
        var (bool): Set to TRUE when smoothing variance data.

    Returns:
        numpy.ndarray: The smoothed data cube.

    """
    #Make copy - do not modify input cube directly
    cubeFilt = cube.copy()

    #Set kernel type
    if ktype=='box':
        kernel = Box1DKernel(scale)
    elif ktype=='gaussian':
        sig = fwhm2sigma(scale)
        kernel = np.array(Gaussian1DKernel(sig))
        kernel_vol =  np.sum(kernel)#2*np.pi*np.sqrt(np.max(kernel))*(sig**2)

    else: raise ValueError("Kernel type ('%s') not found"%ktype)

    kernel = np.array(kernel)

    if var: kernel = np.power(kernel,2)

    #Apply kernel
    for a in axes:
        cubeFilt = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'),
                                       axis=a,
                                       arr=cubeFilt.copy()
        )

    #Re-normalize data

    #if var=True: kernel

    #Return
    return cubeFilt
        
def get_box_(fitsIN, paramsIN, boxSize, fill=0):

    wcs2D = WCS(cubes.get_header2d(fitsIN[0].header))

    xC,yC = wcs2D.all_world2pix(paramsIN["RA"],paramsIN["DEC"],0)

    pkpc_per_px = get_pkpc_px(wcs2D,paramsIN["Z"])
    boxSize_px = boxSize/pkpc_per_px

    NBCutout = Cutout2D(fitsIN[0].data,(xC,yC),boxSize_px,wcs2D,mode='partial',fill_value=fill)

    return NBCutout.data

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
        >>> from cwitools.analysis import rebin
        >>> myfits = fits.open("mydata.fits")
        >>> binned_fits = rebin(myfits, zbin = 4)
        >>> binned_fits.writeto("mydata_binned.fits")


    """


    #Extract useful structures
    data = inputfits[0].data.copy()
    head = inputfits[0].header.copy()

    #Get dimensions & Wav array
    z, y, x = data.shape
    wav = cubes.get_wavaxis(head)

    #Get new sizes
    znew = int(z/zbin)  + 1 if zbin > 1 else z
    ynew = int(y/xybin) + 1 if xybin > 1 else y
    xnew = int(x/xybin) + 1 if xybin > 1 else x

    #Perform wavelenght-binning first, if bin provided
    if zbin > 1:

        #Get new bin size in Angstrom
        zbinSize = zbin*head["CD3_3"]

        #Create new data cube shape
        data_zbinned = np.zeros((znew, y, x))

        #Run through all input wavelength layers and add to new cube
        for zi in range(z): data_zbinned[int(zi/zbin)] += data[zi]

        #Normalize so that units remain as "erg/s/cm2/A"
        if vardata: data_zbinned /= zbin**2
        else: data_zbinned /= zbin

        #Update central reference and pixel scales
        head["CD3_3"] *= zbin
        head["CRPIX3"] /= zbin

    else: data_zbinned = data

    #Perform spatial binning next
    if xybin > 1:

        #Get new shape
        data_xybinned = np.zeros((znew, ynew, xnew))

        #Run through spatial pixels and add
        for yi in range(y):
            for xi in range(x):
                data_xybinned[:, int(yi/xybin), int(xi/xybin)] += data_zbinned[:, yi, xi]

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

    binnedFits = fits.HDUList([fits.PrimaryHDU(data_zbinned)])
    binnedFits[0].header = head

    return binnedFits

def psf_subtract(inputfits, rmin=1.5, rmax=5.0, reg=None, pos=None, recenter=True, \
 auto=7, wl_window=200, local_window=0, scalemask=1.0, zmasks=((0,0)), zunit='A', verbose=False):
    """Models and subtracts point-sources in a 3D data cube.

    Args:
        inputfits (astrop FITS object): Input data cube/FITS.
        rmin (float): Inner radius, used for fitting PSF.
        rmax (float): Outer radius, used to subtract PSF.
        reg (str): Path to a DS9 region file containing sources to subtract.
        pos (float tuple): (x,y) position of the source to subtract.
        auto (float): SNR above which to automatically detect/subtract sources.
            Note: One of the parameters reg, pos, or auto must be provided.
        wl_window (int): Size of white-light window (in Angstrom) to use.
            This is the window used to form a white-light image centered
            on each wavelength layer. Default: 200A.
        local_window (int): Size of local window (in Angstrom) to use.
            This is the window used around each wavelength layer to form the
            local narrowband image. Default: 0 (i.e. single layer only)
        scalemask (float): Scaling factor for output PSF mask (Default: 1)
        zmask (int tuple): Wavelength region to exclude from white-light images.
        zunit (str): Unit of argument zmask.
            Can be Angstrom ('A') or pixels ('px'). Default: 'A'.
        verbose (bool): Set to True to display progress and information.

    Returns:
        numpy.ndarray: PSF-subtracted data cube
        numpy.ndarray: PSF model cube
        numpy.ndarray: 2D mask of sources

    Raises:
        FileNotFoundError: If region file is not found.

    Examples:

        To subtract point sources from an input cube using a DS9 region file:

        >>> from astropy.io import fits
        >>> from cwitools.analysis import psf_subtract
        >>> myregfile = "mysources.reg"
        >>> myfits = fits.open("mydata.fits")
        >>> sub_cube, psf_model, mask_2d = psf_subtract(myfits, reg = myregfile)

        To subtract using automatic source detection with photutils, and a
        source S/N ratio >5:

        >>> sub_cube, psf_model, mask_2d = psf_subtract(myfits, auto = 5)

        Or to subtract a single source from a specific location (x,y)=(21.1,34.6):

        >>> sub_cube, psf_model, mask_2d = psf_subtract(myfits, pos=(21.1, 34.6))

    """

    #Open fits image and extract info

    cube = inputfits[0].data
    header = inputfits[0].header
    w, y, x = cube.shape
    W, Y, X = np.arange(w), np.arange(y), np.arange(x)
    wav = cubes.get_wavaxis(header)

    #Remove NaN values
    cube = np.nan_to_num(cube,nan=0.0,posinf=0,neginf=0)

    #Create cube for subtracted cube and for model of psf
    psf_cube = np.zeros_like(cube)
    sub_cube = cube.copy()

    #Make mask canvas
    msk2D = np.zeros((y, x))

    #Create white-light image
    maskwav = np.zeros_like(wav, dtype=bool)
    for (z0, z1) in zmasks:
        print(z0,z1)
        if zunit == 'A': z0, z1 = cubes.get_indices(z0, z1, header)
        maskwav[z0:z1] = 1

    wlcube = cube.copy()
    wlcube[maskwav] = 0
    wlImg = np.sum(wlcube, axis=0)

    #Get WCS information
    wcs = WCS(header)
    pxScales = proj_plane_pixel_scales(wcs)

    #Get mask of spaxels with no data
    emptyPxMask2D = (np.sum(cube, axis=0) == 0)

    #Convert plate scale to arcseconds
    xScale, yScale = (pxScales[:2]*u.deg).to(u.arcsecond)
    zScale = (pxScales[2]*u.meter).to(u.angstrom)

    #Convert fitting & subtracting radii from arcsecond/Angstrom to pixels
    rmin_px = rmin/xScale.value
    rmax_px = rmax/xScale.value
    delZ_px = int(round(0.5*wl_window/zScale.value))

    #Get fitter for PSF fit
    boxSize = 3*int(round(rmax_px))
    psfFitter = fitting.LevMarLSQFitter()
    psfModel = models.Gaussian2D(amplitude=1, x_mean=boxSize/2, y_mean=boxSize/2)

    #Get box size and indices for fitting
    yy, xx = np.mgrid[:boxSize, :boxSize]

    #Get sources from region file or position input
    sources = []
    if reg != None:

        if verbose: print("Using region file to locate sources.")

        if os.path.isfile(reg): regFile = pyregion.open(reg)
        else:
            raise FileNotFoundError("Region file could not be found: %s"%reg)

        for src in regFile:
            ra, dec, pa = src.coord_list
            xP, yP, wP = wcs.all_world2pix(ra, dec, header["CRVAL3"], 0)
            sources.append((xP, yP))

    elif pos != None:
        if verbose: print("Using provided source position.")
        sources = [pos]

    else:

        if verbose: print("Automatically detecting sources with photutils. (SNR>%.1f)"%auto)

        stddev = np.std(wlImg[wlImg <= 10*np.std(wlImg)])

        #Run source finder
        daofind = DAOStarFinder(fwhm=8.0, threshold=auto*stddev)
        autoSrcs = daofind(wlImg)

        #Get list of peak values
        peaks = list(autoSrcs['peak'])

        #Make list of sources
        sources = []
        for i in range(len(autoSrcs['xcentroid'])):
            sources.append((autoSrcs['xcentroid'][i], autoSrcs['ycentroid'][i]))

        #Sort according to peak value (this will be ascending)
        peaks, sources = zip(*sorted(zip(peaks, sources)))

        #Cast back to list
        sources = list(sources)

        #Reverse to get descending order (brightest sources first)
        sources.reverse()

    if verbose:
        print("Subtracting %i source(s)."%len(sources))
        pbar = tqdm(total=len(sources))

    #Run through sources
    for (xP, yP) in sources:

        #Get meshgrid of distance from P
        YY, XX = np.meshgrid(X-xP, Y-yP)
        RR = np.sqrt(XX**2 + YY**2)

        if np.min(RR) > rmin_px: continue
        else:

            #Get cut-out around source
            psfBox = Cutout2D(wlImg, (xP, yP), (boxSize, boxSize), mode='partial', fill_value=-99)
            psfBox = psfBox.data

            #Get useable spaxels
            fitXY = np.array(psfBox != -99, dtype=int)

            #Run fit
            psfFit = psfFitter(psfModel, yy, xx, psfBox, weights=fitXY)

            #Get sigma/fwhm
            xfwhm, yfwhm = 2.355*psfFit.x_stddev.value, 2.355*psfFit.y_stddev.value

            #We take larger of the two for our purposes
            fwhm = max(xfwhm, yfwhm)

            #Only continue with well-fit, high-snr sources
            if 1 or (psfFitter.fit_info['nfev'] < 100 and fwhm < 10):

                if recenter:
                    yP = psfFit.x_mean.value+yP-boxSize/2
                    xP = psfFit.y_mean.value+xP-boxSize/2

                #Update meshgrid of distance from P
                YY, XX = np.meshgrid(X-xP, Y-yP)
                RR = np.sqrt(XX**2 + YY**2)

                #Get half-width-half-max
                hwhm = fwhm/2.0

                #Add source to mask
                msk2D[RR <= scalemask*hwhm] = 1

                #Get boolean masks for
                fitPx = RR <= rmin_px
                subPx = (RR <= rmax_px) & (~emptyPxMask2D)

                meanRs = []
                #Run through wavelength layers
                for wi in range(w):

                    #Get this wavelength layer and subtract any median residual
                    wl1, wl2 = max(0, wi-local_window), min(w, wi+local_window)+1
                    layer = np.mean(cube[wl1:wl2], axis=0)

                    #Get upper and lower-bounds for creating WL image
                    a = max(0, wi-delZ_px)
                    b = min(w, a+delZ_px)

                    #Create PSF image
                    psfImg = np.sum(wlcube[a:b], axis=0)
                    psfImg[psfImg < 0] = 0

                    scalingFactors = layer[fitPx]/psfImg[fitPx]
                    scalingFactors_Clipped = sigmaclip(scalingFactors, high=3.5, low=3.5)
                    scalingFactors_Mean = np.mean(scalingFactors)
                    A = scalingFactors_Mean
                    if A < 0 or np.isinf(A) or np.isnan(A): A = 0

                    #Subtract fit from data
                    sub_cube[wi][subPx] -= A*psfImg[subPx]

                    #Add to PSF model
                    psf_cube[wi][subPx] += A*psfImg[subPx]

                #Update WL cube and image after subtracting this source
                wlImg = np.sum(sub_cube[maskwav == 0], axis=0)

        if verbose: pbar.update(1)

    if verbose: pbar.close()

    return sub_cube, psf_cube, msk2D

def estimate_variance(inputfits, zwindow=10, rescale=True, sigmaclip=4, zmask=(0, 0), fmin=0.9, fmax=10):
    """Estimates the 3D variance cube of an input cube.

    Args:
        inputfits (astropy.io.fits.HDUList): FITS object to estimate variance of.
        zWindow (int): Size of z-axis bins to use for 2D variance estimation. Default: 10.
        rescale (bool): Set to TRUE to perform layer-by-layer rescaling of 2D variance.
        sigmaclip (float): Threshold (in stddevs) for sigma-clipping data before estimation.
        zmask (int tuple): Wavelength layers to exclude while estimating variance.
        fMin (float): The minimum rescaling factor (Default 0.9)
        fMax (float): The maximum rescaling factor (Default: 10)
        fileExt (str): The extension to use for the output cube (Default .var.fits)

    Returns:

        NumPy ndarray: Estimated variance cube

    Examples:

        >>> from astropy.io import fits
        >>> from cwitools.analysis import estimate_variance
        >>> myfits = fits.open("mydata.fits")
        >>> varcube = estimate_variance(myfits)
        >>> varfits = fits.HDUList([fits.primaryHDU(varcube)])
        >>> varfits[0].header = myfits[0].header
        >>> varfits.writeto("mydata_var.fits")

    """

    cube = inputfits[0].data
    z0, z1 = zmask
    dz = zwindow

    #Output warning
    if z1-z0 >= dz:
        warnings.warn("""Your z-mask is large relative to your zwindow size.\
        \nVariance estimate may be unreliable.""")

    #Parse boolean input
    rescale = True if rescale == "True" else False

    #Run sigma-clip if set
    if sigmaclip > 0: cube = sigma_clip(cube, sigma=sigmaclip).data

    #Make first estimate by binning data
    varcube = np.zeros_like(cube)
    i = 0
    a, b = (i*dz), (i+1)*dz
    while b < cube.shape[0]:
        varcube[a:b] = np.var(cube[a:b], axis=0)
        i += 1
        a, b = (i*dz), (i+1)*dz
    varcube[a:] = np.var(cube[a:], axis=0)

    #Adjust first estimate by rescaling, if set to do so
    if rescale:
        for wi in range(len(varcube)):

            sig = np.sqrt(varcube[wi])

            useXY = sig > 0

            varNorm = np.var(cube[wi][useXY]/sig[useXY])

            #Normalize so that variance of layer as a whole is ~1
            #
            # Note: this assumes most of the 3D field is empty of real signal.
            # Z and XY Masks should be supplied if that is not the case
            #

            rsFactor = (1/varNorm)

            rsFactor = max(rsFactor, fmin)
            rsFactor = min(rsFactor, fmax)

            varcube[wi] *= rsFactor

    return varcube


def bg_subtract(inputfits, method='polyfit', poly_k=1, median_window=31, zmasks=[(0, 0)], zunit='A'):
    """
    Subtracts extended continuum emission / scattered light from a cube

    Args:
        cubePath (str): Path to the data cube to be subtracted.
        method (str): Which method to use to model background
            'polyfit': Fits polynomial to the spectrum in each spaxel (default.)
            'median': Subtract the spatial median of each wavelength layer.
            'medfilt': Model spectrum in each spaxel by median filtering it.
            'noiseFit': Model noise in each z-layer and subtract mean.
        poly_k (int): The degree of polynomial to use for background modeling.
        median_window (int): The filter window size to use if median filtering.
        zmask (int tuple): Wavelength region to mask, given as tuple of indices.
        zunit (str): If using zmask, indices are given in these units.
            'A': Angstrom (default)
            'px': pixels
        saveModel (bool): Set to TRUE to save background model cube.
        fileExt (str): File extension to use for output (Default: .bs.fits)

    Returns:
        NumPy.ndarray: Background-subtracted cube
        NumPy.ndarray: Cube containing background model which was subtracted.

    Examples:

        To model the background with 1D polynomials for each spaxel's spectrum,
        using a quadratic polynomial (k=2):

        >>> from cwitools.analysis import bg_subtract
        >>> from astropy.io import fits
        >>> myfits = fits.open("mydata.fits")
        >>> bgsub_cube, bgmodel_cube = bg_subtract(myfits, method='polfit', poly_k=2)

    """

    #Load header and data
    header = inputfits[0].header.copy()
    cube = inputfits[0].data.copy()
    W = cubes.get_wavaxis(header)
    z, y, x = cube.shape
    xySize = cube[0].size
    maskZ = False
    modelC = np.zeros_like(cube)

    #Get empty regions mask
    mask2D = np.sum(cube, axis=0) == 0

    #Convert zmask to pixels if given in angstrom
    useZ = np.ones_like(W, dtype=bool)
    for (z0, z1) in zmasks:
        print(z0,z1)
        if zunit == 'A': z0, z1 = cubes.get_indices(z0, z1, header)
        useZ[z0:z1] = 0

    #Subtract background by fitting a low-order polynomial
    if method == 'polyfit':

        fitter = fitting.LevMarLSQFitter()
        pModel0 = models.Polynomial1D(degree=poly_k)

        #Track progress % using n
        n = 0

        #Run through spaxels and subtract low-order polynomial
        for yi in range(y):
            for xi in range(x):

                n += 1
                p = 100*float(n)/xySize
                sys.stdout.write('%5.2f percent complete\r'%p)
                sys.stdout.flush()

                #Extract spectrum at this location
                spectrum = cube[:, yi, xi].copy()

                #Fit polynomial to data, ignoring masked pixels
                pModel1 = fitter(pModel0, W[useZ], spectrum[useZ])

                #Get background model
                bgModel = pModel1(W)

                if mask2D[yi, xi] == 0:

                    cube[:, yi, xi] -= bgModel

                    #Add to model
                    modelC[:, yi, xi] += bgModel

    #Subtract background by estimating it with a median filter
    elif method == 'medfilt':

        #Get +/- 5px windows around masked region, if mask is set
        if z1 > 0:

            #Get size of window region used to interpolate (minimum 5 to get median)
            nw = max(5, (z1-z0))

            #Get left and right index of window regions
            a = max(0, z0-nw)
            b = min(z, z1+nw)

            #Get two z mid-points which we will use for calculating line slope/intercept
            ZA = (a+z0)/2.0
            ZB = (b+z1)/2.0

            maskZ = True

        #Track progress % using n
        n = 0

        for yi in range(y):
            for xi in range(x):
                n += 1
                p = 100*float(n)/xySize
                sys.stdout.write('%5.2f percent complete\r'%p)
                sys.stdout.flush()

                #Extract spectrum at this location
                spectrum = cube[:, yi, xi].copy()

                #Fill in masked region with smooth linear interpolation
                if maskZ:

                    #Calculate slope and intercept
                    YA = np.mean(spectrum[a:z0]) if (z0-a) < 5 else np.median(spectrum[a:z0])
                    YB = np.mean(spectrum[z1:b]) if (b-z1) < 5 else np.median(spectrum[z1:b])
                    m = (YB-YA)/(ZB-ZA)
                    c = YA - m*ZA

                    #Get domain for masked pixels
                    ZZ = np.arange(z0, z1+1)

                    #Apply mask
                    spectrum[z0:z1+1] = m*ZZ + c

                #Get median filtered spectrum as background model
                bgModel = generic_filter(spectrum, np.median, size=median_window, mode='reflect')

                if mask2D[yi, xi] == 0:

                    #Subtract from data
                    cube[:, yi, xi] -= bgModel

                    #Add to model
                    modelC[:, yi, xi] += bgModel

    #Subtract layer-by-layer by fitting noise profile
    elif method == 'noiseFit':
        fitter = fitting.SimplexLSQFitter()
        medians = []
        for zi in range(z):

            #Extract layer
            layer = cube[zi]
            layerNonZ = layer[~mask2D]

            #Get median
            median = np.median(layerNonZ)
            stddev = np.std(layerNonZ)
            trimmed_stddev = tstd(layerNonZ, limits=(-3*stddev, 3*stddev))
            trimmed_median = np.median(layerNonZ[np.abs(layerNonZ-median) < 3*trimmed_stddev])

            medians.append(trimmed_median)
        medians = np.array(medians)
        bgModel0 = models.Polynomial1D(degree=2)

        bgModel1 = fitter(bgModel0, W[useZ], medians[useZ])
        for i, wi in enumerate(W):
            cube[i][~mask2D] -= bgModel1(wi)
            modelC[i][~mask2D] = bgModel1(wi)

    #Subtract using simple layer-by-layer median value
    elif method == "median":

        sigclip = SigmaClip(sigma=2)
        dataclipped = sigclip(cube)

        for zi in range(z):
            modelC[zi][mask2D == 0] = np.median(dataclipped[zi][mask2D == 0])
        modelC = medfilt(modelC, kernel_size=(3, 1, 1))
        cube -= modelC

    return cube, modelC

#Function to smooth along wavelength axis
def smooth2d(img, scale, ktype='gaussian', var=False):
    """Smooth along all/any axes of a data cube with a box or gaussian kernel.

    Args:
        cube (numpy.ndarray): The input datacube.
        scale (float): The smoothing scale.
            For a gaussian kernel, this is full-width at half-maximum (FWHM)
            For a box kernel, this is the width of the box.
        axes (int tuple): The axes to smooth along.
        ktype (str): The kernel type ('gaussian' or 'box')
        var (bool): Set to TRUE when smoothing variance data.

    Returns:
        numpy.ndarray: The smoothed data cube.

    """
    #Make copy - do not modify input cube directly
    imgFilt = img.copy()

    #Set kernel type
    if ktype=='box': kernel = Box2DKernel(scale)
    elif ktype=='gaussian': kernel = Gaussian2DKernel(fwhm2sigma(scale))
    else: raise ValueError("Kernel type ('%s') not found"%ktype)

    print(np.sum(np.array(kernel)))

    if var: kernel = np.power(np.array(kernel), 2)

    imgFilt = convolve(img, kernel)

    #Return
    return imgFilt
