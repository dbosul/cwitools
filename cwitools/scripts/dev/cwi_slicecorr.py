from cwitools.libs import params,science,cubes
from astropy.wcs import WCS
from astropy import units as u
from photutils import DAOStarFinder
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.modeling import models, fitting
from scipy.signal import find_peaks

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

matplotlib.use('TkAgg')

def slicecorr(inputFits, instrument="KCWI", src_snr=5):

    cube = inputFits[0].data.copy()
    header = inputFits[0].header
    Nw, Ny, Nx = cube.shape
    X, Y = np.arange(Nx), np.arange(Ny)
    xx, yy = np.meshgrid(X, Y)
    fit_rad_arcsec = 1
    box_rad_arcsec = 2

    wcs2D  = WCS(cubes.get_header2d(header))
    px_scl = proj_plane_pixel_scales(wcs2D)
    px_scl_x = (px_scl[0]*u.deg).to(u.arcsec).value
    px_scl_y = (px_scl[1]*u.deg).to(u.arcsec).value

    box_rad_x = int(round(box_rad_arcsec/px_scl_x))
    box_rad_y = int(round(box_rad_arcsec/px_scl_y))
    fit_rad_x = int(round(fit_rad_arcsec/px_scl_x))
    fit_rad_y = int(round(fit_rad_arcsec/px_scl_y))

    psf_fitter = fitting.LevMarLSQFitter()

    smthscale = 1.5
    cube[np.isnan(cube)] = 0
    wl_img = np.sum(cube,axis=0)
    wl_img = science.smooth3d(wl_img, smthscale, axes=(0,1))
    wl_img -= np.median(wl_img)

    wl_std = np.std(wl_img)
    wl_thresh = src_snr*wl_std

    sharplow = 0.0
    roundlow = -5.0
    roundhi = 5

    #Run source finder
    if instrument == "KCWI":

        kcwi_theta = 90 #PA of in-slice axis (KCWI slices run vertically)
        kcwi_aspect_ratio = abs(px_scl[1]/px_scl[0]) #Size of slice pixel vs in-slice pixel
        fwhm = smthscale*(1.3/3600.0)/px_scl[1] #Roughly 1'' FWHM is typical for keck

        starfinder = DAOStarFinder(wl_thresh, fwhm, roundlo=-5, roundhi=5)

    elif instrument == "PCWI":

        pcwi_theta = 0 #PA of in-slice axis (PCWI slices run horizontally)
        pcwi_aspect_ratio = abs(px_scl[0]/px_scl[1]) #Size of slice pixel vs in-slice pixel
        fwhm = smthscale*(1.75/3600.0)/px_scl[0] #Roughly 1.75'' FWHM is typical for Palomar

        starfinder = DAOStarFinder(wl_thresh, fwhm, ratio=pcwi_aspect_ratio, theta=pcwi_theta,
                                   sharplo=sharplow, roundlo=roundlow)

    else: raise ValueError("Instrument (%s) not recognized."%instrument)


    auto_sources = starfinder(wl_img)
    N_src = len(auto_sources)
    src_mask = np.zeros_like(wl_img, dtype=int)
    for i, src in enumerate(auto_sources):

        x, y = src['xcentroid'], src['ycentroid']

        xint = int(round(x))
        yint = int(round(y))

        #Get distance mesh
        rr = np.sqrt( (yy-y)**2 + (xx-x)**2 )

        #Extract box around source

        boxL, boxR = max(0, yint-box_rad_y), min(wl_img.shape[0]-1, yint+box_rad_y+2)
        boxB, boxT = max(0, xint-box_rad_x), min(wl_img.shape[1]-1, xint+box_rad_x+2)

        box = wl_img[boxL:boxR, boxB:boxT]
        box_yy, box_xx = np.indices(box.shape, dtype=float)
        box_yy -= (boxR-boxL)/2.0
        box_xx -= (boxT-boxB)/2.0

        box_msk = np.zeros_like(wl_img)
        box_msk[boxL:boxR, boxB:boxT] = 1



        #Set bounds on model
        fit_bounds = {'amplitude':(0, 5*np.max(box)),
                         'x_mean':(-fit_rad_x, fit_rad_x),
                         'y_mean':(-fit_rad_y, fit_rad_y),
                         'x_stddev':(0.5, 4),
                         'y_stddev':(0.5, 4)
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
        model_fit = psf_fitter(model_guess, box_xx, box_yy, box)


        #Create elliptical source mask from fitted PSF
        src_mask_model = models.Ellipse2D(x_0 = model_fit.x_mean+x+1,
                                y_0 = model_fit.y_mean+y+1,
                                theta = model_fit.theta,
                                a = 6*model_fit.x_stddev,
                                b = 6*model_fit.y_stddev
        )
        src_mask_i = src_mask_model(xx, yy)

        src_mask[src_mask_i == 1] = 1

        if 0:
            fig, axes = plt.subplots(1, 3)
            for ax in axes: ax.set_aspect('equal')
            axes[0].pcolor(wl_img)
            axes[0].contour(box_msk, levels=[0.5], colors=['w'])
            axes[0].contour(src_mask, levels=[0.5], colors=['w'])
            axes[1].pcolor(box)
            axes[2].pcolor(model_fit(box_xx, box_yy))
            fig.show()
            input("")
            plt.close()



        #Adjust center of PSF back to global coords
        model_fit.x_mean += x
        model_fit.y_mean += y

    if instrument=="KCWI":

        for wi in range(Nw):
            for xi in range(Nx):

                inputFits[0].data[wi, :, xi] -= np.median(inputFits[0].data[wi, :, xi])


    return inputFits

def main():

    #Handle input with argparse
    parser = argparse.ArgumentParser(description="""
    TBD
    """)
    parser.add_argument('-cube',
                        type=str,
                        help='Cube to be cropped (for working on a single cube).',
                        default=None
    )
    parser.add_argument('-params',
                        type=str,
                        help='CWITools parameter file (for working on a list of input cubes).',
                        default=None
    )
    parser.add_argument('-cubetype',
                        type=str,
                        help='The cube type to load (e.g. icubes.fits) if working with a parameter file.',
                        default=None

    )
    parser.add_argument('-ext',
                        type=str,
                        help='The filename extension to add to modified cubes. Default: .f.fits',
                        default=".sc.fits"
    )
    args = parser.parse_args()

    #Make list out of single cube if working in that mode
    if args.cube!=None and args.params==None and args.cubetype==None:

        if os.path.isfile(args.cube): fileList = [args.cube]
        else:
            raise FileNotFoundError("Input file not found. \nFile:%s"%args.cube)

    #Load list from parameter files if working in that mode
    elif args.cube==None and args.params!=None and args.cubetype!=None:

        # Check if any parameter values are missing (set to set-up mode if so)
        if os.path.isfile(args.params): parameters = params.loadparams(args.params)
        else:
            raise FileNotFoundError("Parameter file not found.\nFile:%s"%args.params)

        # Get filenames
        fileList = params.findfiles(parameters,args.cubetype)

    #Make sure usage is understood if some odd mix
    else:
        raise SyntaxError("""
        Usage should be one of the following modes:\
        \n\nUse -cube argument to specify one input cube to crop\
        \nOR\
        \nUse -params AND -cubetype flag together to load cubes from parameter file.
        """)

    # Open fits objects
    for fileName in fileList:

        fitsFile = fits.open(fileName)

        # Pass to trimming function
        fitsFile_corrected = slicecorr(fitsFile)

        outFileName = fileName.replace('.fits',args.ext)
        fitsFile_corrected.writeto(outFileName,overwrite=True)
        print("Saved %s"%outFileName)

if __name__=="__main__": main()
