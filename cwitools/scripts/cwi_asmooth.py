from astropy.io import fits
from astropy.convolution import Box1DKernel,Gaussian1DKernel,convolve_fft,Gaussian2DKernel
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d,uniform_filter1d
from scipy.signal import boxcar,gaussian,medfilt,convolve2d

import argparse
import numpy as np
import scipy
import sys
import time

from cwitools import libs


def asmooth3d(cube_path, var_path, snr_min = 5, snr_max = None,
        rmode = 'gaussian', wmode = 'gaussian',
        r_range = (2, 4), w_range = (2, 4),
        rstep_min = 0.5, wstep_min = 0.5,
        save_wker = False, save_rker = False, save_snr = False,
        ext = ".AKS.fits", logpath = "", verbose = True):

    #Timer start
    tStart = time.time()


    if logpath == "": logpath = cube_path.replace(".fits",".AKS.log")

    logfile = open(logpath,'w')

    #Output wrapper
    def output(s, log, verbose):
        if verbose: print(s,end='')
        log.write(s)

    #Exit with proper log file handling
    def error(errmsg, log):
        log.close()
        raise RuntimeError(errmsg)


    output("# Input intensity data: %s\n" % cube_path, logfile, verbose)
    output("# Input variance data: %s\n" % var_path, logfile, verbose)
    output("# XY Smoothing mode: %s\n" % rmode, logfile, verbose)
    output("# Wav Smoothing mode: %s\n" % wmode, logfile, verbose)


    #Open input intensity cube
    try: fI = fits.open(cube_path)
    except: error("# Error opening file %s."%cube_path, logfile, verbose)

    #Open input variance cube
    try: fV = fits.open(var_path)
    except: error("# Error opening file %s."%var_path, logfile, verbose)


    ## VARIABLES & DATA STRUCTURES

    #Load input data
    I = fI[0].data.copy()   #Original intensity cube
    V = fV[0].data.copy()   #Original variance cube

    #Convert from intensity to variance-weighted intensity (Credit:E.D.)
    V[V<=0] = np.inf
    I /= V
    V = 1/V

    #Create required cubes
    D = np.zeros_like(I)     #Detection cube
    DVar = np.zeros_like(I)     #Detection variance cube
    M = np.zeros_like(I)     #Detection mask cube
    S = np.zeros_like(I)     #SNR Cube
    Kr = np.zeros_like(I)    #Spatial kernel sizes
    Kw = np.zeros_like(I)    #Wavelength kernel sizes


    #Calculate signal-to-noise parameters
    snr_min = float(snr_min)
    snr_max = snr_min*1.1 if snr_max==None else snr_max

    #Make sure smoothing scale maximums aren't too large
    rScale0,rScale1 = r_range
    wScale0,wScale1 = w_range
    rScaleMax = np.min(D.shape[1:])/4.0
    wScaleMax = D.shape[0]/4.0
    if rScale1>rScaleMax: rScale1 = rScaleMax
    if wScale1>wScaleMax: wScale1 = wScaleMax

    ## PRE-PROCESSING FOR MAIN LOOP

    #Create mask of empty spaxels (i.e. non-observed regions)
    mask2D = (np.max(I,axis=0)==0)

    #Create 3D mask equivalent of mask2D by masking spectra in each masked spaxel
    M = M.T
    M[mask2D.T] = 1
    M = M.T

    #Get number of pixels already mapped before starting
    N0 = np.sum(M)

    #Initialize spatial kernel variables
    rScale = rScale0
    rStep = rstep_min

    #Initialize backup variables
    rScale_old = rScale
    rStep_old  = rStep


    ## MAIN LOOP
    output("# %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n" %\
    ('wScale','wStep','rScale','rStep','Npix','% Done','minSNR','medSNR','maxSNR','mid/med'), logfile, verbose)

    while rScale < rScale1: #Run through wavelength bins

        #Spatially smooth weighted intensity data and corresponding variance
        Ir  = libs.science.smooth3d(I,rScale,axes=[1,2],ktype=rmode,var=False)
        Vr  = libs.science.smooth3d(V,rScale,axes=[1,2],ktype=rmode,var=False)

        #Smooth variance with kernel squared for error propagation
        Vr2 = libs.science.smooth3d(V,rScale,axes=[1,2],ktype=rmode,var=True)

        #Initialize wavelelength kernel variables
        wScale = wScale0
        wStep  = wstep_min

        #Initialize backups
        wScale_old = wScale
        wStep_old  = wStep

        #Keep track of total number of detections at this rScale
        Nr_tot = 0

        while wScale < wScale1:

            #Output first half of diagnostic info
            output("%8.2f %8.3f %8.2f %8.3f" % (wScale,wStep,rScale,rStep), logfile, verbose)

            #Reset some values
            detFlag = False #Flag for detections
            breakFlag = False #Flag for breaking out of inner loop
            f = -1 #Ratio of median detected SNR to midSNR

            #Wavelength-smooth data, as above
            Irw  = libs.science.smooth3d(Ir,wScale,axes=[0],ktype=wmode,var=False)
            Vrw  = libs.science.smooth3d(Vr,wScale,axes=[0],ktype=wmode,var=False)

            #Smooth variance with kernel squared for error propagation
            Vrw2 = libs.science.smooth3d(Vr2,wScale,axes=[0],ktype=wmode,var=True)

            #Replace non-positive values
            libs.science.nonpos2inf(Vrw2)

            #Calculate SNR cube (Credit:E.D.)
            # Intensity values are weighted by w=1/V, so
            # Signal = sum(I*w*f)/sum(w*f)
            # Noise  = sqrt( sum(w*f^2)/sum(w*f) )

            ker_vol = np.sqrt(2*np.pi*np.power(rScale/2.355,2)*wScale)
            Vrw2 *= ker_vol**2

            SNR = (Irw/np.sqrt(Vrw2))

            #Get indices of detections
            detections = (SNR >= snr_min) & (M==0)

            #Get SNR values and total # of new detections
            SNRS = SNR[detections]
            Nvox = len(SNRS)

            #Condition 1: 5 or more detections, so median is well defined
            if Nvox>=5:

                #Calculate median
                medianSNR = np.median(SNRS)

                # Calculate ratio of mid-point to median
                # We use this value to determine how under/over-smoothed we are
                f = (snr_min+snr_max)/(2*medianSNR)

                #Condition 1.1: If we are oversmoothed (i.e. median detected SNR > midSNR)
                if f<1:

                    #Condition 1.1.1: Oversmoothed but wav kernel is larger than min
                    if wScale>wScale0:

                        #Do not update backups
                        #Do not raise detection flag

                        #Set step-size to half distance between current and previous scales
                        wStep = (wScale - wScale_old)/2.0

                        #Make sure step-size does not get smaller than minimum
                        if wStep<wstep_min: wStep=wstep_min

                        #Step backwards
                        wScale -= wStep

                        #Make sure w scale does not go below minimum
                        if wScale<wScale0: wScale=wScale0

                    #Condition 1.1.2: Oversmoothed, w kernel is minimum, r kernel is not
                    elif rScale>rScale0:

                        #Do not update w kernel
                        #Do not update r kernel backups
                        #Do not raise detection flag

                        #Set step-size to half distance between current and previous scales
                        rStep = (rScale - rScale_old)/2.0

                        #Make sure step-size does not get smaller than minimum
                        if rStep<rstep_min: rStep=rstep_min

                        #Step backwards
                        rScale -= rStep

                        #Make sure w scale does not go below minimum
                        if rScale<rScale0: rScale=rScale0

                        #Set flag to break out of inner loop after detections phase
                        breakFlag = True

                    #Condition 1.1.3: Oversmoothed but already at smallest kernel sizes for both kernels
                    else:

                        #Backup w kernel params
                        old_wScale = wScale
                        old_wStep  = wStep

                        #Raise detection flag
                        detFlag = True

                        #Decrease step-size by 50%
                        wStep *= 0.5

                        #Increase wScale
                        wScale += wStep

                #Condition 1.2: Undersmoothed (medianSNR < midSNR)
                if f>1:

                    #If this was the first step after spatial smoothing, update spatial step size
                    if wScale==wScale0:

                        #Backup old values
                        rScale_old = rScale
                        rStep_old  = rStep

                        #Update step size using f
                        rStep  = (f-1)*rScale_old

                        #Make sure step size is at least the minimum value
                        if rStep<rstep_min: rStep = rstep_min

                    #Backup old w kernel values
                    wScale_old = wScale
                    wStep_old  = wStep

                    #Calculate corresponding w step size
                    wStep  = (f-1)*wScale_old

                    #Make sure step size is at least the minimum value
                    if wStep<wstep_min: wStep = wstep_min

                    #Update wScale
                    wScale += wStep

                    #Raise detection flag
                    detFlag = True

                #Condition 1.3: medianSNR == midSNR, so can't update using f
                else:

                    #Backup old values
                    wScale_old = wScale
                    wStep_old = wStep

                    #Update using current wStep
                    wScale += wStep

                    #Raise detection flag
                    detFlag = True

            #Condition 2: Fewer than 5 (not non-zero) detections found
            elif Nvox>0:

                #Backup old values
                wScale_old = wScale
                wStep_old  = wStep

                #Increase step size by 25%
                wStep *= 1.25

                #Increase kernel size
                wScale += wStep

                #Raise detections flag
                detFlag = True

            #Condition 3: No detections
            else:

                #Backup old values
                wScale_old = wScale
                wStep_old = wStep

                #Increase step size by 50%
                wStep *= 1.5

                #Increase kernel size
                wScale += wStep


            #Detection Phase
            if detFlag:

                #Divide out the variance component to recover intensity
                libs.science.nonpos2inf(Vrw)
                uIrw = Irw/Vrw #Divide by inverted var (i.e. multiply by original var)

                SNR_Test1 = Irw/np.sqrt(Vrw)
                SNR_Test2 = uIrw/np.sqrt(1/Vrw)


                #Update relevant cubes
                D[detections] = uIrw[detections]
                M[detections] = 1
                S[detections] = SNR[detections]
                DVar[detections] = 1/Vrw[detections]#np.power(uIrw[detections]/SNR[detections], 2)
                Kr[detections] = rScale
                Kw[detections] = rScale

                #Null the detected voxels to prevent further contributions
                I[detections] = 0
                V[detections] = 0

                #Update outer-loop smoothing at current scale after subtraction
                Ir  = libs.science.smooth3d(I,rScale_old,axes=(1,2),ktype=rmode,var=False)
                Vr  = libs.science.smooth3d(V,rScale_old,axes=(1,2),ktype=rmode,var=False)
                Vr2 = libs.science.smooth3d(V,rScale_old,axes=(1,2),ktype=rmode,var=True)

            ## Output some diagnostics
            perc = 100*(np.sum(M)-N0)/M.size
            if Nvox>0:
                maxS,minS = np.max(SNRS),np.min(SNRS)
                if Nvox>5: medS = np.median(SNRS)
                else: medS = np.mean(SNRS)
            else: maxS,minS,medS = 0,0,0

            Nr_tot += Nvox
            output("%8i %8.3f %8.4f %8.4f %8.4f %8s\n" %\
            (Nvox,perc,minS,medS,maxS,str(round(f,5))), logfile, verbose)

            sys.stdout.flush()

            if breakFlag: break

        if Nr_tot<5: rStep*=2
        rScale += rStep

    outFileName = cube_path.replace('.fits',ext)
    hdu = fits.PrimaryHDU(D)
    hdulist = fits.HDUList([hdu])
    hdulist[0].header = fI[0].header
    hdulist.writeto(outFileName,overwrite=True)
    output("# Wrote %s\n" % outFileName, logfile, verbose)

    DVar[np.isnan(DVar)] = np.inf
    outFileNameVar = outFileName.replace('.fits','.var.fits')
    hdu = fits.PrimaryHDU(DVar)
    hdulist = fits.HDUList([hdu])
    hdulist[0].header = fI[0].header
    hdulist.writeto( outFileNameVar, overwrite=True)
    output("# Wrote %s\n" % outFileNameVar, logfile, verbose)

    if save_rker:
        outFileNameRkernel = outFileName.replace('.fits','.rKer.fits')
        hdu = fits.PrimaryHDU(Kr)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = fI[0].header
        hdulist.writeto(outFileNameRkernel,overwrite=True)
        output("# Wrote %s\n" % outFileNameRkernel, logfile, verbose)

    if save_wker:
        outFileNameWkernel = outFileName.replace('.fits','.wKer.fits')
        hdu = fits.PrimaryHDU(Kw)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = fI[0].header
        hdulist.writeto(outFileNameWkernel,overwrite=True)
        output("# Wrote %s\n" % outFileNameWkernel, logfile, verbose)

    if save_snr:
        outFileNameSNR = outFileName.replace('.fits','.SNR.fits')
        hdu = fits.PrimaryHDU(S)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = fI[0].header
        hdulist.writeto(outFileNameSNR,overwrite=True)
        output("# Wrote %s\n" % outFileNameSNR, logfile, verbose)

    #Timer end
    tFinish = time.time()
    output("Elapsed time: %.2f seconds" % (tFinish-tStart), logfile, verbose)

def main():

    # Use python's argparse to handle command-line input
    parser = argparse.ArgumentParser(description='Perform Adaptive-Kernel-Smoothing on a data cube (requires variance cube).')
    mainGroup = parser.add_argument_group(title="Main",description="Basic input")
    mainGroup.add_argument('cube',
                        type=str,
                        metavar='input cube',
                        help='The cube to be smoothed.'
    )
    mainGroup.add_argument('var',
                        type=str,
                        metavar='variance',
                        help='The associated variance cube.'
    )
    methodGroup = parser.add_argument_group(title="Method",description="Smoothing parameters.")
    methodGroup.add_argument('-snr_min',
                        type=float,
                        metavar='float',
                        help='The objective minimum signal-to-noise level (Default:3)',
                        default=3
    )
    methodGroup.add_argument('-snr_max',
                        type=float,
                        metavar='float',
                        help='(Soft) maximum SNR, used to determine when oversmoothing occurs. Default: 1.1*snr_min',
                        default=None
    )
    methodGroup.add_argument('-rmode',
                        type=str,
                        metavar='str',
                        help='Spatial moothing mode (box/gaussian) - Default: gaussian',
                        default='gaussian',
                        choices=['box','gaussian']
    )
    methodGroup.add_argument('-wmode',
                        type=str,
                        metavar='str',
                        help='Wavelength moothing mode (box/gaussian) - Default: gaussian',
                        default='gaussian',
                        choices=['box','gaussian']
    )
    methodGroup.add_argument('-rScale0',
                        type=float,
                        metavar='float (px)',
                        help='Minimum spatial smoothing scale (Default:3)',
                        default=2
    )
    methodGroup.add_argument('-wScale0',
                        type=float,
                        metavar='float (px)',
                        help='Minimum wavelength smoothing scale (Default:2)',
                        default=2
    )
    methodGroup.add_argument('-rScale1',
                        type=float,
                        metavar='float (px)',
                        help='Maximum spatial smoothing scale (Default:10)',
                        default=4
    )
    methodGroup.add_argument('-wScale1',
                        type=float,
                        metavar='float (px)',
                        help='Maximum wavelength smoothing scale (Default:5)',
                        default=4
    )
    methodGroup.add_argument('-r_stepmin',
                        type=float,
                        metavar='float (px)',
                        help='Minimum spatial scale step-size (Default:0.1px)',
                        default=0.5
    )
    methodGroup.add_argument('-wstep_min',
                        type=float,
                        metavar='float (px)',
                        help='Minimum wavelength scale step-size (Default:0.5px)',
                        default=0.5
    )

    fileIOGroup = parser.add_argument_group(title="File I/O",description="File input/output options.")

    fileIOGroup.add_argument('-ext',
                        type=str,
                        metavar='str',
                        help='Extension to append to subtracted cube (.AKS.fits)',
                        default='.AKS.fits'
    )
    fileIOGroup.add_argument('-save_rker',
                        type=str,
                        metavar='bool',
                        help='Save spatial smoothing kernel sizes as cube (True/False)',
                        choices=["True","False"],
                        default="False"
    )
    fileIOGroup.add_argument('-save_wker',
                        type=str,
                        metavar='bool',
                        help='Save wavelength smoothing kernel sizes as cube (True/False)',
                        choices=["True","False"],
                        default="False"
    )
    fileIOGroup.add_argument('-save_snr',
                        type=str,
                        metavar='bool',
                        help='Save SNR output cubes (True/False)',
                        choices=["True","False"],
                        default="False"
    )
    args = parser.parse_args()

    #Convert str args to bools
    args.save_rker=(args.save_rker=="True")
    args.save_wker=(args.save_wker=="True")
    args.save_snr=(args.save_snr=="True")

    asmooth3d(args.cube,args.var,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        rmode=args.rmode,
        wmode=args.wmode,
        r_range=(args.rScale0,args.rScale1),
        w_range=(args.wScale0,args.wScale1),
        rstep_min=args.r_stepmin,
        wstep_min=args.wstep_min,
        ext=args.ext,
        save_wker=args.save_wker,
        save_rker=args.save_rker,
        save_snr=args.save_snr
    )

if __name__=="__main__": main()
