fileExtfrom astropy.io import fits
from astropy.convolution import Box1DKernel,Gaussian1DKernel,convolve_fft,Gaussian2DKernel
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d,uniform_filter1d
from scipy.signal import boxcar,gaussian,medfilt,convolve2d

import argparse
import numpy as np
import scipy
import sys
import time

import libs

#Output wrapper
logFile=None
def output(s):
    global logFile
    print(s,end='')
    logFile.write(s)

#Exit with proper log file handling
def exit():
    global logFile
    logFile.close()
    sys.exit()

def run(cubePath,varPath,snrMin=5,snrMax=None,rMode='gaussian',wMode='gaussian',
        rSmoothRange=(3,20),wSmoothRange=(2.5,12),rStepMin=0.5,wStepMin=0.5,
        ext=".AKS.fits",saveWKernel=False,saveRKernel=False,saveSNR=False ):

    #Timer start
    tStart = time.time()

    #Load parameters
    Ifile = cubePath
    Vfile = varPath
    logFile = open(Ifile.replace('.fits','.AKS.log'),'w')

    output("# Input intensity data: %s\n" % Ifile)
    output("# Input variance data: %s\n" % Vfile)
    output("# XY Smoothing mode: %s\n" % rMode)
    output("# Wav Smoothing mode: %s\n" % wMode)


    #Open input intensity cube
    try: fI = fits.open(Ifile)
    except: output("# Error opening file %s. Please check and try again.\n"%Ifile);exit()

    #Open input variance cube
    try: fV = fits.open(Vfile)
    except: output("# Error opening file %s. Please check it exists and try again, or set variance input with 'varcube='.\n"%Vfile);exit()


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
    snrMin = float(snrMin)
    snrMax = snrMin*1.1 if snrMax==None else snrMax

    #Make sure smoothing scale maximums aren't too large
    rScale0,rScale1 = rScaleRange
    wScale0,wScale1 = wScaleRange
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
    rStep = rStepMin

    #Initialize backup variables
    rScale_old = rScale
    rStep_old  = rStep


    ## MAIN LOOP
    output("# %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n" % ('wScale','wStep','rScale','rStep','Npix','% Done','minSNR','medSNR','maxSNR','mid/med')  )
    while rScale < rScale1: #Run through wavelength bins

        #Spatially smooth weighted intensity data and corresponding variance
        Ir  = libs.science.smooth3D(I,rScale,axes=[1,2],ktype=rMode,var=False)
        Vr  = libs.science.smooth3D(V,rScale,axes=[1,2],ktype=rMode,var=False)

        #Smooth variance with kernel squared for error propagation
        Vr2 = libs.science.smooth3D(V,rScale,axes=[1,2],ktype=rMode,var=True)

        #Initialize wavelelength kernel variables
        wScale = wScale0
        wStep  = wStepMin

        #Initialize backups
        wScale_old = wScale
        wStep_old  = wStep

        #Keep track of total number of detections at this rScale
        Nr_tot = 0

        while wScale < wScale1:

            #Output first half of diagnostic info
            output("%8.2f %8.3f %8.2f %8.3f" % (wScale,wStep,rScale,rStep))

            #Reset some values
            detFlag = False #Flag for detections
            breakFlag = False #Flag for breaking out of inner loop
            f = -1 #Ratio of median detected SNR to midSNR

            #Wavelength-smooth data, as above
            Irw  = libs.science.smooth3D(Ir,wScale,axes=[0],ktype=wMode,var=False)
            Vrw  = libs.science.smooth3D(Vr,wScale,axes=[0],ktype=wMode,var=False)

            #Smooth variance with kernel squared for error propagation
            Vrw2 = libs.science.smooth3D(Vr2,wScale,axes=[0],ktype=wMode,var=True)

            #Replace non-positive values
            libs.science.nonpos2inf(Vrw2)

            #Calculate SNR cube (Credit:E.D.)
            # Intensity values are weighted by w=1/V, so
            # Signal = sum(I*w*f)/sum(w*f)
            # Noise  = sqrt( sum(w*f^2)/sum(w*f) )
            SNR = (Irw/np.sqrt(Vrw2))

            #Get indices of detections
            detections = (SNR >= snrMin) & (M==0)

            #Get SNR values and total # of new detections
            SNRS = SNR[detections]
            Nvox = len(SNRS)

            #Condition 1: 5 or more detections, so median is well defined
            if Nvox>=5:

                #Calculate median
                medianSNR = np.median(SNRS)

                # Calculate ratio of mid-point to median
                # We use this value to determine how under/over-smoothed we are
                f = (snrMin+snrMax)/(2*medianSNR)

                #Condition 1.1: If we are oversmoothed (i.e. median detected SNR > midSNR)
                if f<1:

                    #Condition 1.1.1: Oversmoothed but wav kernel is larger than min
                    if wScale>wScale0:

                        #Do not update backups
                        #Do not raise detection flag

                        #Set step-size to half distance between current and previous scales
                        wStep = (wScale - wScale_old)/2.0

                        #Make sure step-size does not get smaller than minimum
                        if wStep<wStepMin: wStep=wStepMin

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
                        if rStep<rStepMin: rStep=rStepMin

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
                        if rStep<rStepMin: rStep = rStepMin

                    #Backup old w kernel values
                    wScale_old = wScale
                    wStep_old  = wStep

                    #Calculate corresponding w step size
                    wStep  = (f-1)*wScale_old

                    #Make sure step size is at least the minimum value
                    if wStep<wStepMin: wStep = wStepMin

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
                uIrw = Irw/Vrw

                #Update relevant cubes
                D[detections] = uIrw[detections]
                M[detections] = 1
                S[detections] = SNR[detections]
                DVar[detections] = 1/Vrw[detections]
                Kr[detections] = rScale
                Kw[detections] = rScale

                #Null the detected voxels to prevent further contributions
                I[detections] = 0
                V[detections] = 0

                #Update outer-loop smoothing at current scale after subtraction
                Ir  = libs.science.smooth3D(I,rScale_old,axes=(1,2),ktype=rMode,var=False)
                Vr  = libs.science.smooth3D(V,rScale_old,axes=(1,2),ktype=rMode,var=False)
                Vr2 = libs.science.smooth3D(V,rScale_old,axes=(1,2),ktype=rMode,var=True)

            ## Output some diagnostics
            perc = 100*(np.sum(M)-N0)/M.size
            if Nvox>0:
                maxS,minS = np.max(SNRS),np.min(SNRS)
                if Nvox>5: medS = np.median(SNRS)
                else: medS = np.mean(SNRS)
            else: maxS,minS,medS = 0,0,0

            Nr_tot += Nvox
            output("%8i %8.3f %8.4f %8.4f %8.4f %8s\n" % (Nvox,perc,minS,medS,maxS,str(round(f,5))))

            sys.stdout.flush()

            if breakFlag: break

            if time.time()-tStart>90000:
                print("aSmooth taking longer than 20minutes. Exiting.")
                sys.exit()

        if Nr_tot<5: rStep*=2
        rScale += rStep

    outFileName = Ifile.replace('.fits',fileExt)
    hdu = fits.PrimaryHDU(D)
    hdulist = fits.HDUList([hdu])
    hdulist[0].header = fI[0].header
    hdulist.writeto(outFileName,overwrite=True)
    output("# Wrote %s\n" % outFileName)

    outFileNameVar = outFileName.replace('.fits','.var.fits')
    hdu = fits.PrimaryHDU(DVar)
    hdulist = fits.HDUList([hdu])
    hdulist[0].header = fI[0].header
    hdulist.writeto( outFileNameVar, overwrite=True)
    output("# Wrote %s\n" % outFileNameVar)

    if saveRKernel:
        outFileNameRkernel = outFileName.replace('.fits','.rKer.fits')
        hdu = fits.PrimaryHDU(Kr)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = fI[0].header
        hdulist.writeto(outFileNameRkernel,overwrite=True)
        output("# Wrote %s\n" % outFileNameRkernel)

    if saveWKernel:
        outFileNameWkernel = outFileName.replace('.fits','.wKer.fits')
        hdu = fits.PrimaryHDU(Kw)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = fI[0].header
        hdulist.writeto(outFileNameWkernel,overwrite=True)
        output("# Wrote %s\n" % outFileNameWkernel)

    if saveSNR:
        outFileNameSNR = outFileName.replace('.fits','.SNR.fits')
        hdu = fits.PrimaryHDU(S)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = fI[0].header
        hdulist.writeto(outFileNameSNR,overwrite=True)
        output("# Wrote %s\n" % outFileNameSNR)

    #Timer end
    tFinish = time.time()
    print("Elapsed time: %.2f seconds" % (tFinish-tStart))

if __name__=="__main__":

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
    methodGroup.add_argument('-snrMin',
                        type=float,
                        metavar='float',
                        help='The objective minimum signal-to-noise level (Default:3)',
                        default=3
    )
    methodGroup.add_argument('-snrMax',
                        type=float,
                        metavar='float',
                        help='(Soft) maximum SNR, used to determine when oversmoothing occurs. Default: 1.1*snrMin',
                        default=None
    )
    methodGroup.add_argument('-rMode',
                        type=str,
                        metavar='str',
                        help='Spatial moothing mode (box/gaussian) - Default: gaussian',
                        default='gaussian',
                        choices=['box','gaussian']
    )
    methodGroup.add_argument('-wMode',
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
                        default=3
    )
    methodGroup.add_argument('-wScale0',
                        type=float,
                        metavar='float (px)',
                        help='Minimum wavelength smoothing scale (Default:2)',
                        default=2.5
    )
    methodGroup.add_argument('-rScale1',
                        type=float,
                        metavar='float (px)',
                        help='Maximum spatial smoothing scale (Default:10)',
                        default=20
    )
    methodGroup.add_argument('-wScale1',
                        type=float,
                        metavar='float (px)',
                        help='Maximum wavelength smoothing scale (Default:5)',
                        default=12
    )
    methodGroup.add_argument('-rStepMin',
                        type=float,
                        metavar='float (px)',
                        help='Minimum spatial scale step-size (Default:0.1px)',
                        default=0.5
    )
    methodGroup.add_argument('-wStepMin',
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
    fileIOGroup.add_argument('-saveRKernel',
                        type=str,
                        metavar='bool',
                        help='Save spatial smoothing kernel sizes as cube (True/False)',
                        choices=["True","False"],
                        default="False"
    )
    fileIOGroup.add_argument('-saveWKernel',
                        type=str,
                        metavar='bool',
                        help='Save wavelength smoothing kernel sizes as cube (True/False)',
                        choices=["True","False"],
                        default="False"
    )
    fileIOGroup.add_argument('-saveSNR',
                        type=str,
                        metavar='bool',
                        help='Save SNR output cubes (True/False)',
                        choices=["True","False"],
                        default="False"
    )
    args = parser.parse_args()

    #Convert str args to bools
    args.saveRKernel=(args.saveRKernel=="True")
    args.saveWKernel=(args.saveWKernel=="True")
    args.saveSNR=(args.saveSNR=="True")

    run(args.cube,args.var,
        snrMin=args.snrMin,
        snrMax=args.snrMax,
        rMode=args.rMode,
        wMode=args.wMode,
        rSmoothRange=(args.rScale0,args.rScale1),
        wSmoothRange=(args.wScale0,args.wScale1),
        rStepMin=args.rStepMin,
        wStepMin=args.wStepMin,
        fileExt=args.ext,
        saveWKernel=args.saveWKernel,
        saveRKernel=args.saveRKernel,
        saveSNR=args.saveSNR
    )
