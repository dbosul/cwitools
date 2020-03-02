import argparse
import numpy as np
from cwitools import libs


from astropy.convolution import Box1DKernel,Gaussian1DKernel
from astropy.io import fits
from astropy.modeling import models,fitting
from astropy.stats import sigma_clipped_stats

from scipy.stats import norm
from skimage import measure

# Use python's argparse to handle command-line input
parser = argparse.ArgumentParser(description='Threshold a data cube and label the resulting regions.')
mainGroup = parser.add_argument_group(title="Main",description="Basic input")
mainGroup.add_argument('cube',
                    type=str,
                    metavar='cube',
                    help='The input data cube.'
)
sgGroup = parser.add_argument_group(title="Segmentation Parameters")
sgGroup.add_argument('-var',
                    type=str,
                    metavar='path',
                    help='Variance cube, for calculating signal-to-noise.',
                    default=None

)
sgGroup.add_argument('-t',
                    type=float,
                    metavar='float',
                    help='The threshold to use, in units determined by the \'-u\' flag. Default: 0.',
                    default=0.0

)
sgGroup.add_argument('-u',
                    type=str,
                    metavar='string',
                    help='Units for the given threshold. Can be in standard deviations (std) or whatever flux units the cube already has (flx). Default: cbe.',
                    default='cbe',
                    choices=['std','flx']
)
sgGroup.add_argument('-c',
                    type=float,
                    metavar='float',
                    help='Connectivity. The maximum number of orthogonal hops from one voxel to another to be considered the same object.',
                    default=2
)
sgGroup.add_argument('-n',
                    type=int,
                    metavar='int',
                    help='Minimum number of voxels in an object to be counted.',
                    default=1
)
sgGroup.add_argument('-sW',
                    type=float,
                    metavar='float',
                    help='FWHM of wavelength smoothing kernel (gaussian).',
                    default=None
)
sgGroup.add_argument('-sR',
                    type=float,
                    metavar='float',
                    help='FWHM of spatial smoothing kernel (gaussian).',
                    default=None
)
sgGroup.add_argument('-wRange',
                    type=str,
                    metavar='tuple',
                    help='Wavelength range, in Angstrom, to consider for object detection e.g. 4100,4200',
                    default=None
)
args = parser.parse_args()


print("\nSegmenting %s"%args.cube)

cube,header = fits.getdata(args.cube,header=True)

if args.var!=None:
    usevar=True
    varcube = fits.getdata(args.var)
    libs.science.nonpos2inf(varcube)

else: usevar=False

#Convert to variance weighted flux for SNR calculation
if args.u=='std' and usevar:
    cube /= varcube
    varcube = 1/varcube


#Smooth in wavelength if requested
if args.sW!=None:
    cube = libs.science.smooth3D(cube,args.sW,axes=[0],ktype='box',var=False)
    if usevar: varcube = libs.science.smooth3D(varcube,args.sW,axes=[0],ktype='gaussian',var=True)

#Smooth spatially if requested
if args.sR!=None:
    cube = libs.science.smooth3D(cube,args.sR,axes=[1,2],ktype='box',var=False)
    if usevar: varcube = libs.science.smooth3D(varcube,args.sR,axes=[1,2],ktype='gaussian',var=True)

#Set wavelength range to be used
if args.wRange==None: w0,w1 = 0,np.inf
else:
    try: w0,w1 = ( int(x) for x in args.wRange.split(',') )
    except: print("Error parsing wRange argument. Should be tuple of ints or floats (e.g. 4100,4200)")

#Set threshold
threshold = args.t

#Convert cube to SNR units if requested
if args.u=='std':
    if usevar:
        varcube[varcube<=0] = np.inf
        cube = cube/np.sqrt(varcube)

        cube[np.abs(cube)>1000] = 0

        cubeClipped = cube[np.abs(cube)<10]
        cubeClippedStd = np.std(cubeClipped)

        cube /= cubeClippedStd

    else: cube/=np.std(cube)

#Cancel out wavelength ranges not requested by user
W = libs.cubes.get_wav_axis(header)
useW = (W>=w0) & (W<=w1)
cube[~useW] = 0

#Convert cube to binary mask by applying threshold
cube[cube<=args.t]=0
cube[cube>args.t]=1

#Label objects in cube
labelled = measure.label(cube,connectivity=args.c)

print("%i initial objects detected." % np.max(labelled))
labelNew = 1
rejects = 0
if args.n>1:
    for i in range(1,np.max(labelled)+1):

        mask_i = labelled==i
        nVox_i = np.count_nonzero(mask_i)
        if nVox_i<args.n:
            rejects+=1
            labelled[mask_i] = 0
        else:
            labelled[mask_i] = labelNew
            labelNew += 1


    print("%i objects rejected by n=%i voxel count limit."%(rejects,args.n))

outFits = fits.HDUList([fits.PrimaryHDU(labelled)])
outFits[0].header = header
outPath = args.cube.replace('.fits','.OBJ.fits')
outFits.writeto(outPath,overwrite=True)

Nexp=int(round((0.0233/100)*labelled.size))
print("%i objects remaining." % np.max(labelled))
print("Wrote %s"%outPath)
