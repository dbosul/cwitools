from astropy.io import fits as fitsIO
import numpy as np
import sys
import libs

# Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Get cubetype without the file extensions ('icube', 'scube', 'ocube')
cubetypeShort = cubetype.split(".")[0]


# Load params from paramfile
params = libs.params.loadparams(parampath)

# Get regular and sky filenames   
files = libs.io.findfiles(params,cubetype)

# Get regular and sky filenames   
fits = [ fitsIO.open(f) for f in files ]

# Get Nod-and-Shuffle status of each fits
nas = np.array([ f[0].header["NASMASK"]==True for f in fits ])

#Get length before any sky files are added
N = len(fits)

# Get any sky images that are needed
if not nas.all():

    snums  = [ s for s in set(params["SKY_ID"]) if s not in params["IMG_ID"] ]
    
    sfiles = [ files[0].replace(params["IMG_ID"][0],s) for s in snums ]

    sfits  = [ fitsIO.open(s) for s in sfiles ]
    
    for i in range(len(sfiles)):
    
        files.append(sfiles[i])
        fits.append(sfits[i])
        

#If there are any non Nod-and-Shuffle exposures
for i,fileName in enumerate(files):

    print fileName
    
    #If this is a nod-and-shuffle exposure, use the object cube
    if nas[i]:
        radecFile = fileName.replace(cubetypeShort,'icube')
        skyFile   = fileName.replace(cubetypeShort,'scube')

        #Open FITS file
        radecFITS = fitsIO.open(radecFile)
        skyFITS   = fitsIO.open(skyFile)        
    # Otherwise use the file itself (this assumes user is not calling fixWCS on sky-subtracted data)
    else:
    
        #Open FITS file
        radecFITS = fits[i]
        skyFITS   = fits[i]

    

    #Measure RA/DEC center values for this exposure
    crval12,crpix12 = libs.cubes.fitRADEC(radecFITS,params["RA"],params["DEC"])
    
    #Measure wavelength center this exposure
    crval3,crpix3 = libs.cubes.fitWav(skyFITS,4358.33)
    
    print crval12,crval3,crpix12,crpix3
    
