from astropy.io import fits as fitsIO
import numpy as np
import sys
import libs

import matplotlib.pyplot as plt

# Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Get cubetype without the file extensions ('icube', 'scube', 'ocube')
cubetypeShort = cubetype.split(".")[0][:-1]

# Load params from paramfile
params = libs.params.loadparams(parampath)

# Get regular and sky filenames   
files = libs.io.findfiles(params,cubetype)

# Get regular and sky filenames   
fits = [ fitsIO.open(f) for f in files ]

# Get Nod-and-Shuffle status of each fits
nas = [ f[0].header["NASMASK"]==True for f in fits ]

#Get length before any sky files are added
N = len(fits)

inst = [ x for x in params["INST"] ]

# Get any sky images that are needed
if not np.array(nas).all():

    #Add image numbers and instrument names to lists
    snums,sinst = [],[]
    for i,s in enumerate(params["SKY_ID"]):   
        if s not in params["IMG_ID"] and s not in snums:          
            snums.append(s)
            sinst.append(params["INST"][i])
            
    #Create file paths
    sfiles = [ files[0].replace(params["IMG_ID"][0],s) for s in snums ]

    #Load fits files
    sfits  = [ fitsIO.open(s) for s in sfiles ]
    
    #Update relevant lists
    for i in range(len(sfiles)):
    
        files.append(sfiles[i])
        fits.append(sfits[i])
        inst.append(sinst[i])
        nas.append(False)
        
        
#Run through all images now and perform corrections
for i,fileName in enumerate(files):

    print fileName
    
    #If this is a nod-and-shuffle exposure, use the object cube
    if nas[i]:
    
        #Open icube for locating source and scube for sky-line fitting
        radecFile = fileName.replace(cubetypeShort,'icube')
        skyFile   = fileName.replace(cubetypeShort,'scube')

        #Open FITS files
        radecFITS = fitsIO.open(radecFile)
        skyFITS   = fitsIO.open(skyFile)      
        
    # Otherwise use the file itself (this assumes user is not calling fixWCS on sky-subtracted data)
    else:
    
        #Open icube for both source location and sky fitting
        radecFITS = fits[i]
        skyFITS   = fits[i]
        
    #Measure RA/DEC center values for this exposure
    crval1,crval2,crpix1,crpix2 = libs.cubes.fixRADEC(radecFITS,params["RA"],params["DEC"])

    #Save 0-indexed value to parameter file
    params["SRC_X"][i] = crpix1
    params["SRC_Y"][i] = crpix2
    
    #Measure wavelength center this exposure
    crval3,crpix3 = libs.cubes.fixWav(skyFITS,params["INST"][i])
       
    #Close current FITS without saving any changes
    radecFITS.close()
    skyFITS.close()
    
    #Create lists of updated crval/crpix values
    crvals = [ crval1, crval2, crval3 ]
    crpixs = [ crpix1+1, crpix2+1, crpix3 ]
    
    #Make list of relevant cubes to be corrected
    cubes = ['icube','vcube']
    if nas[i]:
        cubes.append('scube')
        cubes.append('ocube')
    

    fits = fitsIO.open(fileName)
    im = np.sum(fits[0].data,axis=0)
    
    #Load fits, modify header and save for each cube type
    for c in cubes:
        
        filePath = fileName.replace(cubetypeShort,c)
        f = fitsIO.open(filePath)
        
        for j in range(3):
        
            f[0].header["CRVAL%i"%(j+1)] = crvals[j]
            f[0].header["CRPIX%i"%(j+1)] = crpixs[j]
        
        print f[0].header["CRPIX1"]
        print f[0].header["CRPIX2"]
        
        wcPath = filePath.replace('.fits','.wc.fits')
        f[0].writeto(wcPath,overwrite=True)
        print("Saved %s"%wcPath)

libs.params.writeparams(params,parampath)      


    
