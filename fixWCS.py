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

# Get Nod-and-Shuffle status of each fits (based on paramfile)
nas = []
for i,f in enumerate(fits):   
    if params["IMG_ID"][i]==params["SKY_ID"][i]: nas.append(True)
    else: nas.append(False)


#Get length before any sky files are added
N = len(fits)
inst = [ x for x in params["INST"] ]

# Get any sky images that are needed
if not np.array(nas).all() and not (np.array(params["INST"])=="KCWI").all():

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

    #If this is a nod-and-shuffle exposure, use the object cube
    if nas[i] or inst[i]=="KCWI":
    
        #Open icube for locating source and scube for sky-line fitting
        radecFile = fileName.replace(cubetypeShort,'icube')
        skyFile   = fileName.replace(cubetype,'scube.fits')

        #Open FITS files
        radecFITS = fitsIO.open(radecFile)
        skyFITS   = fitsIO.open(skyFile)      
        
    # Otherwise use the file itself (this assumes user is not calling fixWCS on sky-subtracted data)
    else:
    
        #Open icube for both source location and sky fitting
        radecFITS = fits[i]
        skyFITS   = fits[i]
    
    #If this is a target image, we can use target to correct RA/DEC
    if i<len(params["IMG_ID"]):
        
        #Measure RA/DEC center values for this exposure
        crval1,crval2,crpix1,crpix2 = libs.cubes.fixRADEC(radecFITS,params["RA"],params["DEC"])
       
    else: crval1,crval2,crpix1,crpix2 = ( radecFITS[0].header[k] for k in ["CRVAL1","CRVAL2","CRPIX1","CRPIX2"] )
        
    #Measure wavelength center this exposure
    crval3,crpix3 = libs.cubes.fixWav(skyFITS,inst[i])
       
    #Close current FITS without saving any changes
    radecFITS.close()
    skyFITS.close()
    
    #Create lists of updated crval/crpix values
    crvals = [ crval1, crval2, crval3 ]
    crpixs = [ crpix1+1, crpix2+1, crpix3 ]
    
    #Make list of relevant cubes to be corrected
    cubes = ['icube','vcube']
    if nas[i] or inst[i]=="KCWI":
        cubes.append('scube')
        cubes.append('ocube')
    

    #Load fits, modify header and save for each cube type
    for c in cubes:
        
        filePath = fileName.replace(cubetypeShort,c)
        
        try: f = fitsIO.open(filePath)
        except:
            print("Could not open %s. Cube will not be corrected." % filePath)
            continue
            
        for j in range(2):
            
            f[0].header["CRVAL%i"%(j+1)] = crvals[j]
            f[0].header["CRPIX%i"%(j+1)] = crpixs[j]

        wcPath = filePath.replace('.fits','.wc.fits')
        f[0].writeto(wcPath,overwrite=True)
        print("Saved %s"%wcPath)

libs.params.writeparams(params,parampath)      


    
