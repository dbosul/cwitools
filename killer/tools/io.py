import astropy.io.fits as fitsIO
import numpy as np
import os


##################################################################################################
# Find FITS files using params and given cube type
def findfiles(params,cubetype):
    print("Locating and loading FITS files:")  

    if not os.path.isdir(params["DATA_DIR"]):
        print("Data directory (%s) does not exist. Please correct and try again." % params["DATA_DIR"])
        sys.exit()
    target_files = ["" for i in range(len(params["IMG_ID"]))]

    for root, dirs, files in os.walk(params["DATA_DIR"]):

        rec = root.replace(params["DATA_DIR"],'').count("/")

        if rec > params["DATA_DEPTH"]: continue
        else:
            
            for f in files:

                if cubetype in f:
                    
                    for i,ID in enumerate(params["IMG_ID"]):
                    
                        if ID in f:
                        
                            target_files[i] = os.path.join(root,f)

    #Print file paths or file not found errors
    for i,f in enumerate(target_files):
        if f!="": print f
        else: print("File not found: ID:%s Type:%s" % (params["IMG_ID"][i],cubetype))

    
    return target_files
    
    

#######################################################################
#Output image as fits
def saveFits(data,path,header):
        print("""Saving %s""" % path)
        hdu = fitsIO.PrimaryHDU(data)
        hdulist = fitsIO.HDUList([hdu])
        hdulist[0].header = header
        hdulist.writeto(path,clobber=True)    
