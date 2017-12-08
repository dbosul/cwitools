#from astroquery.sdss import SDSS
from scipy.ndimage.interpolation import shift

import numpy as np

#Local imports
import qso

  
def fixWCS(fits_list,params):
    
    #Run through each fits image
    for i,fits in enumerate(fits_list):
    
        #First, get accurate in-cube X,Y location of QSO
        plot_title = "Select the object at RA:%.4f DEC:%.4f" % (params["RA"],params["DEC"])
        qfinder = qso.qsoFinder(fits,params["Z"],title=plot_title)
        x,y = qfinder.run()
        
        #Update parameters with new X,Y location
        params["SRC_X"][i] = x
        params["SRC_Y"][i] = y

        #Insert param-based RA/DEC into header
        h = fits[0].header
        if "RA" in h["CTYPE1"] and "DEC" in h["CTYPE2"]:
                 
            fits[0].header["CRVAL1"] = params["RA"]
            fits[0].header["CRVAL2"] = params["DEC"]
            
            fits[0].header["CRPIX1"] = x
            fits[0].header["CRPIX2"] = y
            
        elif "DEC" in h["CTYPE1"] and "RA" in h["CTYPE2"]:
        
            fits[0].header["CRVAL1"] = params["DEC"]
            fits[0].header["CRVAL2"] = params["RA"]
            
            fits[0].header["CRPIX1"] = y
            fits[0].header["CRPIX2"] = x        
        
        else:
        
            print "%s - RA/DEC not aligned with X/Y axes. WCS correction for this orientation is not yet implemented." % params["IMG_ID"][i]
        
        print fits[0].header["CRVAL1"],fits[0].header["CRVAL2"]
        print fits[0].header["CRPIX1"],fits[0].header["CRPIX2"]
    return fits_list
      
#######################################################################
#Take rotated, stacked images, use center of QSO to align
def wcsAlign(fits_list,params):


    print("Aligning modified cubes using QSO centers")
    
    good_fits,xpos,ypos = [],[],[]
    
    #Calculate positions of QSOs in cropped, rotated, scaled images
    x,y = [],[]
             
    xpos = np.array([f[0].header["CRPIX1"] - f[0].data.shape[2]/2 for f in fits_list])
    ypos = np.array([f[0].header["CRPIX2"] - f[0].data.shape[1]/2 for f in fits_list])
     

    #Calculate offsets from first image
    dx = xpos - xpos[0]
    dy = ypos - ypos[0] 
    
    #Get max size of any image in X and Y dimensions
    cube_shapes = np.array( [ f[0].data.shape for f in fits_list ] )
    Xmax,Ymax = np.max(cube_shapes[:,2]),np.max(cube_shapes[:,1])

    #Get maximum shifts needed in either direction
    dx_max = np.max(np.abs(dx))
    dy_max = np.max(np.abs(dy))
    
    #Create max canvas size needed for later stacking
    Y,X = int(round(Ymax + 2*dy_max + 2)), int(round(Xmax + 2*dx_max + 2))

    for i,fits in enumerate(fits_list):

        #Extract shape and imgnum info
        w,y,x = fits[0].data.shape
        
        #Get padding required to initially center data on canvas
        xpad,ypad = int((X-x)/2), int((Y-y)/2)

        #Create new cube, fill in data and apply shifts
        new_cube = np.zeros( (w,Y,X) )
        new_cube[:,ypad:ypad+y,xpad:xpad+x] = np.copy(fits[0].data)
        
        #Update reference pixel after padding
        fits[0].header["CRPIX1"]  += xpad
        fits[0].header["CRPIX2"]  += ypad
        
        #Using linear interpolation, shift image by sub-pixel values
        new_cube = shift(new_cube,(0,-dy[i],-dx[i]),order=1)
        
        #Update header after shifting
        fits[0].header["CRPIX1"]  -= dx[i]
        fits[0].header["CRPIX2"]  -= dy[i]
        
        #Update data in FITS image
        fits[0].data = np.copy(new_cube)


        
        
    return fits_list
#######################################################################

#######################################################################
#Take rotated, stacked images, use center of QSO to align
def coadd(fits_list,params):
   
    print("Coadding aligned cubes.")
    
    #Create empty stack and exposure mask for coadd
    w,y,x = fits_list[0][0].data.shape
    
    stack = np.zeros((w,y,x))
    exp_mask = np.zeros((y,x))

    header = fits_list[0][0].header

    #Create Stacked cube and fill out mask of exposure times
    for i,fits in enumerate(fits_list):
    
        if params["INST"][i]=="PCWI": exptime = fits[0].header["EXPTIME"]
        elif params["INST"][i]=="KCWI": exptime = fits[0].header["TELAPSE"]
        else:
            print("Bad instrument parameter - %s" % params["INST"][i])
            raise Exception
        
        stack += fits[0].data
        img = np.sum(fits[0].data,axis=0)
        img[img!=0] = exptime
        exp_mask += img
   
    #Divide each spaxel by the exposure count
    for yi in range(y):
        for xi in range(x):
            E = exp_mask[yi,xi]            
            if E>0:
                #if vardata: stack[:,yi,xi] /= E**2 #Variance rules
                stack[:,yi,xi] /= E

    stack_img = np.sum(stack,axis=0)
    
    #Trim off 0/nan edges from grid
    trim_mode = "nantrim"
    if trim_mode=="nantrim": 
        y1,y2,x1,x2 = 0,y-1,0,x-1
        while np.sum(stack_img[y1])==0: y1+=1
        while np.sum(stack_img[y2])==0: y2-=1
        while np.sum(stack_img[:,x1])==0: x1+=1
        while np.sum(stack_img[:,x2])==0: x2-=1
    elif trim_mode=="overlap":
        expmax = np.max(exp_mask)
        y1,y2,x1,x2 = 0,y-1,0,x-1
        while np.max(exp_mask[y1])<expmax: y1+=1
        while np.max(exp_mask[y2])<expmax: y2-=1
        while np.max(exp_mask[:,x1])<expmax: x1+=1
        while np.max(exp_mask[:,x2])<expmax: x2-=1        

    #Crop stacked cube
    stack = stack[:,y1:y2,x1:x2]

    #Update header after cropping
    header["CRPIX1"] -= x1
    header["CRPIX2"] -= y1
    
    return stack,header
