# STAGE 3 - CROP, SCALE, ROTATE, ALIGN AND COADD INDIVIDUAL DATA CUBES
#
#
# FLOW:
# 1. CHECK PARAM FILE EXISTS, RUN SETUP IF NOT
# 2. PARSE PIPELINE PARAM FILE
# 3. GET IMAGE NUMBERS FOR TARGET
# 4. LOAD FITS DATA
# 5. CROP CUBES SPATIALLY AND IN WAVELENGTH
# 6. SCALE CUBES TO 1:1
# 7. ROTATE CUBES TO PA=0
# 8. ALIGN CUBES ON LARGE CANVAS
# 9. COADD CUBES
# 

from scipy.ndimage.interpolation import shift

import astropy
import killer_quickTools
import numpy as np
import matplotlib.pyplot as plt
import sys

#######################################################################
# THIS METHOD STILL VERY SPECIFIC TO PCWI/KCWI DATA
def scale(fits_list,m=4,mode='PCWI'):
    global params,vardata
    #mode='KCWI'
    
    print("Scaling images to a 1:1 aspect ratio")

    #Method for scaling cubes to 1:1 given aspect ratio (r) and short axis (axis)
    def scale_cube(a,r,axis=1):

        #Get shorter axis (one to be scaled)
        axis = np.nanargmin(a.shape)
        
        #Create new array with required shape
        new_shape = np.copy(a.shape)
        new_shape[axis] = int(new_shape[axis]*r)
        a_new = np.zeros(new_shape)
    

        #Need scaling factor for intensity depending on
        R = new_shape[axis]/a.shape[axis]
        
        #Scaling factor squared for variance data
        if vardata: R = R**2
        
        #Run along given axis of new array, assigning values correctly
        for i in range(1,new_shape[axis]+1):

            #Figure out which original indices are contributing to the current pixel
            g1 = round(((i-1)%r)%1.0,2)
            g2 = 1 - g1    
            f1 = round((i%r)%1.0,2)
            f2 = 1-f1

            #If true we are in middle of a single slice (i.e. index)
            if f1==g1 or f1==1.0: 

                #Get slice number
                s = int((i-1)/r)

                #Fill in new array, whichever axis we're using
                if axis==1: a_new[:,i-1,:] = a[:,s,:]/R
                elif axis==2: a_new[:,:,i-1] = a[:,:,s]/R

            #We are in between two original indices/slices
            else: 

                #Get slices (s) and their respective weights (w)
                w1,s1 = f2,int((i-1)/r)
                w2,s2 = f1,int(i/r)
                
                #Fill in new array values
                if axis==1: a_new[:,i-1,:] = (w1*a[:,s1,:] + w2*a[:,s2,:])/R
                elif axis==2: a_new[:,:,i-1] = (w1*a[:,:,s1] + w2*a[:,:,s2])/R

        return a_new
    
    for i,f in enumerate(fits_list):
    
        if params["INST"][i]=='PCWI': yxRatio =  abs(f[0].header["CD1_2"]/f[0].header["CD2_1"] )
        elif params["INST"][i]=='KCWI': yxRatio = abs(f[0].header["CD1_1"]/f[0].header["CD2_2"])
        
        #All cubes are in same orientation at this point, so short axis=1
        f[0].data = scale_cube(f[0].data,yxRatio)

        #Update spatial scale of 'longer' axis to new, smaller scale
        f[0].header["CD1_2"] /= np.round(yxRatio,2)
        f[0].header["CD2_2"] /= np.round(yxRatio,2)        
        f[0].header["CRPIX2"] = int(round(yxRatio*f[0].header["CRPIX2"]))
              
    return fits_list

#######################################################################
#Take 1:1 scaled PCWI images and rotate all to same position angle
def rotate(fits_list):
    
    print("Rotating all images to Position Angle of Zero")
    
    for i,fits in enumerate(fits_list):

        c = fits[0].data #Get data
        w,y,x = c.shape #Cube dimensions
        c_rot = np.zeros((w,x,y)) #Mirror cube for 90deg rotate data
        
        pa = int(fits[0].header["ROTPA"])#OSN"])

        if pa==0: continue         
        elif pa==90:   

            #Rotate +270deg (or -90deg)
            for wi in range(len(c)): c_rot[wi] = np.rot90( c[wi], k=3 ) 
            fits[0].data = c_rot

            #Update header keywords for orientation
            cd1_1 = fits[0].header["CD1_1"]
            cd1_2 = fits[0].header["CD1_2"]
            cd2_1 = fits[0].header["CD2_1"]
            cd2_2 = fits[0].header["CD2_2"]                                    
            fits[0].header["CD1_1"] = -cd1_2
            fits[0].header["CD1_2"] = cd1_1
            fits[0].header["CD2_1"] = -cd2_2
            fits[0].header["CD2_2"] = cd2_1
          
        elif pa==270:

            #Rotate +90deg
            for wi in range(len(c)): c_rot[wi] = np.rot90( c[wi],k=1)
            fits[0].data = c_rot

            #Update header keywords for orientation
            cd1_1 = fits[0].header["CD1_1"]
            cd1_2 = fits[0].header["CD1_2"]
            cd2_1 = fits[0].header["CD2_1"]
            cd2_2 = fits[0].header["CD2_2"]                             
            fits[0].header["CD1_1"] = cd1_2
            fits[0].header["CD1_2"] = cd1_1
            fits[0].header["CD2_1"] = cd2_2
            fits[0].header["CD2_2"] = -cd2_1
                    
        elif pa==180:

            #Update header keywords for orientation
            cd1_1 = fits[0].header["CD1_1"]
            cd1_2 = fits[0].header["CD1_2"]
            cd2_1 = fits[0].header["CD2_1"]
            cd2_2 = fits[0].header["CD2_2"]                                    
            fits[0].header["CD1_1"] = -cd1_1
            fits[0].header["CD1_2"] = -cd1_2
            fits[0].header["CD2_1"] = -cd2_1
            fits[0].header["CD2_2"] = -cd2_2

            #Rotate 180deg
            for wi in range(len(c)): c[wi] = c[wi][::-1] 
            fits[0].data = c

        fits[0].header["ROTPA"] = 0.0

    return fits_list
    
#######################################################################
#Take rotated, stacked images, use center of QSO to align
def align(fits_list):
    global params,setupMode
    
    print("Aligning modified cubes using QSO centers")
    
    good_fits,xpos,ypos = [],[],[]
    
    #Calculate positions of QSOs in cropped, rotated, scaled images
    x,y = [],[]
 
    #If new centers not yet measured and saved
    if setupMode: 
        for i,f in enumerate(fits_list):
            qfinder = killer_quickTools.qsoFinder(f,params["Z"])
            xc,yc = qfinder.run()
            xc -= f[0].data.shape[2]/2
            yc -= f[0].data.shape[1]/2
            params["QSO_XA"][i] = xc
            params["QSO_YA"][i] = yc
            
    xpos = np.array(params["QSO_XA"])
    ypos = np.array(params["QSO_YA"])

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

        #Using linear interpolation, shift image by sub-pixel values
        new_cube = shift(new_cube,(0,-dy[i],-dx[i]),order=1)
        
        #Update data in FITS image
        fits[0].data = np.copy(new_cube)
        fits[0].header["CRVAL1"] = params["RA"]
        fits[0].header["CRVAL2"] = params["DEC"]
        fits[0].header["CRPIX1"] = xpos[0] + X/2
        fits[0].header["CRPIX2"] = ypos[0] + Y/2
        
        
    return fits_list
#######################################################################


#######################################################################
#Take rotated, stacked images, use center of QSO to align
def coadd(fits_list):
    global vardata
    
    print("Coadding aligned cubes.")
    
    #Create empty stack and exposure mask for coadd
    w,y,x = fits_list[0][0].data.shape
    
    stack = np.zeros((w,y,x))
    exp_mask = np.zeros((y,x))
    exp_times = []
    header = fits_list[0][0].header

    #Get median exposure time to compare other exposures to
    #for fits in fits_list: exp_times.append(fits[0].header["XPOSURE"])
    #exp0 = np.median(exp_times)

    #Create Stacked cube and fill out mask of exposure times
    for fits in fits_list:
        stack += fits[0].data
        
        #img = np.sum(fits[0].data,axis=0)
        #img[img!=0] = float(fits[0].header["XPOSURE"])
        #exp_mask += img
   
    #Divide each spaxel by the exposure count
    #for yi in range(y):
    #    for xi in range(x):
    #        E = exp_mask[yi,xi]            
    #        if E>0:
    #            if vardata: stack[:,yi,xi] /= E**2 #Variance rules
    #            else: stack[:,yi,xi] /= E

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
    header["CRPIX1"] -= x1
    header["CRPIX2"] -= y1
     
    return stack,header
#######################################################################


#######################################################################
# MAIN
#######################################################################


#Get user input parameters               
parampath = sys.argv[1]
cubetype = sys.argv[2]

#Check for variance cube input
vardata = True if 'vcube' in cubetype else False

#Check file extension is included in given cube type
if not ".fits" in cubetype: cubetype2 = cubetype+".fits"

#Check if any parameter values are missing (set to set-up mode if so)
params = killer_quickTools.loadparams(parampath)

#Load data
files = killer_quickTools.findfiles(params,cubetype)

fits = [astropy.io.fits.open(f) for f in files] #Open FITS files

#Check if parameters are complete
if killer_quickTools.paramsMissing(params): setupMode = True
else: setupMode = False

#If in setup-mode, fill instrument + header data into params
if setupMode: params = killer_quickTools.parseHeaders(params,fits)


fits = scale(fits) #Scale images to 1:1 aspect ratio    

fits = rotate(fits) #Rotate images to same position Angle            

fits = align(fits) #Align cubes to be stacked - obj.pos file must be created by now

stacked,header = coadd(fits) #Stack cubes and trim   

#Update target parameter file
killer_quickTools.writeparams(params,parampath)

#SAVE STACKED DATA
stackedpath = '%s/%s_%s' % (params["PRODUCT_DIR"],params["NAME"],cubetype)
killer_quickTools.saveFits(stacked,stackedpath,header)
        
