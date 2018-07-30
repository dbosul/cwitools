from astropy.io import fits as fitsIO
from astropy.wcs import WCS
import numpy as np
import pyregion
import sys

fitspath = sys.argv[1]
regpath = sys.argv[2]
if len(sys.argv)>3: R = float(sys.argv[3])
else: R = 2.0

fits = fitsIO.open(fitspath)
regfile = pyregion.open(regpath)

#EXTRACT/CREATE USEFUL VARS############
data3D = fits[0].data
head3D = fits[0].header

W,Y,X = data3D.shape #Dimensions
mask = np.zeros((Y,X),dtype=int) #Mask to be filled in
x,y = np.arange(X),np.arange(Y) #Create X/Y image coordinate domains
xx, yy = np.meshgrid(x, y) #Create meshgrid of X, Y
ww = np.array([ head3D["CRVAL3"] + head3D["CD3_3"]*(i - head3D["CRPIX3"]) for i in range(W)])

#BUILD MASK############################
if regfile[0].coord_format=='image':

    rr = np.sqrt( (xx-x0)**2 + (yy-y0)**2 )
    mask[rr<=R] = i+1          
                
elif regfile[0].coord_format=='fk5':  

    head2D = head3D.copy() #Create a 2D header by modifying 3D header
    for key in ["NAXIS3","CRPIX3","CD3_3","CRVAL3","CTYPE3","CNAME3","CUNIT3"]: head2D.remove(key)
    head2D["NAXIS"]=2
    head2D["WCSDIM"]=2
    wcs = WCS(head2D)    
    ra, dec = wcs.wcs_pix2world(xx, yy, 0) #Get meshes of RA/DEC
    
    for i,reg in enumerate(regfile):    
    
        ra0,dec0,r = reg.coord_list #Extract location and default radius    
        rr = 3600*np.sqrt( (np.cos(dec*np.pi/180)*(ra-ra0))**2 + (dec-dec0)**2 ) #Create meshgrid of distance to source 
                    
        if np.min(rr) > R: continue #Skip any sources more than one radius outside the FOV
        
        else: mask[rr < R] = i+1 #Label region

  
#Just replace with zeros
for wi in range(fits[0].data.shape[0]): fits[0].data[wi][mask>0] = 0

outpath = fitspath.replace('.fits','.m.fits')
fits.writeto(outpath,overwrite=True)

print "Saved %s" % outpath
