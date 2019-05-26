import numpy as np
import sys

from astropy.io import fits as fIO
from astropy.wcs import WCS

#Take user input
path  = sys.argv[1]
binXY = int(sys.argv[2])
binW  = int(sys.argv[3])

#Optional user settings
settings = {"varData":False,"outName":None}
if len(sys.argv)>4:
    for sysArg in sys.argv[4:]:
        key,val=sysArg.split('=')
        try: settings[key] = val
        except: print(("Input argument %s not understood."%sysArg))
        
#Load and extract data
fits = fIO.open(path)
data = fits[0].data
head = fits[0].header

#Get dimensions & Wav array
w,y,x = data.shape
W = np.array([ head["CRVAL3"] + (i-head["CRPIX3"])*head["CD3_3"] for i in range(w) ])

#Get new sizes
wnew = w/binW  + 1 if binW >1 else w
ynew = y/binXY + 1 if binXY>1 else y
xnew = x/binXY + 1 if binXY>1 else x

#Perform wavelenght-binning first, if bin provided
if binW>1:
    
    #Get new bin size in Angstrom
    wBinSize = binW*head["CD3_3"]  
    
    #Create new data cube shape 
    data_W = np.zeros((wnew,y,x))
    
    #Run through all input wavelength layers and add to new cube
    for wi in range(w): data_W[ int(wi/binW) ] += data[wi]
    
    #Normalize so that units remain as "erg/s/cm2/A"   
    if settings["varData"]: data_W /= binW**2
    else: data_W /= binW 
    
    #Update central reference and pixel scales
    head["CD3_3"] *= binW
    head["CRPIX3"] /= binW
    
else: data_W = data

#Perform spatial binning next
if binXY>1:
    
    #Get new shape
    data_XY = np.zeros((wnew,ynew,xnew))

    #Run through spatial pixels and add
    for yi in range(y):
        for xi in range(x):        
           data_XY[:,yi/binXY,xi/binXY] += data_W[:,yi,xi]

    #
    # No normalization needed for binning spatial pixels.
    # Units remain as 'per pixel' but pixel size changes.
    #
    
    #Update pixel scales
    head["CRPIX1"] /= float(binXY)
    head["CRPIX2"] /= float(binXY)


    #Update central reference pixel
    head["CD1_1"]  *= binXY
    head["CD1_2"]  *= binXY
    head["CD2_1"]  *= binXY
    head["CD2_2"]  *= binXY

else: data_XY = data_W

if settings["outName"]==None: settings["outName"] = path.replace(".fits",".{0}_{1}.fits".format(binXY,binW))
newFITS = fIO.HDUList( [ fIO.PrimaryHDU(data_XY) ] )
newFITS[0].header = head
newFITS.writeto(settings["outName"],overwrite=True)
print(("Saved %s"%settings["outName"]))


