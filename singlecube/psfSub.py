from astropy.io import fits
from astropy.modeling import models,fitting
import matplotlib.pyplot as plt
import numpy as np
import sys

####
file_in = sys.argv[1]
wavpairs = [ [ float(x) for x in w.split(':') ] for w in sys.argv[2].split(',')]
x,y = [ int(x) for x in sys.argv[3].split(',') ]

####
r = 10
f = fits.open(file_in)
h = f[0].header
d = f[0].data


w0,dw,p0 = h["CRVAL3"],h["CD3_3"],h["CRPIX3"]
w0 -= p0*dw
dpx = 0.55

cont_image = np.zeros_like(d[0])
for (w1,w2) in wavpairs:
	a,b = ( int((w1-w0)/dw), int((w2-w0)/dw) )
	print a,b
	cont_image += np.sum(d[a:b],axis=0)
cont_image/=np.sum( [ B-A for (A,B) in wavpairs ] )
cont_image -=np.median(cont_image)

cont_image = cont_image[x-r:x+r+1,y-r:y+r+1]
d = d[:,x-r:x+r+1,y-r:y+r+1]

fitter = fitting.LevMarLSQFitter()

XX,YY = np.meshgrid(np.arange(-r,+r+1),np.arange(-r,+r+1))
useIm = (XX**2 + YY**2)< 60/dpx

for wi in range(len(d)):
          
    scale_init = models.Scale(factor=np.max(d[wi])/np.max(cont_image))

    
    layer = d[wi].copy()
    layer-=np.median(layer)
    layer = layer[useIm]
    
    scale_fit = fitter(scale_init,cont_image[useIm].flatten(),layer.flatten())
    model = scale_fit.factor.value*cont_image#Add this wavelength layer to the model
    model[useIm==0] = 0
    
    f[0].data[wi,x-r:x+r+1,y-r:y+r+1] -= model #-= model
f.writeto("test.fits",overwrite=True)
