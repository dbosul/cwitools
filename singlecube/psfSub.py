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
d = d[:,x-r:x+r,y-r:y+r]

w0,dw,p0 = h["CRVAL3"],h["CD3_3"],h["CRPIX3"]
w0 -= p0*dw
####
cont_image = np.zeros_like(d[0])
for (w1,w2) in wavpairs:
	a,b = ( int((w1-w0)/dw), int((w2-w0)/dw) )
	print a,b
	cont_image += np.sum(d[a:b],axis=0)
cont_image/=np.sum( [ B-A for (A,B) in wavpairs ] )

####


fitter = fitting.LevMarLSQFitter()

XX,YY = np.meshgrid(np.arange(-r,+r),np.arange(-r,+r))
gg_init = models.Gaussian2D(amplitude=np.max(cont_image)) + models.Const2D(np.median(cont_image))
gg_fit = fitter(gg_init,XX,YY,cont_image)
plt.figure()
plt.subplot(311)
plt.pcolor(cont_image)
plt.colorbar()
plt.subplot(312)
plt.pcolor(gg_fit(XX,YY))
plt.colorbar()
plt.subplot(313)
plt.pcolor(cont_image-gg_fit(XX,YY))
plt.colorbar()
plt.show()


for wi in range(len(d)):
          
    scale_init = models.Scale(factor=np.max(d[wi])/np.max(cont_image))
    scale_fit = fitter(scale_init,cont_image.flatten(),d[wi].flatten())
    model = scale_fit.factor.value*cont_image #Add this wavelength layer to the model
    f[0].data[wi,x-r:x+r,y-r:y+r] -= model
f.writeto("test.fits",overwrite=True)
