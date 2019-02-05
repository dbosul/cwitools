from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import ndimage
import numpy as np
import sys

#Take any additional input params, if provided
settings = {"zWindow":10,"rescale":True}
if len(sys.argv)>3:
    for item in sys.argv[3:]:      
        key,val = item.split('=')
        if settings.has_key(key):
            if key in ["zWndow"]: val=int(val)
            elif key in ["rescale"]: val=bool(val)
            settings[key]=val
        else:
            print "Input argument not recognized: %s" % key
            sys.exit()
            
fitsPath = sys.argv[1]

zWindow = settings["zWindow"]
rescale = settings["rescale"]

#Extract
F = fits.open(fitsPath)
D = F[0].data
#D = sigma_clip(D,sigma=3).data

V = np.zeros_like(D)
i   = 0
a,b = (i*zWindow), (i+1)*zWindow

while b < D.shape[0]: 
    V[a:b] = np.var(D[a:b],axis=0) 
    i+=1
    a,b = (i*zWindow), (i+1)*zWindow
    
V[a:] = np.var(D[a:],axis=0)

Rmax,Rmin = 0.9,10
for wi in range(len(V)):
    varD = np.std(sigma_clip(D[wi]))**2
    varV = np.mean(V[wi])
    rFac = varV/varD

    print rFac
    V[wi] /= rFac
varPath = fitsPath.replace('.fits','.var.fits')
F[0].data = V
F.writeto(varPath,overwrite=True)
print("Saved %s"%varPath)

