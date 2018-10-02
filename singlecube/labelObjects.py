import numpy as np
import matplotlib.pyplot as plt
import skimage
import sys

from astropy.io import fits as fitsIO
from skimage import measure


path = sys.argv[1]

#print "Finding objects in %s" % path
print "%20s\t" % path,
sys.stdout.flush()

try: fits = fitsIO.open(path)
except: print "File not found."
D = fits[0].data.copy()
B = D.copy()
B[B>0] = 1
L = measure.label(B)

fits[0].data=D
fits.writeto(sys.argv[1].replace('.fits','.OBJ.fits'),overwrite=True)

#Temp code to measure and print Area,SpecExtent,Intensity of "large blobs" in images
Lflat = L.flatten()
M = np.zeros_like(Lflat)
r  = 1.5
dw = 3

for label in range(1,np.max(L)+1):

    C = D.copy().flatten()
    C[Lflat!=label] = 0 #Isolate this one object
    C = np.reshape(C,D.shape)
    
    Cimg = np.sum(C,axis=0)
    Cspec = np.sum(np.sum(C,axis=1),axis=1)
    
    Cimg[Cimg>0] = 1
    area = np.sum(Cimg)
    
    Cspec[Cspec>0] = 1
    wExt = np.sum(Cspec)
    
    if area>np.pi*r**2 and wExt>dw:
        
        M[Lflat==label] = 1 #Mark for use

dataFlat = fits[0].data.copy().flatten()
dataFlat[M==0] = 0
dataNew = np.reshape(dataFlat,fits[0].data.shape)
fits[0].data = dataNew
fits.writeto(sys.argv[1].replace('.fits','.LG.fits'),overwrite=True)


M = np.reshape(M,D.shape)

Mimg = np.sum(M,axis=0)
Mimg[Mimg>0] = 1
Area = np.sum(Mimg)*(0.5)**2

Mspec = np.sum(np.sum(M,axis=1),axis=1)
Mspec[Mspec>0] = 1
wExt = np.sum(Mspec)*0.55

Dflat = D.flatten()
Dflat[M.flatten()==0]=0
D2 = np.reshape(Dflat,D.shape)
Int  = 0.55*np.sum( D2 )
targ=path.split('/')[-1].split('.')[0]


print "%10.2f\t%10.2f\t%10.2f" % (Area,wExt,Int)
sys.stdout.flush()

plt.style.use('ggplot')
fig = plt.figure()
plt.title(targ)
plt.subplot(211)
plt.pcolor(np.sum(D2,axis=0))
plt.colorbar()
plt.subplot(212)
plt.plot(np.sum(D2,axis=(1,2)))
plt.tight_layout()
fig.savefig("/home/donal/data/flashes/meta/%s.png" % targ)

plt.close()

