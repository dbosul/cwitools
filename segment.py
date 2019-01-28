import numpy as np
import sys

from astropy.io import fits
from scipy.stats import norm
from skimage import measure

aksPath = sys.argv[1]

print("\nSegmenting %s"%aksPath)

aksFits = fits.open(aksPath)
AKS = aksFits[0].data.copy()
AKS[AKS>0] = 1

LAB = measure.label(AKS)
objFits = fits.HDUList([fits.PrimaryHDU(LAB)])
objFits[0].header = aksFits[0].header

objPath = aksPath.replace('.fits','.OBJ.fits')
aksFits[0].data=LAB
aksFits.writeto(objPath,overwrite=True)

s_vals = [2,2.5,3.0,3.5,4.0,4.5,5.0]
p_vals = norm.sf(s_vals)

print("%5s %8s" % ("Sigma","#FalseP"))
for i in range(len(s_vals)):
    s,p = s_vals[i],p_vals[i]
    N = int(round(p*LAB.size))
    print("%5.2f %8i" % (s,N))
    
Nexp=int(round((0.0233/100)*LAB.size))
print "%i objects detected." % np.max(LAB)
print("Wrote %s"%objPath)
