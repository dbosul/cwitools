from astropy.io import fits
from killer.fits3D import HDUList3D
from killer.tools.qso import qsoFinder
 
#Open up an image at 90deg PA
f = HDUList3D.fits3D("/home/donal/data/kcwi/171122/redux/kb171122_00127_icubed.fits")
redshift = 2.790
RA = 168.2185417
DEC = 15.3565278

qF = qsoFinder(f,redshift)
x,y = qF.run()

h = f[0].header
if "RA" in h["CTYPE1"] and "DEC" in h["CTYPE2"]:
         
    f[0].header["CRVAL1"] = RA
    f[0].header["CRVAL2"] = DEC
    
    f[0].header["CRPIX1"] = x
    f[0].header["CRPIX2"] = y
    
elif "DEC" in h["CTYPE1"] and "RA" in h["CTYPE2"]:

    f[0].header["CRVAL1"] = DEC
    f[0].header["CRVAL2"] = RA
    
    f[0].header["CRPIX1"] = y
    f[0].header["CRPIX2"] = x  
    
else: print "womp"



f.save("sdss1112_wcsCor.fits")


f.crop(xx=(2,-2),yy=(18,80),ww=(200,300))
f.scale1to1()
f.rotate90(N=1)


f.save("sdss1112_scale.fits")
