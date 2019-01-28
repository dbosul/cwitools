import numpy as np

from astropy.cosmology import WMAP9 as cosmo
from CWITools.libs import cubes
from scipy.ndimage.measurements import center_of_mass

import matplotlib.pyplot as plt

class Obj3D():
    
    def __init__(self,inFits,idFits,ID):

        #Isolate 3D object in input and ID cubes
        inCube = inFits[0].data.copy()
        idCube = idFits[0].data.copy()
        inCube[idCube!=ID] = 0
        idCube[idCube!=ID] = 0   

        #Get projections of 3D object along each axis
        objZ = np.sum(idCube,axis=(1,2))>0
        objY = np.sum(idCube,axis=(0,2))>0
        objX = np.sum(idCube,axis=(0,1))>0
             
        #Save ID of object
        self.id = ID
 
        #Extract subcube from input cube
        self.cube = ((inCube.copy()[objZ])[:,objY])[:,:,objX]
                
        #Create 2D binary mask of object image
        self.maskXY = np.sum(idCube,axis=0)
        self.maskXY[self.maskXY>0] = 1
        
        #Create 1D binary mask of object spectrum
        self.maskZ = np.sum(idCube,axis=(1,2))
        self.maskZ[self.maskZ>0] = 1
         
        #Save number of voxels/area/wavelength
        self.nXYZ = np.count_nonzero(idCube)
        self.nXY  = np.count_nonzero(self.maskXY)
        self.nZ   = np.count_nonzero(self.maskZ)
           
        #Create 2D image of object (To do: adjust units correctly)
        self.image = np.sum(self.cube,axis=0)

        #Get center of mass in XY
        self.comYX = center_of_mass(self.image)
        self.comYX = ( max( 0, min(self.cube.shape[1],self.comYX[0]) ), max( 0, min(self.cube.shape[2],self.comYX[1]) ) )

        #Create spectrum summed over detected spaxels
        self.spectrum = np.sum( self.cube, axis=(1,2) )
        
        #Get Z center of mass of object
        self.comZ = center_of_mass(self.spectrum)[0]
        self.comZ = min(self.comZ,len(self.spectrum)-1)
        self.comZ = max(self.comZ,0)
        

        
        
