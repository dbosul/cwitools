from astropy.io import fits
import numpy as np
import sys

#Take in table data
tableIn  = sys.argv[1]
objTable = np.loadtxt(tableIn,usecols=(1,2,3,4,5,6,7,8))

#Extract shorthand references to x,y,z coordinates of objects
IDS = objTable[:,0]
Z = objTable[:,3]
X = objTable[:,6]
Y = objTable[:,7]

#Keep track of object IDs which have already been matched up with another
matched = np.zeros_like(Z,dtype=bool)

#This array will keep track of the updated ID for each object
newID = objTable[:,0].copy()

#Loop over objects
for i,obj in enumerate( objTable ):
    
    #Get ID and coordinates
    _id,z0,x0,y0 = obj[0],obj[3],obj[6],obj[7]
    
    #Check for z difference smaller than 50 km/s
    zNeighbs = abs(objTable[:,3]-obj[3]) <= 0.4#(50.0/3e5)*1215.7
    
    #Check for xy distance less than 10 px
    rNeighbs = np.sqrt( (objTable[:,6]-obj[6])**2 + (objTable[:,7]-obj[7])**2 ) <= 16
    
    #Combine these two conditions
    matches = rNeighbs & zNeighbs
    
    #Ignore self
    matches[i] = False
    
    #Get number of matches
    N = np.count_nonzero(matches)
    
    
    if N>0:

        minMatchID = min(_id,np.min(newID[matches]))
        addMatches = newID==minMatchID
        addMatches[i] = False
        
        matches = addMatches | matches
        
        newID[matches] = minMatchID

        print _id,IDS[matches],newID[matches]
        
#Apply changes to .OBJ.fits
objFits = fits.open(tableIn.replace('.tab','.fits'))
objData = objFits[0].data 
for i in range(len(IDS)): objData[objData==IDS[i]] = newID[i]
objFits[0].data = objData
objFits.writeto(tableIn.replace('.tab','.rec.fits'),overwrite=True)
print("Saved %s"%tableIn.replace('.tab','.rec.fits'))
