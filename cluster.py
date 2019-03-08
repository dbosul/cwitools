import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.cluster import KMeans

tabs = []
for tabFile in sys.argv[1:]:
    _in=np.loadtxt(tabFile,usecols=(1,2,3,4,5,6,7,8))
    if _in.shape==(7,): _in = np.array([_in])
    if len(_in)>0: tabs.append(_in)
combTable = np.concatenate(tabs)
#IDs = np.concatenate([ np.loadtxt(tabFile,usecols=(1)) for tabFile in sys.argv[1:] ])
#inputTabs = [ np.loadtxt(tabFile,usecols=(2,3,4,5,6,7)) for tabFile in sys.argv[1:] ]
#combTable = np.concatenate(inputTabs,axis=0)

#IDs = IDs[combTable[:,1]<50]
#combTable = combTable[combTable[:,1]<50]

#IDs = IDs[combTable[:,1]<1000]
#combTable = combTable[combTable[:,1]<1000]

IDs   = combTable[:,0]
Nvox  = combTable[:,1]
Area  = combTable[:,2]
dWav  = combTable[:,3]
wavCR = combTable[:,4]
R_QSO = combTable[:,5]
I_peak = combTable[:,6]
I_tot = combTable[:,7]

featLabs = ["dWav","Nvox","Area","w0","R_QSO","I_peak","I_tot"]
I_tot[I_tot>50] = 0
feats = [dWav,Nvox,Area,wavCR,R_QSO,I_peak,I_tot]


labels = np.ones_like(I_tot)+1
use =  (np.abs(wavCR.copy()-1215.7)<=4.1)
labels[~use] = 1

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

goodRows = combTable[use]

print("Done clustering")

Nfeats = len(feats)

print("Making figure")
plt.style.use('ggplot')
fig,axes = plt.subplots(Nfeats,Nfeats,figsize=(12,12))

for i,f in enumerate(feats):
    feats[i]=np.nan_to_num(feats[i])
    feats[i] = feats[i][use]
feats = np.array(feats)
for i in range(Nfeats-1):

    print np.max(feats[i+1]),np.min(feats[i+1]),feats[i+1]
    bottomAx = axes[Nfeats-1,i+1]    
    bottomAx.hist(feats[i],facecolor='black')
    

    bottomAx.set_yticks([])
    bottomAx.set_xlabel(featLabs[i])
    
    leftAx = axes[i,0]
    leftAx.hist(feats[i+1],orientation='horizontal',facecolor='black')
    
    leftAx.set_xticks([])
    leftAx.set_ylabel(featLabs[i+1])
    
    for j in range(i+1):
        

        vsAx = axes[i,j+1] 
        #if featLabs[i+1]=="I_tot" and featLabs[j]=="R_QSO": vsAx.plot([0,300],[0,22.5],'k--')
        #elif featLabs[i+1]=="I_tot" and featLabs[j]=="Area": vsAx.plot([0,240],[15,0],'k--')
        vsAx.plot(feats[j],feats[i+1],'o',alpha=0.7)
        vsAx.set_xticks([])
        vsAx.set_yticks([])
        #vsAx.set_title("%s v. %s" % (featLabs[i+1],featLabs[j]))
        
    for j in range(i+1,Nfeats-1):
        axes[i,j+1].remove()
fig.tight_layout()
fig.show()

raw_input("")
