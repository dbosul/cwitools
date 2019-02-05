import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.cluster import KMeans

tabs = []
for tabFile in sys.argv[1:]:
    _in=np.loadtxt(tabFile,usecols=(1,2,3,4,5,6,7))
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
Area  = combTable[:,1]
dWav  = combTable[:,2]
wavCR = combTable[:,3]
R_QSO = combTable[:,4]
I_peak = combTable[:,5]
I_tot = combTable[:,6]

featLabs = ["dWav","Area","w0","R_QSO","I_peak","I_tot"]
feats = [dWav,Area,wavCR,R_QSO,I_peak,I_tot]
feats = np.array(feats)

labels = np.ones_like(I_tot)+1
use =  (I_tot > R_QSO/15) & ( I_tot>12-Area/20.0) #(I_tot>10) & (np.abs(wavCR-1215.7)<=8.1)# & (I_peak>3) #&  & (R_QSO<200) & (Area>100)
labels[~use] = 1
print np.count_nonzero(IDs[labels==2])

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

goodRows = combTable[use]

print("Done clustering")

Nfeats = len(feats)

print("Making figure")
plt.style.use('ggplot')
fig,axes = plt.subplots(Nfeats,Nfeats,figsize=(12,12))
axes[Nfeats-1,0].remove()

for i in range(Nfeats-1):

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
        if featLabs[i+1]=="I_tot" and featLabs[j]=="R_QSO": vsAx.plot([0,300],[0,22.5],'k--')
        elif featLabs[i+1]=="I_tot" and featLabs[j]=="Area": vsAx.plot([0,240],[15,0],'k--')
        for lab in labels_unique: vsAx.plot(feats[j][labels==lab],feats[i+1][labels==lab],'o',alpha=0.7)
        vsAx.set_xticks([])
        vsAx.set_yticks([])
        #vsAx.set_title("%s v. %s" % (featLabs[i+1],featLabs[j]))
        
    for j in range(i+1,Nfeats-1):
        axes[i,j+1].remove()
fig.tight_layout()
fig.show()

raw_input("")
