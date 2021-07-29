import numpy as np
import random
import io
import os


inputfile="fit.input"
outputfile="GPR_para_error"
ref_num=[1500,1000]
fr=0.1
alpha=[1]
dist0=[1.45,1.55]
os.system("../linear_forceM.r")
os.system("../calc_lin_forceM.r")
with open(outputfile,'a+') as f2:
    f2.write('reference num,force ratio: '+str(ref_num)+' '+str(fr)+'\n')
    f2.write('linear results ')
    filename="energyL.pred.1"
    data = np.loadtxt(filename)
    # plt.subplot(lenx,leny,k)
    error=np.sum((data[:,1]-data[:,0])**2,axis=0)
    f2.write(str(error)+' ')
    filename="energyL.pred.2"
    data = np.loadtxt(filename)
    # plt.subplot(lenx,leny,k)
    error=np.sum((data[:,1]-data[:,0])**2,axis=0)
    f2.write(str(error)+' ')
    filename="forceL.pred.1"
    data = np.loadtxt(filename)
    # plt.subplot(lenx,leny,k)
    error=np.sum((data[:,1]-data[:,0])**2,axis=0)
    f2.write(str(error)+' ')     
    filename="forceL.pred.2"
    data = np.loadtxt(filename)
    # plt.subplot(lenx,leny,k)
    error=np.sum((data[:,1]-data[:,0])**2,axis=0)
    f2.write(str(error)+' '+'\n') 

with open(inputfile,'r') as f:
    get_all=f.readlines()

for s in alpha:
    for dis in dist0:
        with open(inputfile,'w') as f:
            for k,line in enumerate(get_all):         ## STARTS THE NUMBERING FROM 1 (by default it begins with 0)                
                if k == 3:                              ## OVERWRITES line:1
                    f.writelines(str(s)+","+str(dis)+"             ! alpha,dist0 (for kernel)\n")          
                else:           
                    f.writelines(line)
        os.system("../GPR_fit_forceM.r")
        os.system("../calc_E_forceM.r")

        with open(outputfile,'a+') as f2:
            f2.write(str(s)+' '+str(dis)+' ')
            filename="energy.pred.1"
            data = np.loadtxt(filename)
            # plt.subplot(lenx,leny,k)
            error=np.sum((data[:,1]-data[:,0])**2,axis=0)
            f2.write(str(error)+' ')
            filename="energy.pred.2"
            data = np.loadtxt(filename)
            # plt.subplot(lenx,leny,k)
            error=np.sum((data[:,1]-data[:,0])**2,axis=0)
            f2.write(str(error)+' ')
            filename="force.pred.1"
            data = np.loadtxt(filename)
            # plt.subplot(lenx,leny,k)
            error=np.sum((data[:,1]-data[:,0])**2,axis=0)
            f2.write(str(error)+' ')     
            filename="force.pred.2"
            data = np.loadtxt(filename)
            # plt.subplot(lenx,leny,k)
            error=np.sum((data[:,1]-data[:,0])**2,axis=0)
            f2.write(str(error)+' '+'\n')                           



