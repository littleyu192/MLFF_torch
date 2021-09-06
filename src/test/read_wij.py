#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run this code in working dir, it will read the file ./fread_dfeat/NN_output/NNFi/Wij.npy and output txt to fread_dfeat/Wij.txt

import os
import sys
sys.path.append(os.getcwd())
import parameters as pm
codepath=os.path.abspath(sys.path[0])
print(codepath)
sys.path.append(codepath+'/../model')
from FC import MLFFNet
import numpy as np
import torch

#f_npfile='data_scaler.npy'
def pt2npy(pt_file):
    npy_file=os.path.join(pm.fitModelDir,'NN_output/NNFi/Wij.npy')

    checkpoint = torch.load(pt_file)
    weights = checkpoint['model']
    layer_name = []
    for weight_name in weights.keys():
        layer_name.append(weight_name)
        # print(weight_name)
        # print(weights[weight_name].shape)

    wij = []
    for i in range(len(layer_name)):
        model_w = weights[layer_name[i]].cpu().numpy()
        if model_w.ndim == 2:
            wij.append(model_w.T)
        elif model_w.ndim == 1:
            wij.append(np.expand_dims(model_w.T,0))
        else:
            pass
        print(f"wij shape:{np.squeeze(weights[layer_name[i]].T.cpu().numpy()).shape}")
    # for i in range(len(layer_name)):
    #     print(wij[i].shape)
    np.save(npy_file, np.array(wij,dtype=object))

    
def read_wij():
    f_npfile=os.path.join(pm.fitModelDir,'NN_output/NNFi/Wij.npy')
    dsnp = np.load(f_npfile, allow_pickle=True)
    with open(os.path.join(pm.fitModelDir,'Wij.txt'),'w') as f:
        f.writelines('test '+str(f_npfile)+'  '+str(dsnp.dtype)+'  '+str(dsnp.shape)+'\n')
        f.writelines('shape '+str(dsnp.shape)+'\n')
        f.writelines('dim '+str(dsnp.ndim)+'\n')
        f.writelines('size '+str(dsnp.size)+'\n')
        n=dsnp.size
        typesize=int(dsnp.shape[0]/pm.ntypes)
        count = 0
        while count <= n-1:
            a=dsnp[count]
            m1=a.shape[0]
            if len(a.shape)== 2:
                m2=a.shape[1]
                f.writelines('m12= '+str(m1)+', '+str(m2)+', '+str(int(count/typesize)+1)+'\n')
                for i in range(0,m1):
                    for j in range(0,m2):
                        f.writelines(str(i)+'  '+str(j)+'  '+str(a[i,j])+'\n')
                #  f.writelines(a)
                count = count+1
            else:
                f.writelines('m12= '+str(m1)+', '+str(0)+', '+str(int(count/typesize)+1)+'\n')
                for i in range(0,m1):
                    # for j in range(0,0):
                    f.writelines(str(i)+'  '+str(0)+'  '+str(a[i])+'\n')
                #  f.writelines(a)
                count = count+1

def read_scaler():
    f_npfile=os.path.join(pm.fitModelDir,'NN_output/NNFi/data_scaler.npy')
    dsnp = np.load(f_npfile, allow_pickle=True)
    with open(os.path.join(pm.fitModelDir,'data_scaler.txt'),'w') as f:
        f.writelines('test '+str(f_npfile)+'  '+str(dsnp.dtype)+'  '+str(dsnp.shape)+'\n')
        f.writelines('shape '+str(dsnp.shape)+'\n')
        f.writelines('dim '+str(dsnp.ndim)+'\n')
        f.writelines('size '+str(2*pm.ntypes)+'\n')
        n=dsnp.size
        typesize=int(dsnp.shape[0]/pm.ntypes)
        count = 0
        while count <= n-1:
            if count % typesize == 1 or count % typesize == 2:
                a=dsnp[count]
                m1=a.shape[0]
                if len(a.shape)== 2:
                    m2=a.shape[1]
                    f.writelines('m12= '+str(m1)+', '+str(m2)+', '+str(int(count/typesize)+1)+'\n')
                    for i in range(0,m1):
                        for j in range(0,m2):
                            f.writelines(str(i)+'  '+str(j)+'  '+str(a[i,j])+'\n')
                    #  f.writelines(a)
                    count = count+1
                else:
                    f.writelines('m12= '+str(m1)+', '+str(0)+', '+str(int(count/typesize)+1)+'\n')
                    for i in range(0,m1):
                        # for j in range(0,0):
                        f.writelines(str(i)+'  '+str(0)+'  '+str(a[i])+'\n')
                    #  f.writelines(a)
                    count = count+1
            else:
                count=count+1

if __name__ =="__main__":
    pt_file=os.path.join(pm.codedir,'NN_output/NNFi/data_scaler.npy')
    # pt_file = "./FC3model_mini_force/3layers_MLFFNet.pt"
    pt_file = sys.argv[1]
    pt2npy(pt_file)
    read_wij()
    read_scaler()