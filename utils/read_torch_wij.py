#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run this code in working dir, it will read the file ./fread_dfeat/NN_output/NNFi/Wij.npy and output txt to fread_dfeat/Wij.txt

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../src/lib')
sys.path.append(os.getcwd())
import parameters as pm
#import parse_input
#parse_input.parse_input()
# every py file in git_version/bin/ should be invoke directly from shell, and must parse parameters.

#f_npfile='data_scaler.npy'

# nfeature = 42, and net info [60, 30, 1]
# 42 -> 60 -> 30 -> 1
# torch wij shape [60,42], [30,60], [1,30]
# torch bj  shape [60], [30], [1]
# Wij.txt structure: w0, b0, w1, b1, w2, b2
#   [42,60], [60], [60,30], [30], [30,1], [1]
def read_wij():
    #print (os.getcwd())
    #return
    pt_file = os.getcwd() +r'/record/model/latest.pt' 
    #print (pt_file)
    #return 
    #pt_file=os.path.join(pm.fitModelDir,'3layers_preMLFFNet.pt')
    
    chpt = torch.load(pt_file,map_location=torch.device('cpu'))
    nn_model = chpt['model']
    nlayers = len(nn_model) // pm.ntypes // 2
    #print('nlayers %d' % (nlayers))
    info_net = []
    for i in range(nlayers):
        info_net.append(np.array(nn_model[r'models.0.weights.'+str(i)].shape))

    wij_all = [ [ np.zeros((info_net[ilayer]),dtype=float) for ilayer in range(nlayers) ] for itype in range(pm.ntypes)]
    bij_all = [ [ np.zeros((info_net[ilayer]),dtype=float) for ilayer in range(nlayers) ] for itype in range(pm.ntypes)]
    with open(os.path.join(pm.fitModelDir,'Wij.txt'),'w') as f:
        f.write('test ' + str(pt_file) + '  \n')
        f.write('shape '+str(nlayers*2)+'\n')
        f.write('dim 1'+'\n')
        f.write('size '+str(nlayers*2)+'\n')
        count = 0
        for itype in range(pm.ntypes):
            for ilayer in range(nlayers):
                wij_all[itype][ilayer] = nn_model[r'models.'+str(itype)+r'.weights.'+str(ilayer)]
                bij_all[itype][ilayer] = nn_model[r'models.'+str(itype)+r'.bias.'+str(ilayer)]
                wij = wij_all[itype][ilayer]
                bij = bij_all[itype][ilayer]
                m1 = info_net[ilayer][1]
                m2 = info_net[ilayer][0]
                #print('wij_shape:')
                #print(wij.shape)
                #print('m1 m2: %d %d' % (m1, m2))
                # write wij
                f.write('m12= '+str(m1) + ',' +str(m2)+', 1\n')
                for i in range(0,m1):
                    for j in range(0,m2):
                        f.write(str(i)+'  '+str(j)+'  '+str(float(wij[j][i]))+'\n')
                # write bj
                f.write('m12= 1,' + str(m2)+', 1\n')
                for j in range(0,m2):
                    f.write(str(j)+',  0  '+str(float(bij[j]))+'\n')


def read_scaler():
    f_npfile=os.path.join(pm.fitModelDir,'NN_output/NNFi/data_scaler.npy')
    chpt = np.load(f_npfile, allow_pickle=True)
    with open(os.path.join(pm.fitModelDir,'data_scaler.txt'),'w') as f:
        f.writelines('test '+str(f_npfile)+'  '+str(chpt.dtype)+'  '+str(dsnp.shape)+'\n')
        f.writelines('shape '+str(chpt.shape)+'\n')
        f.writelines('dim '+str(chpt.ndim)+'\n')
        f.writelines('size '+str(2*pm.ntypes)+'\n')
        n=chpt.size
        typesize=int(chpt.shape[0]/pm.ntypes)
        count = 0
        while count <= n-1:
            if count % typesize == 1 or count % typesize == 2:
                a=chpt[count]
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
    read_wij()
    #read_scaler()
