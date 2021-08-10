#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import sys, os
sys.path.append(os.getcwd())
import parameters as pm    
# import prepare as pp
# pp.readFeatnum()
if pm.torch_dtype == 'float32':
    torch_dtype = torch.float32
    print('info: torch.dtype = torch.float32 in Pytorch trainning.')
else:
    torch_dtype = torch.float64
    print('info: torch.dtype = torch.float64 in Pytorch trainning. (it may be slower)')


################################################################
# fully connection nn
# Ei Neural Network
################################################################

ACTIVE = torch.relu
B_INIT= -0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FCNet(nn.Module):
    def __init__(self,BN = False,Dropout = False, itype = 0):  #atomtypes=len(pm.atomType)
        super(FCNet,self).__init__()
        self.dobn = BN
        self.dodrop = Dropout                                  
        self.fcs=[]
        self.bns=[]
        self.drops=[]
        self.itype= itype  # itype =0,1 if CuO                                       
        for i in range(pm.nLayers-1):
            in_putsize = pm.nFeatures if i==0 else pm.nNodes[i-1,itype] #0为type1,1为type2
            fc = nn.Linear(in_putsize,pm.nNodes[i,itype])
            setattr(self,'fc%i'%i,fc)   #setattr函数用来设置属性，其中第一个参数为继承的类别，第二个为名称，第三个是数值
            self.fcs.append(fc)
            self.__set__init(fc)  #初始化网络训练参数
            if self.dobn:
                bn = nn.BatchNorm1d(pm.nNodes[i,itype], momentum=0.5)
                setattr(self,'bn%i'%i,bn)
                self.bns.append(bn)
            if self.dodrop:
                drop = nn.Dropout(0.5)
                setattr(self,'drop%i'%i,drop)
                self.drops.append(drop)
        self.output = nn.Linear(pm.nNodes[pm.nLayers-2,itype],1)  #最后一层
        self.__set__init(self.output)

    def __set__init(self,layer):
        init.normal_(layer.weight, mean = 0, std = 1)       
        init.constant_(layer.weight,val=B_INIT)

    def forward(self, x):
        input = x
        for i in range(pm.nLayers-1):                         
            x = self.fcs[i](x)
            if self.dobn:       
                x = self.bns[i](x)
            if self.dodrop:              
                self.drops[i](x)
        x = ACTIVE(x)         #激活函数，可以自定义
        predict = self.output(x)  #网络的最后一层
        return input, predict
# nets = [FCNet(),FCNet(BN=True),FCNet(Dropout=True)]  #默认一个原子类型
# nets = [FCNet(itype=1),FCNet(BN=True, itype=1),FCNet(Dropout=True, itype=1)] 
# for i,net in enumerate(nets):
#     print('the %i th network:'%i)
#     print(net)

class MLFFNet(nn.Module):
    def __init__(self, atomType = pm.atomType, natoms = pm.natoms):  #atomType=[8,32]
        super(MLFFNet,self).__init__()
        self.atomType = atomType
        self.natoms = pm.natoms   #[32,32]
        self.models = nn.ModuleList()
        for i in range(len(self.atomType)):  #i=[0,1]
            self.models.append(FCNet(itype = i))


    def forward(self, image, dfeat, neighbor):
        natoms_index = [0]
        temp = 0
        for i in self.natoms:
            temp += i
            natoms_index.append(temp)    #[0,32,64]
        input_data = image
        for i in range(len(natoms_index)-1):
            x = input_data[:, natoms_index[i]:natoms_index[i+1]]
            _, predict = self.models[i](x)
            if(i==0):
                Ei=predict #[32, 1]
            else:
                Ei=torch.cat((Ei, predict), dim=1)    #[64,1]
        out_sum = Ei.sum()
        Etot = Ei.sum(dim=1)
        out_sum.backward(retain_graph=True)
        input_grad_allatoms = input_data.grad

        batch_size = image.shape[0]
        Force = torch.zeros((batch_size, natoms_index[-1], 3)).to(device)
        for batch_index in range(batch_size):
            atom_index_temp = 0
            for idx, natom in enumerate(self.natoms):  #[32,32]    
                for i in range(natom):
                    neighbori = neighbor[batch_index, atom_index_temp + i]  # neighbor [40, 64, 100] neighbori [1, 100]
                    neighbor_number = neighbori.shape[-1]
                    atom_force = torch.zeros((1, 3)).to(device)
                    for nei in range(neighbor_number):
                        nei_index = neighbori[nei] - 1 #第几个neighbor
                        if(nei_index == 0):
                            continue 
                        atom_force += torch.matmul(input_grad_allatoms[batch_index, nei_index, :], dfeat[batch_index, atom_index_temp + i, nei, :, :])
                    Force[batch_index, atom_index_temp+i] = atom_force
        return Force, Etot, Ei