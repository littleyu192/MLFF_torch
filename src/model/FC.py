#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import os
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

class FCNet(nn.Module):
    def __init__(self,BN = False,Dropout = False):              
        super(FCNet,self).__init__()
        self.dobn = BN
        self.dodrop = Dropout                                  
        self.fcs=[]
        self.bns=[]
        self.drops=[]                                          
        for i in range(pm.nLayers-1):
            in_putsize = pm.nFeatures if i ==0 else pm.nNodes[i-1,0] #0为type1,1为type2
            fc = nn.Linear(in_putsize,pm.nNodes[i,0])
            setattr(self,'fc%i'%i,fc)   #setattr函数用来设置属性，其中第一个参数为继承的类别，第二个为名称，第三个是数值
            self.fcs.append(fc)
            self.__set__init(fc)  #初始化网络训练参数
            if self.dobn:
                bn = nn.BatchNorm1d(in_putsize,momentum=0.5)
                setattr(self,'bn%i'%i,bn)
                self.bns.append(bn)
            if self.dodrop:
                drop = nn.Dropout(0.5)
                setattr(self,'drop%i'%i,drop)
                self.drops.append(drop)
        self.output = nn.Linear(pm.nNodes[pm.nLayers-2,0],1)  #最后一层
        self.__set__init(self.output)

    def __set__init(self,layer):
        init.normal_(layer.weight, mean = 0, std = 1)       
        init.constant_(layer.weight,val=B_INIT)

    def forward(self, x):
        input = x
        for i in range(pm.nLayers-1):                         
            x = self.fcs[i](x)
            if self.dobn:       
                x = self.bns(x)
            if self.dodrop:              
                self.drops[i]
        x = ACTIVE(x)         #激活函数，可以自定义
        predict = self.output(x)  #网络的最后一层
        return input, predict

# nets = [FCNet(),FCNet(BN=True),FCNet(Dropout=True)]
# for i,net in enumerate(nets):
#     print('the %i th network:'%i)
#     print(net)