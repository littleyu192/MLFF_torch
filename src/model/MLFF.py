#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import init
from torch.autograd import Variable
import sys, os
sys.path.append(os.getcwd())
import parameters as pm    
import time

################################################################
# fully connection nn
# Ei Neural Network
################################################################



# try1:
def dtanh(x):
    return 1.0-torch.tanh(x)**2

ACTIVE = torch.tanh
dACTIVE = dtanh



# ACTIVE = F.softplus 
# ACTIVE = torch.relu  

# try2:
# def dsigmoid(x):
#     return torch.sigmoid(x) * (1 - torch.sigmoid(x))
# ACTIVE = torch.sigmoid
# dACTIVE = dsigmoid

# try3:
# ACTIVE = F.softplus
# dACTIVE = torch.sigmoid


# try4:
# ACTIVE = torch.relu
# def drelu(x):
#     res = torch.zeros_like(x)
#     mask = x > 0
#     res[mask] = 1
#     return res
# dACTIVE = drelu

# def no_act(x):
#     return x
# def no_dact(x):
#     return torch.ones_like(x)
# ACTIVE = no_act
# dACTIVE = no_dact

# try5:
# ACTIVE = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
# def dLeakyReLU(x):
#     res = torch.ones_like(x)
#     mask = x < 0
#     res[mask] = 0.01
#     return res
# dACTIVE = dLeakyReLU


class FCNet(nn.Module):
    def __init__(self, BN = False, Dropout = False, itype = 0):  #atomtypes=len(pm.atomType)
        super(FCNet,self).__init__()
        self.dobn = BN
        self.dodrop = Dropout                                  
        self.fcs=[]
        self.bns=[]
        self.drops=[]
        self.itype= itype  # itype =0,1 if CuO         
        self.weights = nn.ParameterList()
        self.bias = nn.ParameterList()

        for i in range(pm.nLayers-1):
            in_putsize = pm.nFeatures if i==0 else pm.nNodes[i-1,itype] #0为type1,1为type2
            w = nn.Parameter(nn.init.xavier_uniform_(torch.randn(pm.nNodes[i, itype], in_putsize)))
            b = torch.empty(pm.nNodes[i,itype])
            b = nn.Parameter(torch.full_like(b,0.0))
            self.weights.append(w)
            self.bias.append(b)
        w = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1,pm.nNodes[pm.nLayers-2,itype])))
        b = nn.Parameter(torch.tensor([pm.itype_Ei_mean[itype]]), requires_grad=True)
        #w = nn.Parameter(nn.init.xavier_normal_(torch.randn(1, pm.nNodes[pm.nLayers - 2, itype])))
        #if itype==0:
        #    b = nn.Parameter(torch.tensor([pm.itype_Ei_mean[itype]]), requires_grad=True)
        #if itype==1:
        #    b = nn.Parameter(torch.tensor([437.4]), requires_grad=True)
        #b = nn.Parameter(torch.empty(1))
        self.weights.append(w)
        self.bias.append(b)

    def forward(self, x):
        L = []
        dL = []
        # print("L[0]=F.linear(x, self.weights[0]")
        # print(x)
        # print('~'*10)
        # print(self.weights[0])
        # print('~'*10)
        # dL.append(dACTIVE(F.linear(x, self.weights[0], bias=self.bias[0])))
        # L.append(ACTIVE(F.linear(x, self.weights[0], bias=self.bias[0])))
        # Fout_0=F.linear(x, self.weights[0], bias=self.bias[0])
        Fout_0 = torch.matmul(x, self.weights[0].t()) + self.bias[0]
        L.append(ACTIVE(Fout_0))
        dL.append(dACTIVE(Fout_0))
        for ilayer in range(1, pm.nLayers-1):
            # Fout_temp = F.linear(L[ilayer-1], self.weights[ilayer], bias=self.bias[ilayer])
            # L.append(ACTIVE(Fout_temp))
            # dL.append(dACTIVE(Fout_temp))  
            Fout_temp = torch.matmul(L[ilayer-1], self.weights[ilayer].t()) + self.bias[ilayer]
            L.append(ACTIVE(Fout_temp))
            dL.append(dACTIVE(Fout_temp)) 
        # print('L[1]='+str(L[1]))
        # print('dL[1]='+str(dL[1]))
        # predict = F.linear(L[pm.nLayers-2], self.weights[-1], bias=self.bias[-1])  #网络的最后一层
        predict = torch.matmul(L[pm.nLayers-2], self.weights[-1].t()) + self.bias[-1]
        # predict = torch.matmul(Fout_temp,self.weights[-1].t())+self.bias[-1]
        '''warning!!! if loss this dL, will make bias.2 to be None grad'''
        dL.append(1.0*predict)
        ilayer += 1
        grad = dL[-1]*self.weights[ilayer]
        ilayer -= 1
        while ilayer >= 0:
            grad = dL[ilayer] * grad   #(2,108,30)*(1,30)-->(2,108,30)
            grad = grad.unsqueeze(2) * self.weights[ilayer].t()  #(2,108,1,30)*(60,30)-->(2,108,60,30)
            grad = grad.sum(axis=-1)  #(2,108,60,30)-->(2,108,60)
            ilayer -= 1
        return predict, grad


class MLFFNet(nn.Module):
    def __init__(self, device, training_type = torch.float64, atomType = pm.atomType, Dropout = False):  #atomType=[8,32]
        super(MLFFNet,self).__init__()
        self.atomType = atomType
        self.models = nn.ModuleList()
        self.device = device
        self.training_type = training_type
        for i in range(len(self.atomType)):  #i=[0,1]
            self.models.append(FCNet(itype = i, Dropout=Dropout).to(self.training_type))   # Dropout=True


    def forward(self, image, dfeat, neighbor, natoms_img, Egroup_weight, divider, is_calc_f=None):
        start = time.time()
        #image.requires_grad_(True)
        batch_size = image.shape[0]
        natom = image.shape[1]
        neighbor_num=dfeat.shape[2]

        natoms_index = [0]
        temp = 0
        for i in natoms_img[0, 1:]:
            temp += i
            natoms_index.append(temp)    #[0,32,64]
        
        # for i in range(len(natoms_index)-1):
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            x = image[:, natoms_index[i]:natoms_index[i+1]]
            predict, grad = self.models[i](x)
            # scale_feat_a = torch.tensor(self.scalers.feat_as[itype], device=device, dtype=torch.float)
            # grad = grad * scale_feat_a
            if(i==0):
                Ei = predict #[32, 1]
                dE = grad
            else:
                Ei = torch.cat((Ei, predict), dim=1)    #[64,1]
                dE = torch.cat((dE, grad), dim=1)
        # de = self.get_de(image, dfeat, neighbor)
        input_grad_allatoms = dE
        cal_ei_de = time.time()
        Etot = Ei.sum(dim=1)
        F = torch.zeros((batch_size, natom, 3), device=self.device)
        if is_calc_f == False:
            return Etot, Ei, F
        #dE = torch.autograd.grad(res0, in_feature, grad_outputs=mask, create_graph=True, retain_graph=True)

        test = Ei.sum()
        #test.backward(retain_graph=True)
        #test_grad=image.grad
        mask = torch.ones_like(test)
        test_grad = torch.autograd.grad(test,image,grad_outputs=mask, create_graph=True, retain_graph=True)
        test_grad = test_grad[0]   
        dim_feat=pm.nFeatures
        result_dEi_dFeat_fortran = torch.zeros((batch_size, natom + 1, dim_feat)).to(self.training_type)
        result_dEi_dFeat_fortran[:, 1:, :]=test_grad
        n_a_idx_fortran = neighbor.reshape(batch_size * natom * neighbor_num)
        n_a_ofs_fortran_b = torch.arange(0, batch_size * (natom+1), (natom+1))\
                            .repeat_interleave(natom * neighbor_num)
        n_a_idx_fortran_b = n_a_idx_fortran.type(torch.int64) + n_a_ofs_fortran_b.to(self.device)
        dEi_neighbors = result_dEi_dFeat_fortran\
                        .reshape(batch_size * (natom+1), dim_feat)[n_a_idx_fortran_b,]\
                        .reshape(batch_size, natom, neighbor_num, 1, dim_feat)
        Force = torch.matmul(dEi_neighbors.to(self.device), dfeat).sum([2, 3])
        self.Ei = Ei
        self.Etot = Etot
        self.Force = Force
        return Etot, Ei, Force

    def get_egroup(self, Egroup_weight, divider):
        batch_size = self.Ei.shape[0]
        Egroup = torch.zeros_like(self.Ei)
        for i in range(batch_size):
            Etot1 = self.Ei[i]
            weight_inner = Egroup_weight[i]
            E_inner = torch.matmul(weight_inner, Etot1)
            Egroup[i] = E_inner
        Egroup_out = torch.divide(Egroup, divider)
        return Egroup_out

    def get_de(self, image, dfeat, neighbor):
        batch_size = image.shape[0]
        atom_type_num = len(self.atomType)
        for i in range(atom_type_num):
            model_weight = self.models[i].state_dict()
            W = []
            B = []
            L = []
            dL = []
            W.append(model_weight['fc0.weight'].transpose(0, 1))            
            B.append(model_weight['fc0.bias'])            
            dL.append(dACTIVE(torch.matmul(image, W[0]) + B[0]))
            L.append(ACTIVE(torch.matmul(image, W[0]) + B[0]))
            for ilayer in range(1, pm.nLayers-1):
                W.append(model_weight['fc' + str(ilayer) + '.weight'].transpose(0, 1))            
                B.append(model_weight['fc' + str(ilayer) + '.bias'])            
                dL.append(dACTIVE(torch.matmul(L[ilayer-1], W[ilayer]) + B[ilayer]))
                L.append(ACTIVE(torch.matmul(L[ilayer-1], W[ilayer]) + B[ilayer]))
            ilayer += 1
            W.append(model_weight['output.weight'].transpose(0, 1))            
            B.append(model_weight['output.bias'])         
            res = W[ilayer].transpose(0, 1)
            ilayer -= 1
            while ilayer >= 0:
                res = dL[ilayer] * res   #(2,108,30)*(1,30)-->(2,108,30)
                res = res.unsqueeze(2) * W[ilayer]  #(2,108,1,30)*(60,30)-->(2,108,60,30)
                res = res.sum(axis=-1)  #(2,108,60,30)-->(2,108,60)
                ilayer -= 1
            return res


