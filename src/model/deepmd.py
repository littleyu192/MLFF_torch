from re import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import sys, os
import datetime
import math
sys.path.append(os.getcwd())
import parameters as pm    
# import prepare as pp
# pp.readFeatnum()
from model.embedding import EmbedingNet, FittingNet
# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.DeepMD')

def dump(msg, *args, **kwargs):
    logger.log(logging_level_DUMP, msg, *args, **kwargs)
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)
def summary(msg, *args, **kwargs):
    logger.log(logging_level_SUMMARY, msg, *args, **kwargs)
def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)
def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs, exc_info=True)

# MLFF contains two parts:
# MLFF_dmirror implementation and MLFF_autograd implementation
#
def d_sigmoid(x):
    return torch.sigmoid(x) * (1.0 - torch.sigmoid(x))

class preDeepMD(nn.Module):
    def __init__(self, net_cfg, activation_type, device, magic=False):
        super(preDeepMD, self).__init__()
        # config parameters
        self.atomType = pm.atomType
        self.natoms = pm.natoms[0]
        self.device = device
        self.dim_feat = pm.nFeatures
        # network
        if (net_cfg == 'default'):
            self.net_cfg = pm.MLFF_autograd_cfg
            info("pretraining DeepMD: using default net_cfg: pm.DeepMD_cfg")
            info(self.net_cfg)
        else:
            net_cfg = 'pm.' + net_cfg
            self.net_cfg = eval(net_cfg)
            info("pretraining DeepMD: using specified net_cfg: %s" %net_cfg)
            info(self.net_cfg)
        
        self.embeding_net = EmbedingNet(self.net_cfg['embeding_net'], magic)
        fitting_net_input_dim = self.net_cfg['embeding_net']['network_size'][-1]
        self.fitting_net = FittingNet(self.net_cfg['fitting_net'], 16 * fitting_net_input_dim, magic)
    
    def smooth(self, x, mask):
        # return x
        res = torch.zeros_like(x)
        mask_min = x < 10   #set rcut=25, 10  min=0,max=30
        mask_1 = mask & mask_min  #[2,108,100]
        res[mask_1] = 1/x[mask_1]
        # import ipdb; ipdb.set_trace()
        mask_max = x < 25
        mask_2 = ~mask_min & mask_max
        res[mask_2] = 0.5 * torch.cos(math.pi * (x[mask_2]-10)/(25-10)) + 0.5 * torch.ones_like(x[mask_2])
        # import ipdb; ipdb.set_trace()
        return res
    
        
    def forward(self, image_dR):
        # starttime = datetime.datetime.now()
        torch.autograd.set_detect_anomaly(True)
        batch_size = image_dR.shape[0]
        natoms = image_dR.shape[1]
        neighbor_num = image_dR.shape[2]
        list_neigh = image_dR[:,:,:,3]  #[2,108,100]
        image_dR_xyz = image_dR[:,:,:,:3]
        dR2 = torch.sum(image_dR_xyz * image_dR_xyz, -1) #[2,108,100]
        # Rij = torch.pow(dR2, 0.5)
        Rij = torch.sqrt(dR2)
        Rij = dR2
        mask = list_neigh > 0
        S_Rij = self.smooth(Rij.squeeze(-1), mask)
        Ri = torch.zeros((batch_size, natoms, neighbor_num, 4), device=self.device)
        Ri[:, :, :, 0] = S_Rij
        for batch_idx in range(batch_size):
            for i in range(natoms):
                # neighbor_tmp = Rij[batch_idx, i].nonzero().squeeze().type(torch.int64)
                for neighbor_tmp in range(neighbor_num):
                    if list_neigh[batch_idx, i, neighbor_tmp] > 0:
                        Ri[batch_idx, i, neighbor_tmp, 1] = S_Rij[batch_idx, i, neighbor_tmp] * image_dR[batch_idx, i, neighbor_tmp, 0] / Rij[batch_idx, i, neighbor_tmp]
                        Ri[batch_idx, i, neighbor_tmp, 2] = S_Rij[batch_idx, i, neighbor_tmp] * image_dR[batch_idx, i, neighbor_tmp, 1] / Rij[batch_idx, i, neighbor_tmp]
                        Ri[batch_idx, i, neighbor_tmp, 3] = S_Rij[batch_idx, i, neighbor_tmp] * image_dR[batch_idx, i, neighbor_tmp, 2] / Rij[batch_idx, i, neighbor_tmp]
                    else:
                        break
        
        S_Rij = S_Rij.unsqueeze(-1)
        G = self.embeding_net(S_Rij)
        G_t = G[:, :, :, :16].transpose(-2, -1)
        tmpA = torch.matmul(G_t, Ri)
        tmpB = torch.matmul(Ri.transpose(-2, -1), G)
        DR = torch.matmul(tmpA, tmpB).reshape(batch_size, natoms, -1)
        Ei = self.fitting_net(DR)
        Etot = torch.sum(Ei, 1)
        return Etot, Ei


class DeepMD(nn.Module):
    def __init__(self, net_cfg, activation_type, device, magic=False):
        super(DeepMD, self).__init__()
        # config parameters
        self.atomType = pm.atomType
        self.natoms = pm.natoms[0]
        self.device = device
        self.dim_feat = pm.nFeatures
        # network
        if (net_cfg == 'default'):
            self.net_cfg = pm.MLFF_autograd_cfg
            info("DeepMD method: using default net_cfg: pm.DeepMD_cfg")
            info(self.net_cfg)
        else:
            net_cfg = 'pm.' + net_cfg
            self.net_cfg = eval(net_cfg)
            info("DeepMD: using specified net_cfg: %s" %net_cfg)
            info(self.net_cfg)
        
        self.embeding_net = EmbedingNet(self.net_cfg['embeding_net'], magic)
        fitting_net_input_dim = self.net_cfg['embeding_net']['network_size'][-1]
        self.fitting_net = FittingNet(self.net_cfg['fitting_net'], 16 * fitting_net_input_dim, magic)
    
    def smooth(self, x, mask):
        # return x
        res = torch.zeros_like(x)
        mask_min = x < 10   #set rcut=25, 10  min=0,max=30
        mask_1 = mask & mask_min  #[2,108,100]
        res[mask_1] = 1/x[mask_1]
        # import ipdb; ipdb.set_trace()
        mask_max = x < 25
        mask_2 = ~mask_min & mask_max
        res[mask_2] = 0.5 * torch.cos(math.pi * (x[mask_2]-10)/(25-10)) + 0.5 * torch.ones_like(x[mask_2])
        # import ipdb; ipdb.set_trace()
        return res
    
    def sec_to_hms(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m, s)
        
    def forward(self, image_dR, neighbor):
        # starttime = datetime.datetime.now()
        torch.autograd.set_detect_anomaly(True)
        batch_size = image_dR.shape[0]
        natoms = image_dR.shape[1]
        neighbor_num = image_dR.shape[2]
        list_neigh = image_dR[:,:,:,3]  #[2,108,100]
        image_dR_xyz = image_dR[:,:,:,:3]
        dR2 = torch.sum(image_dR_xyz * image_dR_xyz, -1) #[2,108,100]
        # Rij = torch.pow(dR2, 0.5)
        Rij = torch.sqrt(dR2)  
        # import ipdb;ipdb.set_trace()
        # image_000 = torch.zeros([1,3], device=self.device)
        # dR2 = torch.cdist(image_dR_xyz, image_000, 2)
        Rij = dR2
        mask = list_neigh > 0
        S_Rij = self.smooth(Rij.squeeze(-1), mask)
        # import ipdb;ipdb.set_trace()
        Ri = torch.zeros((batch_size, natoms, neighbor_num, 4), device=self.device)

        Ri[:, :, :, 0] = S_Rij
        
        for batch_idx in range(batch_size):
            for i in range(natoms):
                # neighbor_tmp = Rij[batch_idx, i].nonzero().squeeze().type(torch.int64)
                for neighbor_tmp in range(neighbor_num):
                    if list_neigh[batch_idx, i, neighbor_tmp] > 0:
                        Ri[batch_idx, i, neighbor_tmp, 1] = S_Rij[batch_idx, i, neighbor_tmp] * image_dR[batch_idx, i, neighbor_tmp, 0] / Rij[batch_idx, i, neighbor_tmp]
                        Ri[batch_idx, i, neighbor_tmp, 2] = S_Rij[batch_idx, i, neighbor_tmp] * image_dR[batch_idx, i, neighbor_tmp, 1] / Rij[batch_idx, i, neighbor_tmp]
                        Ri[batch_idx, i, neighbor_tmp, 3] = S_Rij[batch_idx, i, neighbor_tmp] * image_dR[batch_idx, i, neighbor_tmp, 2] / Rij[batch_idx, i, neighbor_tmp]
                    else:
                        break
        
        S_Rij = S_Rij.unsqueeze(-1)
        # Rtuta_time = datetime.datetime.now()
        # print("R~ prepare time:")
        # print((Rtuta_time - starttime).seconds)
        G = self.embeding_net(S_Rij)
        # G_time = datetime.datetime.now()
        # print("embedding net time:")
        # print((G_time - Rtuta_time).seconds)
        G_t = G[:, :, :, :16].transpose(-2, -1)
        tmpA = torch.matmul(G_t, Ri)
        tmpB = torch.matmul(Ri.transpose(-2, -1), G)
        DR = torch.matmul(tmpA, tmpB).reshape(batch_size, natoms, -1)
        # DR_time = datetime.datetime.now()
        # print("rotation GRRG time:")
        # print((G_time - Rtuta_time).seconds)
        Ei = self.fitting_net(DR)
        # fitting_time = datetime.datetime.now()
        # print("fitting net time:")
        # print((fitting_time - G_time).seconds)

        mask = torch.ones_like(Ei)
        dE = torch.autograd.grad(Ei, image_dR_xyz, grad_outputs=mask, retain_graph=True, create_graph=True)
        # autograd_time = datetime.datetime.now()
        # print("auto grad time:")
        # print((autograd_time - fitting_time).seconds)
        # import ipdb;ipdb.set_trace()
        dE = torch.stack(list(dE), dim=0).squeeze(0)  #[:,:,:,:-1] #[2,108,100,4]-->[2,108,100,3]
        # import ipdb;ipdb.set_trace()
        dEtot = torch.sum(dE, 2)  #[2,108,3]
        # result_Ei = torch.zeros((batch_size, self.natoms)).to(self.device)
        # result_dEi_dFeat = torch.zeros((batch_size, self.natoms, self.dim_feat)).to(self.device)
        # result_Ei, result_dEi_dFeat = self.net(image)

        Etot = torch.sum(Ei, 1)
        Force = torch.zeros((batch_size, natoms, 3), device=self.device)

        for batch_idx in range(batch_size):
            for i in range(natoms):
                # get atom_idx & neighbor_idx
                my_neighbor = list_neigh[batch_idx, i]  #[100]
                neighbor_idx = my_neighbor.nonzero().squeeze().type(torch.int64)  #[78]
                atom_idx = my_neighbor[neighbor_idx].type(torch.int64) - 1
                # import ipdb;ipdb.set_trace()
                # calculate Force
                for neigh_tmp, neighbor_id in zip(atom_idx, neighbor_idx):
                    # import ipdb;ipdb.set_trace()
                    dEtot[batch_idx, i, :] -= dE[batch_idx, neigh_tmp, neighbor_id, :3]
                Force[batch_idx, i, :] = -dEtot[batch_idx, i, :]
            # import ipdb;ipdb.set_trace()
        # calcF_time = datetime.datetime.now()
        # print("fitting net time:")
        # print((calcF_time - autograd_time).seconds)
        print("Force[0,0,:]:")
        print(Force[0,0,:])
        return Etot, Ei, Force
