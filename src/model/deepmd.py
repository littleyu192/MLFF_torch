from sys import path
from builtins import print
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
        rmin = 5.8     # set rcut=10, min=0,max=30
        rmax = 6.0     # set rcut=25,
        res = torch.zeros_like(x)
        mask_min = x < rmin
        mask_1 = mask & mask_min  #[2,108,100]
        res[mask_1] = 1/x[mask_1]
        # import ipdb; ipdb.set_trace()
        mask_max = x < rmax
        mask_2 = ~mask_min & mask_max
        res[mask_2] = 0.5 * torch.cos(math.pi * (x[mask_2]-rmin)/(rmax-rmin)) + 0.5 * torch.ones_like(x[mask_2])
        # import ipdb; ipdb.set_trace()
        return res
    
        
    def forward(self, image_dR, list_neigh):
        # starttime = datetime.datetime.now()
        torch.autograd.set_detect_anomaly(True)
        batch_size = image_dR.shape[0]
        natoms = image_dR.shape[1]
        neighbor_num = image_dR.shape[2]
        # list_neigh = image_dR[:,:,:,3]  #[2,108,100]
        # image_dR_xyz = image_dR[:,:,:,:3]
        # dR2 = torch.sum(image_dR_xyz * image_dR_xyz, -1) #[2,108,100]
        dR2 = torch.sum(image_dR * image_dR, -1) #[2,108,100]
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
        # Etot.shape:[2,1], Ei.shape:[2,108,1]
        print("Etot[0,0] and Ei[0,1]")
        print(Etot[0,0])
        print(Ei[0,0])
        return Etot, Ei


class DeepMD(nn.Module):
    def __init__(self, net_cfg, activation_type, device, stat, magic=False):
        super(DeepMD, self).__init__()
        # config parameters
        self.atomType = pm.atomType
        self.natoms = pm.natoms[0]
        self.device = device
        self.dim_feat = pm.nFeatures
        # [0: avg, 1: std, 2: ener_shift]
        self.stat = stat

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
        self.fitting_net = FittingNet(self.net_cfg['fitting_net'], 16 * fitting_net_input_dim, self.stat[2], magic)
    
    def smooth(self, image_dR, x, Ri_xyz, mask, inr, davg, dstd):

        inr2 = torch.zeros_like(inr)
        inr3 = torch.zeros_like(inr)
        inr4 = torch.zeros_like(inr)
        
        inr2[mask] = inr[mask] * inr[mask]
        inr4[mask] = inr2[mask] * inr2[mask]
        inr3[mask] = inr4[mask] * x[mask]
        
        uu = torch.zeros_like(x)
        vv = torch.zeros_like(x)
        dvv = torch.zeros_like(x)
        
        res = torch.zeros_like(x)

        # x < rcut_min vv = 1
        mask_min = x < 5.8   #set rcut=25, 10  min=0,max=30
        mask_1 = mask & mask_min  #[2,108,100]
        vv[mask_1] = 1
        dvv[mask_1] = 0

        # rcut_min< x < rcut_max
        mask_max = x < 6.0
        mask_2 = ~mask_min & mask_max & mask
        # uu = (xx - rmin) / (rmax - rmin) ;
        uu[mask_2] = (x[mask_2] - 5.8)/(6.0 -5.8)
        vv[mask_2] = uu[mask_2] * uu[mask_2] * uu[mask_2] * (-6 * uu[mask_2] * uu[mask_2] + 15 * uu[mask_2] - 10) + 1
        du = 1.0 / ( 6.0 - 5.8)
        # dd = ( 3 * uu*uu * (-6 * uu*uu + 15 * uu - 10) + uu*uu*uu * (-12 * uu + 15) ) * du;
        dvv[mask_2] = (3 * uu[mask_2] * uu[mask_2] * (-6 * uu[mask_2] * uu[mask_2] + 15 * uu[mask_2] -10) + uu[mask_2] * uu[mask_2] * uu[mask_2] * (-12 * uu[mask_2] + 15)) * du
 
        mask_3 = ~mask_max & mask
        vv[mask_3] = 0
        dvv[mask_3] = 0

        res[mask] = 1.0 / x[mask]
        Ri = torch.cat((res.unsqueeze(-1), Ri_xyz), dim=-1)
        Ri_d = torch.zeros_like(Ri).unsqueeze(-1).repeat(1, 1, 1, 1, 3) # 2 108 100 4 3
        tmp = torch.zeros_like(x)

        # deriv of component 1/r
        tmp[mask] = image_dR[:, :, :, 0][mask] * inr3[mask] * vv[mask] - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[mask]
        Ri_d[:, :, :, 0, 0][mask] = tmp[mask]
        tmp[mask] = image_dR[:, :, :, 1][mask] * inr3[mask] * vv[mask] - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[mask]
        Ri_d[:, :, :, 0, 1][mask] = tmp[mask]
        tmp[mask] = image_dR[:, :, :, 2][mask] * inr3[mask] * vv[mask] - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[mask]
        Ri_d[:, :, :, 0, 2][mask] = tmp[mask]

        # deriv of component x/r
        tmp[mask] = (2 * image_dR[:, :, :, 0][mask] * image_dR[:, :, :, 0][mask] * inr4[mask] - inr2[mask]) * vv[mask] - Ri[:, :, :, 1][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[mask]
        Ri_d[:, :, :, 1, 0][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 0][mask] * image_dR[:, :, :, 1][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 1][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[mask]
        Ri_d[:, :, :, 1, 1][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 0][mask] * image_dR[:, :, :, 2][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 1][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[mask]
        Ri_d[:, :, :, 1, 2][mask] = tmp[mask]
       
        # deriv of component y/r
        tmp[mask] = (2 * image_dR[:, :, :, 1][mask] * image_dR[:, :, :, 0][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 2][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[mask]
        Ri_d[:, :, :, 2, 0][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 1][mask] * image_dR[:, :, :, 1][mask] * inr4[mask] - inr2[mask]) * vv[mask] - Ri[:, :, :, 2][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[mask]
        Ri_d[:, :, :, 2, 1][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 1][mask] * image_dR[:, :, :, 2][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 2][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[mask]
        Ri_d[:, :, :, 2, 2][mask] = tmp[mask]
    
        # deriv of component z/r
        tmp[mask] = (2 * image_dR[:, :, :, 2][mask] * image_dR[:, :, :, 0][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 3][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr[mask]
        Ri_d[:, :, :, 3, 0][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 2][mask] * image_dR[:, :, :, 1][mask] * inr4[mask]) * vv[mask] - Ri[:, :, :, 3][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr[mask]
        Ri_d[:, :, :, 3, 1][mask] = tmp[mask]
        tmp[mask] = (2 * image_dR[:, :, :, 2][mask] * image_dR[:, :, :, 2][mask] * inr4[mask] - inr2[mask]) * vv[mask] - Ri[:, :, :, 3][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr[mask]
        Ri_d[:, :, :, 3, 2][mask] = tmp[mask]

        vv_copy = vv.unsqueeze(-1).repeat(1, 1, 1, 4)
        Ri[mask] *= vv_copy[mask]

        Ri = (Ri - davg) / dstd
        dstd = dstd.unsqueeze(-1).repeat(1, 1, 3)
        Ri_d = Ri_d / dstd 
        
        # res[mask_2] = 0.5 * torch.cos(math.pi * (x[mask_2]-10)/(25-10)) + 0.5 * torch.ones_like(x[mask_2])
        return Ri, Ri_d
    
    def sec_to_hms(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m, s)
        
    def forward(self, image_dR, list_neigh):
        # starttime = datetime.datetime.now()
        # image_dR = torch.tensor(np.load("rij.npy"), device=self.device, requires_grad=True)
        # list_neigh = torch.tensor(np.load("nblist.npy"), device=self.device)
        # image_dR = image_dR.reshape(1, 108, 100, 3)
        # list_neigh = list_neigh.reshape(1, 108, 100)
        # list_neigh = list_neigh + 1

        torch.autograd.set_detect_anomaly(True)
        batch_size = image_dR.shape[0]
        natoms = image_dR.shape[1]
        neighbor_num = image_dR.shape[2]
        # np.save("torch_image_dR.npy", image_dR.cpu().detach().numpy())
        
        mask = list_neigh > 0
        dR2 = torch.zeros_like(list_neigh, dtype=torch.double)
        Rij = torch.zeros_like(list_neigh, dtype=torch.double)
        dR2[mask] = torch.sum(image_dR[mask] * image_dR[mask], -1) #[2,108,100]
        Rij[mask] = torch.sqrt(dR2[mask])

        nr = torch.zeros_like(dR2)
        inr = torch.zeros_like(dR2)
        Ri_xyz = torch.zeros((batch_size, natoms, neighbor_num, 3), device=self.device)
        dR2_copy = dR2.unsqueeze(-1).repeat(1, 1, 1, 3)

        nr[mask] = dR2[mask] / Rij[mask]       # ?
        Ri_xyz[mask] = image_dR[mask] / dR2_copy[mask]
        inr[mask] = 1 / Rij[mask]

        davg = torch.tensor(self.stat[0], device=self.device)
        dstd = torch.tensor(self.stat[1], device=self.device)
        # davg = torch.tensor(np.load('davg.npy'), device=self.device)
        # dstd = torch.tensor(np.load('dstd.npy'), device=self.device)
        Ri, Ri_d = self.smooth(image_dR, nr, Ri_xyz, mask, inr, davg, dstd)
        # np.save("torch_Ri.npy", Ri.cpu().detach().numpy())
        # import ipdb;ipdb.set_trace()
        # np.save("torch_Ri_d.npy", Ri_d.cpu().detach().numpy())
        S_Rij = Ri[:, :, :, 0].unsqueeze(-1)

        G = self.embeding_net(S_Rij)
        tmpA = torch.matmul(Ri.transpose(-2, -1), G)
        tmpA *= 0.01 # batch 108 4 100
        tmpB = tmpA[:, :, :, :16]

        DR = torch.matmul(tmpA.transpose(-2, -1), tmpB)
        DR = DR.reshape(batch_size, natoms, -1)
        # np.save("torch_DR.npy", DR.cpu().detach().numpy())
        # import ipdb;ipdb.set_trace()

        Ei = self.fitting_net(DR)
        # np.save("torch_Ei.npy", Ei.cpu().detach().numpy())
        # import ipdb;ipdb.set_trace()
        Etot = torch.sum(Ei, 1)

        mask = torch.ones_like(Ei)
        dE = torch.autograd.grad(Ei, Ri, grad_outputs=mask, retain_graph=True, create_graph=True)
        dE = torch.stack(list(dE), dim=0).squeeze(0)  #[:,:,:,:-1] #[2,108,100,4]-->[2,108,100,3]

        Ri_d = Ri_d.reshape(batch_size, natoms, -1, 3)
        dE = dE.reshape(batch_size, natoms, 1, -1)

        F = torch.matmul(dE, Ri_d).squeeze(-2) # batch natom 3
        F = F * (-1)

        for batch_idx in range(batch_size):
            for i in range(natoms):
                # get atom_idx & neighbor_idx
                i_neighbor = list_neigh[batch_idx, i]  #[100]
                neighbor_idx = i_neighbor.nonzero().squeeze().type(torch.int64)  #[78]
                atom_idx = i_neighbor[neighbor_idx].type(torch.int64) - 1
                # calculate Force
                for neigh_tmp, neighbor_id in zip(atom_idx, neighbor_idx):
                    tmpA = dE[batch_idx, i, :, neighbor_id*4:neighbor_id*4+4]
                    tmpB = Ri_d[batch_idx, i, neighbor_id*4:neighbor_id*4+4]
                    F[batch_idx, neigh_tmp] += torch.matmul(tmpA, tmpB).squeeze(0)    
        # np.save("torch_force.npy", F.cpu().detach().numpy())
        # import ipdb;ipdb.set_trace()

        print("Ei[0, 5] & Etot[0] & Force[0, 5, :]:")
        print(Ei[0, 5].item())
        print(Etot[0].item())
        print(F[0, 5].tolist())
        # import ipdb;ipdb.set_trace()
        return Etot, Ei, F               

        # # autograd_time = datetime.datetime.now()
        # # print("auto grad time:")
        # # print((autograd_time - fitting_time).seconds)
        # # import ipdb;ipdb.set_trace()
        # # import ipdb;ipdb.set_trace()
        # dEtot = torch.sum(dE, 2)  #[2,108,3]
        # # result_Ei = torch.zeros((batch_size, self.natoms)).to(self.device)
        # # result_dEi_dFeat = torch.zeros((batch_size, self.natoms, self.dim_feat)).to(self.device)
        # # result_Ei, result_dEi_dFeat = self.net(image)

        # Etot = torch.sum(Ei, 1)
        # Force = torch.zeros((batch_size, natoms, 3), device=self.device)

        # for batch_idx in range(batch_size):
        #     for i in range(natoms):
        #         # get atom_idx & neighbor_idx
        #         i_neighbor = list_neigh[batch_idx, i]  #[100]
        #         neighbor_idx = i_neighbor.nonzero().squeeze().type(torch.int64)  #[78]
        #         atom_idx = i_neighbor[neighbor_idx].type(torch.int64) - 1
        #         # calculate Force
        #         # for neigh_tmp, neighbor_id in zip(atom_idx, neighbor_idx):
        #         for neigh_tmp in atom_idx:
        #             j_neighbor = list_neigh[batch_idx, neigh_tmp]
        #             neighbor_id = (j_neighbor == i+1).nonzero().squeeze()
        #             # print(neighbor_id)
        #             # dEtot[batch_idx, i, :] -= dE[batch_idx, neigh_tmp, neighbor_id, :3]
        #             if neighbor_id.dim() > 0:
        #                 tmp = dE[batch_idx, neigh_tmp, neighbor_id, :3].sum(-2)
        #             else:
        #                 tmp = dE[batch_idx, neigh_tmp, neighbor_id, :3]
        #             dEtot[batch_idx, i, :] -= tmp
                    
        #             # for i in range([int(x) for x in neighbor_id.shape][0]):
        #             #     neighbor_idx = neighbor_idx.tolist()
        #             #     dEtot[batch_idx, i, :] -= dE[batch_idx, neigh_tmp, neighbor_id[i], :3]

        #         # import ipdb;ipdb.set_trace()
        #         Force[batch_idx, i, :] = -dEtot[batch_idx, i, :]
            
        # # calcF_time = datetime.datetime.now()
        # # print("fitting net time:")
        # # print((calcF_time - autograd_time).seconds)

        
