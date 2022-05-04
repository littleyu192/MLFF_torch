from sys import path
from builtins import print
from re import S
import numpy as np
import torch
from torch import embedding
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import sys, os
import datetime
import time
import math
sys.path.append(os.getcwd())
import parameters as pm    
# import prepare as pp
# pp.readFeatnum()
from model.embedding import EmbedingNet, FittingNet
from model.calculate_force import CalculateForce
# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.DPFF')

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

class DP(nn.Module):
    def __init__(self, net_cfg, activation_type, device, stat, magic=False):
        super(DP, self).__init__()
        # config parameters
        self.ntypes = pm.ntypes
        self.device = device
        self.stat = stat
        if pm.training_dtype == "float64":
            self.dtype = torch.double
        elif pm.training_dtype == "float32":
            self.dtype = torch.float32
        else:
            raise RuntimeError("train(): unsupported training_dtype %s" % pm.training_dtype)

        # network
        if (net_cfg == 'default'):
            self.net_cfg = pm.MLFF_autograd_cfg
            info("DP method: using default net_cfg: pm.DP_cfg")
            info(self.net_cfg)
        else:
            net_cfg = 'pm.' + net_cfg
            self.net_cfg = eval(net_cfg)
            info("DP: using specified net_cfg: %s" %net_cfg)
            info(self.net_cfg)
        self.embeding_net = nn.ModuleList()
        self.fitting_net = nn.ModuleList()
        # self.embeding_net = EmbedingNet(self.net_cfg['embeding_net'], magic)
        for i in range(self.ntypes):
            for j in range(self.ntypes):
                self.embeding_net.append(EmbedingNet(self.net_cfg['embeding_net'], magic))
            fitting_net_input_dim = self.net_cfg['embeding_net']['network_size'][-1]
            self.fitting_net.append(FittingNet(self.net_cfg['fitting_net'], 16 * fitting_net_input_dim, self.stat[2][i], magic))
        

    
    def smooth(self, image_dR, x, Ri_xyz, mask, inr, davg, dstd, natoms):

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

        for ntype in range(self.ntypes):
            atom_num_ntype = natoms[ntype]
            davg_ntype = davg[ntype].reshape(-1, 4).squeeze().repeat(atom_num_ntype, 1, 1) #[32,100,4]
            dstd_ntype = dstd[ntype].reshape(-1, 4).squeeze().repeat(atom_num_ntype, 1, 1) #[32,100,4]
            if ntype == 0:
                davg_res = davg_ntype
                dstd_res = dstd_ntype
            else:
                davg_res = torch.concat((davg_res, davg_ntype), dim=0)
                dstd_res = torch.concat((dstd_res, dstd_ntype), dim=0)  
        Ri = (Ri - davg_res) / dstd_res  #[1,64,200,4]
        dstd_res = dstd_res.unsqueeze(-1).repeat(1, 1, 1, 3)
        Ri_d = Ri_d / dstd_res 
        
        # res[mask_2] = 0.5 * torch.cos(math.pi * (x[mask_2]-10)/(25-10)) + 0.5 * torch.ones_like(x[mask_2])
        return Ri, Ri_d


    def smooth_v1(self, image_dR, x, Ri_xyz, mask, inr, davg, dstd):

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

        # change 1/Rij to 1/Rij2
        Ri = torch.cat((inr2.unsqueeze(-1), Ri_xyz), dim=-1)
        Ri_d = torch.zeros_like(Ri).unsqueeze(-1).repeat(1, 1, 1, 1, 3) # 2 108 100 4 3
        tmp = torch.zeros_like(x)

        # deriv of component 1/r
        Ri_d[:, :, :, 0, 0][mask] = 2 * image_dR[:, :, :, 0][mask] * inr4[mask] * vv[mask] - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 0][mask] * inr2[mask]
        Ri_d[:, :, :, 0, 1][mask] = 2 * image_dR[:, :, :, 1][mask] * inr4[mask] * vv[mask] - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 1][mask] * inr2[mask]
        Ri_d[:, :, :, 0, 2][mask] = 2 * image_dR[:, :, :, 2][mask] * inr4[mask] * vv[mask] - Ri[:, :, :, 0][mask] * dvv[mask] * image_dR[:, :, :, 2][mask] * inr2[mask]
        
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

        for ntype in range(self.ntypes):
            atom_num_ntype = self.natoms[ntype]
            davg_ntype = davg[ntype].reshape(-1, 4).squeeze().repeat(atom_num_ntype, 1, 1) #[32,100,4]
            dstd_ntype = dstd[ntype].reshape(-1, 4).squeeze().repeat(atom_num_ntype, 1, 1) #[32,100,4]
            if ntype == 0:
                davg_res = davg_ntype
                dstd_res = dstd_ntype
            else:
                davg_res = torch.concat((davg_res, davg_ntype), dim=0)
                dstd_res = torch.concat((dstd_res, dstd_ntype), dim=0)
            
        Ri = (Ri - davg_res) / dstd_res  #[1,64,200,4]
        dstd_res = dstd_res.unsqueeze(-1).repeat(1, 1, 1, 3)
        Ri_d = Ri_d / dstd_res 
        
        # res[mask_2] = 0.5 * torch.cos(math.pi * (x[mask_2]-10)/(25-10)) + 0.5 * torch.ones_like(x[mask_2])
        return Ri, Ri_d

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


    def forward(self, Ri, dfeat, list_neigh, natoms_img, Egroup_weight, divider, is_calc_f=None):

        torch.autograd.set_detect_anomaly(True)
        
        Ri_d = dfeat
        natoms = natoms_img[0, 1:]
        natoms_sum = Ri.shape[1]
        # when batch_size > 1. don't support multi movement file
        if pm.batch_size > 1 and torch.unique(natoms_img[:, 0]).shape[0] > 1:
            raise ValueError("batchsize > 1, use one movement")
        atom_sum = 0

        for ntype in range(self.ntypes):
            for ntype_1 in range(self.ntypes):
                S_Rij = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1*pm.maxNeighborNum:(ntype_1+1)*pm.maxNeighborNum, 0].unsqueeze(-1)
                embedding_index = ntype * self.ntypes + ntype_1
                G = self.embeding_net[embedding_index](S_Rij)
                tmp_a = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1*pm.maxNeighborNum:(ntype_1+1)*pm.maxNeighborNum].transpose(-2, -1)
                tmp_b = torch.matmul(tmp_a, G)
                
                if ntype_1 == 0:
                    xyz_scater_a = tmp_b
                else:
                    xyz_scater_a = xyz_scater_a + tmp_b
            xyz_scater_a = xyz_scater_a * 4.0 / (pm.maxNeighborNum * self.ntypes * 4)
            xyz_scater_b = xyz_scater_a[:, :, :, :16]
            DR_ntype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b)
            DR_ntype = DR_ntype.reshape(pm.batch_size, natoms[ntype], -1)
            if ntype == 0:
                DR = DR_ntype
            else:
                DR = torch.concat((DR, DR_ntype), dim=1)
            Ei_ntype = self.fitting_net[ntype](DR_ntype)
            
            if ntype == 0:
                Ei = Ei_ntype
            else:
                Ei = torch.concat((Ei, Ei_ntype), dim=1)
            atom_sum = atom_sum + natoms[ntype]
        self.Ei = Ei
        Etot = torch.sum(self.Ei, 1)
        # Egroup = self.get_egroup(Ei, Egroup_weight, divider)
        F = torch.zeros((pm.batch_size, atom_sum, 3), device=self.device)
        if is_calc_f == False:
            return Etot, Ei, F
        # start_autograd = time.time()
        # print("fitting time:", start_autograd - start_fitting, 's')

        mask = torch.ones_like(Ei)
        dE = torch.autograd.grad(Ei, Ri, grad_outputs=mask, retain_graph=True, create_graph=True)
        dE = torch.stack(list(dE), dim=0).squeeze(0)  #[:,:,:,:-1] #[2,108,100,4]-->[2,108,100,3]

        Ri_d = Ri_d.reshape(pm.batch_size, natoms_sum, -1, 3)
        dE = dE.reshape(pm.batch_size, natoms_sum, 1, -1)

        # start_force = time.time()
        # print("autograd time:", start_force - start_autograd, 's')
        F = torch.matmul(dE, Ri_d).squeeze(-2) # batch natom 3
        F = F * (-1)
        
        list_neigh = (list_neigh - 1).type(torch.int)
        F = CalculateForce.apply(list_neigh, dE, Ri_d, F)
        
        return Etot, Ei, F

