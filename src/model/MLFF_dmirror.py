import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import sys, os
sys.path.append(os.getcwd())
import parameters as pm    
# import prepare as pp
# pp.readFeatnum()
from model.dmirror import dmirror_FC

# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.MLFF_dmirror')

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

# MLFF_dmirror implementation
#
def d_sigmoid(x):
    return torch.sigmoid(x) * (1.0 - torch.sigmoid(x))

class MLFF_dmirror(nn.Module):
    def __init__(self, net_cfg, activation_type, device, magic=False):
        super(MLFF_dmirror, self).__init__()
        # config parameters
        self.atomType = pm.atomType
        self.natoms = pm.natoms[0]
        self.nneighbors = pm.maxNeighborNum
        self.device = device
        # network
        if (net_cfg == 'default'):
            self.net_cfg = pm.MLFF_dmirror_cfg
            info("MLFF_dmirror: using default net_cfg: pm.MLFF_dmirror_cfg")
            info(self.net_cfg)
        else:
            net_cfg = 'pm.' + net_cfg
            self.net_cfg = eval(net_cfg)
            info("MLFF_dmirror: using specified net_cfg: %s" %net_cfg)
            info(self.net_cfg)
        self.dim_feat = pm.nFeatures
        if (activation_type == 'sigmoid'):
            self.activation_type = 'sigmoid'
            info("MLFF_dmirror: using sigmoid activation")
            self.net = dmirror_FC(self.net_cfg, torch.sigmoid, d_sigmoid, magic)
        elif (activation_type == 'softplus'):
            self.activation_type = 'softplus'
            info("MLFF_dmirror: using softplus activation")
            self.net = dmirror_FC(self.net_cfg, F.softplus, F.sigmoid, magic)
        else:
            error("MLFF_dmirror: unsupported activation_type: %s" %activation_type)
            raise RuntimeError("MLFF_dmirror: unsupported activation_type: %s" %activation_type)

    def forward(self, image, dfeat, neighbor, Egroup_weight, divider):
        batch_size = image.shape[0]
        result_Ei = torch.zeros(
            (batch_size, self.natoms)
        ).to(self.device)
        result_dEi_dFeat_fortran = torch.zeros(
            (batch_size, self.natoms + 1, self.dim_feat)
        ).to(self.device)

        result_Ei, result_dEi_dFeat_fortran[:, 1:, :] = self.net(image)

        Etot = torch.sum(result_Ei, 1)
        Force = torch.zeros((batch_size, self.natoms, 3)).to(self.device)

        # here we use the infinite cell (in Rcut) view to calc F_atom_i
        # the formula is: sum dE(all neighbor atoms in Rcut)/dR_(this atom)
        #
        # for all infinite cells, the calculate of dE(atom)/dFeat is same as
        # the central unit cell, since they are just duplicates of unit cell,
        # but each cell should has its' own version of dFeat/dRi_(unit_cell),
        # so we have two tensors
        #
        # dEi_neighbors: collection of all neighbor atoms' dE(atom)/dFeat
        # dfeat: collection of all neighbor atoms' dFeat/dRi_(unit_cell)
        #
        # dEi_neighbors.shape = [batch_size, self.natoms, self.nneighbors, 1, self.dim_feat]
        # dfeat.shape         = [batch_size, self.natoms, self.nneighbors, self.dim_feat, 3]
        #
        # n_a_idx_fortran: neighbor's atom index in fortran style, starts
        # from 1, and 0 means empty neighbor slot. result_dEi_dFeat_fortran
        # matches this in it's atom dimension
        #
        # n_a_ofs_fortran_b: offset of neighbor's atom index in the batched
        # atom list, the offset step value for each image is (natoms + 1)
        # to match our fortran style atom index accommodation
        #
        # n_a_idx_fortran_b: neighbor's atom index in the batched atom list
        #
        n_a_idx_fortran = neighbor.reshape(batch_size * self.natoms * self.nneighbors)
        n_a_ofs_fortran_b = torch.arange(0, batch_size * (self.natoms + 1), self.natoms + 1)\
                            .repeat_interleave(self.natoms * self.nneighbors).to(self.device)
        n_a_idx_fortran_b = n_a_idx_fortran.type(torch.int64) + n_a_ofs_fortran_b
        dEi_neighbors = result_dEi_dFeat_fortran\
                        .reshape(batch_size * (self.natoms + 1), self.dim_feat)[n_a_idx_fortran_b,]\
                        .reshape(batch_size, self.natoms, self.nneighbors, 1, self.dim_feat)
        Force = torch.matmul(dEi_neighbors, dfeat).sum([2, 3])

        return Etot, Force

'''
    def get_egroup(self, Ei, Egroup_weight, divider):
        batch_size = Ei.shape[0]
        Egroup = torch.zeros_like(Ei)
        for i in range(batch_size):
            Etot1 = Ei[i]
            weight_inner = Egroup_weight[i]
            E_inner = torch.matmul(weight_inner, Etot1)
            Egroup[i] = E_inner
        Egroup_out = torch.divide(Egroup, divider)
        return Egroup_out
'''
