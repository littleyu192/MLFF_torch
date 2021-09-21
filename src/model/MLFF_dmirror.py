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


def d_sigmoid(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))

class MLFF_dmirror(nn.Module):
    def __init__(self, net_cfg, activation_type, device, magic=False):
        super(MLFF_dmirror, self).__init__()
        # config parameters
        self.atomType = pm.atomType
        self.natoms = pm.natoms[0]
        self.device = device
        # network
        if (net_cfg == 'default'):
            self.net_cfg = pm.MLFF_dmirror_cfg
            print("MLFF_dmirror: using default net_cfg: pm.MLFF_dmirror_cfg")
            print(self.net_cfg)
        else:
            net_cfg = 'pm.' + net_cfg
            self.net_cfg = eval(net_cfg)
            print("MLFF_dmirror: using specified net_cfg: %s" %net_cfg)
            print(self.net_cfg)
        self.dim_feat = pm.nFeatures
        if (activation_type == 'sigmoid'):
            self.activation_type = 'sigmoid'
            print("MLFF_dmirror: using sigmoid activation")
            self.net = dmirror_FC(self.net_cfg, torch.sigmoid, d_sigmoid, magic)
        elif (activation_type == 'softplus'):
            self.activation_type = 'softplus'
            print("MLFF_dmirror: using softplus activation")
            self.net = dmirror_FC(self.net_cfg, F.softplus, F.sigmoid, magic)
        else:
            raise RuntimeError("MLFF_dmirror: unsupported activation_type: %s" %activation_type)

    def forward(self, image, dfeat, neighbor, Egroup_weight, divider):
        batch_size = image.shape[0]
        result_Ei = torch.zeros(
            (batch_size, self.natoms)
        ).to(self.device)
        result_dEi_dFeat = torch.zeros(
            (batch_size, self.natoms, self.dim_feat)
        ).to(self.device)

        # FIXME: loops should be eliminated by matmul style network impl
        for batch_idx in range(batch_size):
            for i in range(self.natoms):
                Ei, dEi_dFeat = self.net(image[batch_idx, i, :])
                result_Ei[batch_idx, i] = Ei
                result_dEi_dFeat[batch_idx, i, :] = dEi_dFeat

        Etot = torch.sum(result_Ei, 1)
        Force = torch.zeros((batch_size, self.natoms, 3)).to(self.device)

        # here we use the infinite cell (in Rcut) view to calc F_atom_i
        # the formula is: sum dE(all neighbor atoms in Rcut)/dR_(this atom)
        #
        # for all infinite cells, the calculate of dE(atom)/dFeat is same as
        # the central unit cell, since they are just duplicates of unit cell,
        # but each cell should has its' own version of dFeat/dRi_(unit_cell),
        # so we will have
        #
        # atom_idx     :  to index dE(atom)/dFeat
        # neighbor_idx :  to index dFeat/dRi_(unit_cell)
        #
        for batch_idx in range(batch_size):
            for i in range(self.natoms):
                # get atom_idx & neighbor_idx
                my_neighbor = neighbor[batch_idx, i]
                neighbor_idx = my_neighbor.nonzero().squeeze().type(torch.int64)
                atom_idx = my_neighbor[neighbor_idx].type(torch.int64) - 1
                # calculate Force
                #   a.shape = [neighbor_num, 1, self.dim_feat]
                #   b.shape = [neighbor_num, self.dim_feat, 3]
                #   Force.shape = [batch_size, self.natoms, 3]
                a = result_dEi_dFeat[batch_idx, atom_idx].unsqueeze(1)
                b = dfeat[batch_idx, i, neighbor_idx]
                Force[batch_idx, i, :] = torch.matmul(a, b).sum([0, 1])

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
