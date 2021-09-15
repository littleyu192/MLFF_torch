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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def d_sigmoid(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))

class MLFF_dmirror(nn.Module):
    def __init__(self, activation_type):
        super(MLFF_dmirror, self).__init__()
        self.atomType = pm.atomType
        self.natoms = pm.natoms[0]
        self.net_cfg = pm.MLFF_dmirror_cfg2
        self.dim_feat = pm.nFeatures
        if (activation_type == 'sigmoid'):
            self.activation_type = 'sigmoid'
            self.net = dmirror_FC(self.net_cfg, torch.sigmoid, d_sigmoid)
            print("MLFF_dmirror: using sigmod activation")
        elif (activation_type == 'softplus'):
            self.activation_type = 'softplus'
            self.net = dmirror_FC(self.net_cfg, F.softplus, F.sigmoid)
            print("MLFF_dmirror: using softplus activation")
        else:
            raise RuntimeError("MLFF_dmirror: unsupported activation_type: %s" %activation_type)
        #print(self.natoms)
        #print("111111111111111111")

    def forward(self, image, dfeat, neighbor, Egroup_weight, divider):
        print("defat.shape= ", dfeat.shape)
        print("neighbor.shape = ", neighbor.shape)
        print("dump dfeat ------------------->")
        print(dfea)
        print("dump neighbor ------------------->")
        print(neighbor)
        #print("2222222222222222222")
        batch_size = image.shape[0]
        #print(batch_size)
        #print("3333333333333333333")
        result_Ei = torch.zeros(
            (batch_size, self.natoms)
        ).to(device)
        result_dEi_dFeat = torch.zeros(
            (batch_size, self.natoms, self.dim_feat)
        ).to(device)

        for batch_idx in range(batch_size):
            for i in range(self.natoms):
                Ei, dEi_dFeat = self.net(image[batch_idx, i, :])
                result_Ei[batch_idx, i] = Ei
                result_dEi_dFeat[batch_idx, i, :] = dEi_dFeat

        Etot = torch.sum(result_Ei, 1)
        Force = torch.zeros((batch_size, self.natoms, 3)).to(device)
        result_dEtot_dFeat = result_dEi_dFeat.view(
            [batch_size, self.natoms * self.dim_feat]
        )

        for batch_idx in range(batch_size):
            for i in range(self.natoms):
                a = result_dEtot_dFeat[batch_idx].unsqueeze(0)
                b0 = dfeat[batch_idx, i, :self.natoms]  # FIXME: this is fool?
                b = b0.view([self.natoms * self.dim_feat, 3])
                Force[batch_idx, i, :] = torch.mm(a, b)

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
