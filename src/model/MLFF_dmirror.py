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
if pm.torch_dtype == 'float32':
    torch_dtype = torch.float32
    print('info: torch.dtype = torch.float32 in Pytorch training.')
else:
    torch_dtype = torch.float64
    print('info: torch.dtype = torch.float64 in Pytorch training. (it may be slower)')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLFF_dmirror(nn.Module):
    def __init__(self):
        super(MLFF_dmirror, self).__init__()
        self.atomType = pm.atomType
        self.natoms = pm.natoms
        self.net_cfg = pm.MLFF_dmirror_cfg
        self.dim_feat = pm.nFeatures
        self.net = dmirror_FC(net_cfg, F.softplus, F.sigmoid)
        print(self.natoms)
        print("111111111111111111")

    def forward(self, image, dfeat, neighbor, Egroup_weight, divider):
        print(dfeat.shape)
        print(neighbor.shape)
        print("2222222222222222222")
        batch_size = image.shape[0]
        result_Ei = torch.zeros(
            [batch_size, self.natoms], dtype=torch.float64
        )
        result_dEi_dFeat = torch.zeros(
            [batch_size, self.natoms, self.dim_feat], dtype=torch.float64
        )

        for batch_idx in range(batch_size):
            for i in range(self.natoms):
                Ei, dEi_dFeat = self.net(image[batch_idx, i, :])
                result_Ei[batch_idx, i, 0] = Ei
                result_dEi_dFeat[batch_idx, i, :] = dEi_dFeat

        Etot = torch.sum(result_Ei, 1)
        Force = torch.zeros([batch_size, self.natoms, 3], dtype=torch.float64)
        result_dEtot_dFeat = result_dEi_dFeat.view(
            [batch_size, self.natoms * self.dim_feat]
        )

        for batch_idx in range(batch_size):
            for i in range(self.natoms):
                Force[batch_idx, i, :] = result_dEtot_dFeat[batch_idx, :]



        return Etot, Force

#        '''
        for batch_idx in range(batch_size):
            dFeat dfeat[batch_index, atom_index_temp + i, nei, :, :]
            

        Force = torch.matmul(

            for idx, natom in enumerate(self.natoms):  #[32,32]  [108] 
            if(batch_index==0):
                batches_Ei = Ei.unsqueeze(0) #[108]-->[1,108]
                Etot = one_sample_Etot.unsqueeze(0) #[1] --> [1,1]
                input_grad_allatoms = one_sample_input_grad_allatoms.unsqueeze(0) #[108,42]-->[1,108,42]
            else:
                batches_Ei = torch.cat((batches_Ei, Ei.unsqueeze(0)), dim=0)  #[1,108] --> [2,108]
                Etot = torch.cat((Etot, one_sample_Etot.unsqueeze(0)), dim=0) #[1,1]-->[2,1]   -->2
                input_grad_allatoms = torch.cat((input_grad_allatoms, one_sample_input_grad_allatoms.unsqueeze(0)), dim=0) #[2,108,42]

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
                        if(nei_index == -1):
                            break 
                        atom_force += torch.matmul(input_grad_allatoms[batch_index, nei_index, :], dfeat[batch_index, atom_index_temp + i, nei, :, :])
                        # print("The dEtot/dfeature for batch_index %d, neighbor_inde %d" %(batch_index, nei_index))
                        # print(input_grad_allatoms[batch_index, nei_index, :])
                    Force[batch_index, atom_index_temp+i] = atom_force

        # Egroup = self.get_egroup(Ei, Egroup_weight, divider)

        return Etot, Force

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
