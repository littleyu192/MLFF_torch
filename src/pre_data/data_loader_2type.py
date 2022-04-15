#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
codepath = os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../lib')
sys.path.append(os.getcwd())
import prepare as pp
import parameters as pm


class MovementDataset(Dataset):

    def __init__(self, feat_path, dfeat_path,
                 egroup_path, egroup_weight_path, divider_path,
                 itype_path, nblist_path, weight_all_path,
                 energy_path, force_path, ind_img_path, natoms_img_path , dR_neigh_path=None):  # , natoms_path

        super(MovementDataset, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.feat = np.load(feat_path)
        self.dfeat = np.load(dfeat_path)
        self.egroup = np.load(egroup_path)
        self.egroup_weight = np.load(egroup_weight_path)
        self.divider = np.load(divider_path)

        # self.natoms_sum = natoms
        # self.natoms = pd.read_csv(natoms_path)   #/fread_dfeat/NN_output/natoms_train.csv
        self.itype = np.load(itype_path)
        self.nblist = np.load(nblist_path)
        self.weight_all = np.load(weight_all_path)
        self.ind_img = np.load(ind_img_path)

        self.energy = np.load(energy_path)
        # etot alignment for cu
        '''
        self.dE = np.load("/home/husiyu/software/MLFFdataset/v1_cu1646/PWdata/energy.npy") #dE
        dE = self.dE[:1316]/108.0  # self.energy.shape[0]/108 = 1316
        dE1 = np.repeat(dE,108).reshape(-1, 1)
        self.energy = dE1[:self.energy.shape[0]] + self.energy
        
        # etot alignment for cuo
        self.dE = np.loadtxt("/home/husiyu/software/MLFFdataset/v1_cuo1000/PWdata/energy.txt") #dE
        dE = self.dE[:800]/64.0  # self.energy.shape[0]/64 = 800
        dE1 = np.repeat(dE,64).reshape(-1, 1)
        self.energy = dE1[:self.energy.shape[0]] + self.energy
        '''
        
        self.force = np.load(force_path)
        self.use_dR_neigh = False
        
        self.ntypes = pm.ntypes
        self.natoms_img = np.load(natoms_img_path)
        if dR_neigh_path:
            self.use_dR_neigh = True
            tmp = np.load(dR_neigh_path)
            # import ipdb;ipdb.set_trace()
            self.dR = tmp[:, :, :, :3]
            self.dR_neigh_list = np.squeeze(tmp[:, :, :, 3:], axis=-1).astype(int)
            self.force = -1 * self.force
        
    def __getitem__(self, index):
        # index = index + 10
        ind_image = np.zeros(2)
        ind_image[0] = self.ind_img[index]
        ind_image[1] = self.ind_img[index+1]
        dic = {
            'input_feat': self.feat[self.ind_img[index]:self.ind_img[index+1]],
            'input_dfeat': self.dfeat[self.ind_img[index]:self.ind_img[index+1]],
            'input_egroup': self.egroup[self.ind_img[index]:self.ind_img[index+1]],
            'input_egroup_weight': self.egroup_weight[self.ind_img[index]:self.ind_img[index+1]],
            'input_divider': self.divider[self.ind_img[index]:self.ind_img[index+1]],
            'input_itype': self.itype[self.ind_img[index]:self.ind_img[index+1]],
            'input_nblist': self.nblist[self.ind_img[index]:self.ind_img[index+1]],
            'input_weight_all': self.weight_all[self.ind_img[index]:self.ind_img[index+1]],

            'output_energy': self.energy[self.ind_img[index]:self.ind_img[index+1]],
            'output_force': self.force[self.ind_img[index]:self.ind_img[index+1]],
            'ind_image': ind_image,
            'natoms_img': self.natoms_img[index]
        }
        if self.use_dR_neigh:
            dic['input_dR'] = self.dR[self.ind_img[index]:self.ind_img[index+1]]
            dic['input_dR_neigh_list'] = self.dR_neigh_list[self.ind_img[index]:self.ind_img[index+1]]
        return dic
    
    def __len__(self):
        image_number = self.ind_img.shape[0] - 1
        return image_number

    def __compute_std(self, sum2, sum, sumn):
        if sumn == 0:
            return 1e-2
        val = np.sqrt(sum2/sumn - np.multiply(sum/sumn, sum/sumn))
        if np.abs(val) < 1e-2:
            val = 1e-2
        return val

    def __smooth(self, x, Ri_xyz, mask, inr):

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

        vv_copy = vv.unsqueeze(-1).repeat(1, 1, 1, 4)
        Ri[mask] *= vv_copy[mask]
        return Ri

    def __compute_stat(self, image_num=10):

        self.davg = []
        self.dstd = []
        # self.natoms = sum(self.natoms)

        # only for one atom type
        if image_num > self.__len__():
            image_num = self.__len__()
        image_dR = self.dR[self.ind_img[0]:self.ind_img[image_num]]
        list_neigh = self.dR_neigh_list[self.ind_img[0]:self.ind_img[image_num]]

        natoms_sum = self.natoms_img[0, 0]
        natoms_per_type = self.natoms_img[0, 1:]

        image_dR = np.reshape(image_dR, (-1, natoms_sum, self.ntypes * pm.maxNeighborNum, 3))
        list_neigh = np.reshape(list_neigh, (-1, natoms_sum, self.ntypes * pm.maxNeighborNum))

        image_dR = torch.tensor(image_dR, device=self.device, dtype=torch.double)
        list_neigh = torch.tensor(list_neigh, device=self.device, dtype=torch.int)

        # deepmd neighbor id 从 0 开始，MLFF从1开始
        mask = list_neigh > 0

        dR2 = torch.zeros_like(list_neigh, dtype=torch.double)
        Rij = torch.zeros_like(list_neigh, dtype=torch.double)
        dR2[mask] = torch.sum(image_dR[mask] * image_dR[mask], -1)
        Rij[mask] = torch.sqrt(dR2[mask])

        nr = torch.zeros_like(dR2)
        inr = torch.zeros_like(dR2)

        dR2_copy = dR2.unsqueeze(-1).repeat(1, 1, 1, 3)
        Ri_xyz = torch.zeros_like(dR2_copy)

        nr[mask] = dR2[mask] / Rij[mask]
        Ri_xyz[mask] = image_dR[mask] / dR2_copy[mask]
        inr[mask] = 1 / Rij[mask]
        Ri = self.__smooth(nr, Ri_xyz, mask, inr)
        Ri2 = Ri * Ri

        # Ri = Ri.reshape((-1, 4))

        atom_sum = 0

        for i in range(self.ntypes):
            Ri_ntype = Ri[:, atom_sum:atom_sum+natoms_per_type[i]].reshape(-1, 4)
            Ri2_ntype = Ri2[:, atom_sum:atom_sum+natoms_per_type[i]].reshape(-1, 4)
            sum_Ri = Ri_ntype.sum(axis=0).tolist()
            sum_Ri_r = sum_Ri[0]
            sum_Ri_a = np.average(sum_Ri[1:])
            sum_Ri2 = Ri2_ntype.sum(axis=0).tolist()
            sum_Ri2_r = sum_Ri2[0]
            sum_Ri2_a = np.average(sum_Ri2[1:])
            sum_n = Ri_ntype.shape[0]


            davg_unit = [sum_Ri[0] / (sum_n + 1e-15), 0, 0, 0]
            dstd_unit = [
                self.__compute_std(sum_Ri2_r, sum_Ri_r, sum_n),
                self.__compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
                self.__compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
                self.__compute_std(sum_Ri2_a, sum_Ri_a, sum_n)
            ]

            self.davg.append(np.tile(davg_unit, pm.maxNeighborNum * self.ntypes).reshape(-1, 4))
            self.dstd.append(np.tile(dstd_unit, pm.maxNeighborNum * self.ntypes).reshape(-1, 4))
            atom_sum = atom_sum + natoms_per_type[i]
        
        self.davg = np.array(self.davg).reshape(self.ntypes, -1)
        self.dstd = np.array(self.dstd).reshape(self.ntypes, -1)      
        # import ipdb;ipdb.set_trace()  


    def __compute_stat_output(self, image_num=10,  rcond=1e-3):
        self.ener_shift = []
        natoms_sum = self.natoms_img[0, 0]
        natoms_per_type = self.natoms_img[0, 1:]
        # only for one atom type
        if image_num > self.__len__():
            image_num = self.__len__()
        energy = self.energy[self.ind_img[0]:self.ind_img[image_num]]
        energy = np.reshape(energy, (-1, natoms_sum, 1))
        # natoms_sum = 0
        # for ntype in range(self.ntypes):
        #     energy_ntype = energy[:, natoms_sum:natoms_sum+natoms_per_type[ntype]]
        #     natoms_sum += natoms_per_type[ntype]
        #     energy_sum = energy_ntype.sum(axis=1)
        #     energy_one = np.ones_like(energy_sum) * natoms_per_type[ntype]
        #     ener_shift, _, _, _ = np.linalg.lstsq(energy_one, energy_sum, rcond=rcond)
        #     self.ener_shift.append(ener_shift[0, 0])
        # energy_ntype = energy[:, natoms_sum:natoms_sum+natoms_per_type[ntype]]
        # natoms_sum += natoms_per_type[ntype]
        energy_sum = energy.sum(axis=1)
        energy_avg = np.average(energy_sum)
        # energy_one = np.ones_like(energy_sum) * natoms_per_type[ntype]
        ener_shift, _, _, _ = np.linalg.lstsq([natoms_per_type], [energy_avg], rcond=rcond)
        self.ener_shift = ener_shift.tolist()
        
        

    def get_stat(self, image_num=20, rcond=1e-3):
        # image_num = batch_size * batch_stat_num
        self.__compute_stat(image_num)
        self.__compute_stat_output(image_num, rcond)
        return self.davg, self.dstd, self.ener_shift


def get_torch_data(examplespath):
    '''
    input para:
    examplespath : npy_file_dir
    data_file_frompwmat : read train_data.csv or test_data.csv
    '''
    # examplespath='./train_data/final_train'   # for example
    f_feat = os.path.join(examplespath+'/feat_scaled.npy')
    if pm.dR_neigh:
        f_dR_neigh = os.path.join(examplespath+'/dR_neigh.npy')
    else:
        f_dR_neigh = None
    f_dfeat = os.path.join(examplespath+'/dfeat_scaled.npy')
    f_egroup = os.path.join(examplespath+'/egroup.npy')
    f_egroup_weight = os.path.join(examplespath+'/egroup_weight.npy')
    f_divider = os.path.join(examplespath+'/divider.npy')

    f_itype = os.path.join(examplespath+'/itypes.npy')
    f_nblist = os.path.join(examplespath+'/nblist.npy')
    f_weight_all = os.path.join(examplespath+'/weight_all.npy')
    ind_img = os.path.join(examplespath+'/ind_img.npy')
    natoms_img = os.path.join(examplespath+'/natoms_img.npy')

    f_energy = os.path.join(examplespath+'/engy_scaled.npy')
    f_force = os.path.join(examplespath+'/fors_scaled.npy')
    # f_force = os.path.join(examplespath+'/force.npy')

    torch_data = MovementDataset(f_feat, f_dfeat,
                                 f_egroup, f_egroup_weight, f_divider,
                                 f_itype, f_nblist, f_weight_all,
                                 f_energy, f_force, ind_img, natoms_img, f_dR_neigh)
    return torch_data
