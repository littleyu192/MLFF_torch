#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import re
from ipdb.__main__ import set_trace
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../lib')
sys.path.append(os.getcwd())
import parameters as pm
import torch.utils.data as Data


def get_data_property(filename, valid_ratio):
    # data loader
    f = open(filename, 'r')
    lines = f.readlines()

    # the atom number in one image
    lines0 = lines[0]
    atoms_number_in_one_image = int(re.findall(r"(.*) atom", lines0)[0])
    print("atoms number:", atoms_number_in_one_image)

    # number of all the samples
    number_of_samples = 0
    for lines in lines:
        if "-----------" in lines:
            number_of_samples = number_of_samples + 1
    train_samples = int(number_of_samples * (1 - valid_ratio))
    print("all samples:", number_of_samples)
    print("train samples:", train_samples)
    valid_samples = number_of_samples - train_samples
    return atoms_number_in_one_image, train_samples


def get_data(filename, atoms_number_in_one_image, train_data_file, valid_data_file, train_samples):
    # the generated data, delete them if exist already
    # if os.path.exists(train_data_file):
    #     os.remove(train_data_file)
    # if os.path.exists(valid_data_file):
    #     os.remove(valid_data_file)

    f = open(filename, 'r')
    lines = f.readlines()
    atoms_position_all = []
    atoms_force_all = []
    atoms_energy_all = []
    Etot = []
    Ep = []
    Ek = []
    for (num, value) in enumerate(lines):
        if value.find("atoms") != -1:
            # print(value)
            Etot.append(eval(value.split(",")[4].split()[3]))
            Ep.append(eval(value.split(",")[4].split()[4]))
            Ek.append(eval(value.split(",")[4].split()[5]))
        elif lines[num].startswith(" Position"):
            atom_position = []
            for i in range(1, atoms_number_in_one_image + 1):
                position = lines[num + i].split(' ')
                position = [x for x in position if x != '']
                isotope = [int(x) for x in position[0]]  # 同位素个数
                position_xyz = [float(x) for x in position[1:4]]
                # print(position)
                atom_position.append(position_xyz[0])
                atom_position.append(position_xyz[1])
                atom_position.append(position_xyz[2])
            atoms_position_all.append(atom_position)
        elif lines[num].startswith(" Force"):
            atom_force = []
            for i in range(1, atoms_number_in_one_image + 1):
                force = lines[num + i].split(' ')
                force = [x for x in force if x != '']
                force_xyz = [float(x) for x in force[1:4]]
                atom_force.append(force_xyz[0])
                atom_force.append(force_xyz[1])
                atom_force.append(force_xyz[2])
            atoms_force_all.append(atom_force)
        elif lines[num].startswith(" Atomic-Energy"):
            atom_energy = []
            for i in range(1, atoms_number_in_one_image + 1):
                energy= lines[num + i].split(' ')
                energy = [x for x in energy if x != '']
                energy_diff = [float(x) for x in energy[1:4]]
                atom_energy.append(energy_diff[0])
            atoms_energy_all.append(atom_energy)

    # df_x = pd.DataFrame(atoms_position_all)
    df_y1 = pd.DataFrame(atoms_force_all)
    df_y2 = pd.DataFrame(Etot, columns=['Etot'])
    df_y3 = pd.DataFrame(atoms_energy_all)
    # df_left = df_y1.join([df_y2, df_y3])
    df_temp = df_y1.join(df_y3, lsuffix='_f', rsuffix='_e') # p for position ,f for force, e for atom energy 
    df = df_temp.join(df_y2)
    df_train = df[:train_samples]
    df_valid = df[train_samples:]
    df_train.to_csv(train_data_file, index=False)
    df_valid.to_csv(valid_data_file, index=False)

class MovementDataset(Dataset):
    
    def __init__(self, natoms, feat_path, dfeat_path,
                 egroup_path, egroup_weight_path, divider_path, 
                 itype_path, nblist_path,weight_all_path,
                 energy_path, force_path, ind_img_path, dR_neigh_path=None):  # , natoms_path
        super(MovementDataset, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feat=np.load(feat_path)
        self.dfeat=np.load(dfeat_path)
        self.egroup=np.load(egroup_path)
        self.egroup_weight=np.load(egroup_weight_path)
        self.divider=np.load(divider_path)

        self.natoms = natoms
        # self.natoms = pd.read_csv(natoms_path)   #/fread_dfeat/NN_output/natoms_train.csv
        self.itype=np.load(itype_path)
        self.nblist=np.load(nblist_path)
        self.weight_all=np.load(weight_all_path)
        self.ind_img = np.load(ind_img_path)

        self.energy=np.load(energy_path)
        self.force=np.load(force_path)

        self.use_dR_neigh = False
        if dR_neigh_path:
            self.use_dR_neigh = True
            tmp = np.load(dR_neigh_path)
            self.dR = tmp[:,:,:3]
            self.dR_neigh_list = np.squeeze(tmp[:,:,3:],axis=-1)
            

        # label = pd.read_csv(label_path)
        # labels_Fi = [str(x) + "_f" for x in range(324)]
        # labels_Ei = [str(x) + "_e" for x in range(324)]
        # self.labels_Fi = label[labels_Fi]
        # self.labels_Ei = label[labels_Ei]
        # self.labels_Etot = label["Etot"]

    def __getitem__(self, index):
        ind_image = np.zeros(2)
        ind_image[0] = self.ind_img[index]
        ind_image[1] = self.ind_img[index+1]
        dic = {
            'input_feat': self.feat[self.ind_img[index]:self.ind_img[index+1]],
            'input_dfeat': self.dfeat[self.ind_img[index]:self.ind_img[index+1]],
            'input_egroup': self.egroup[self.ind_img[index]:self.ind_img[index+1]],
            'input_egroup_weight':self.egroup_weight[self.ind_img[index]:self.ind_img[index+1]],
            'input_divider':self.divider[self.ind_img[index]:self.ind_img[index+1]],
            # 'input_natoms':torch.from_numpy(np.array(self.natoms[index:index + 1])).int8,
            'input_itype':self.itype[self.ind_img[index]:self.ind_img[index+1]],
            'input_nblist':self.nblist[self.ind_img[index]:self.ind_img[index+1]],
            'input_weight_all':self.weight_all[self.ind_img[index]:self.ind_img[index+1]],

            'output_energy':self.energy[self.ind_img[index]:self.ind_img[index+1]],
            'output_force':self.force[self.ind_img[index]:self.ind_img[index+1]],
            'ind_image': ind_image
            # 'output_Etot_pwmat': torch.from_numpy(np.array(self.labels_Etot[index:index + 1])).float(),
            # 'output_energy_pwmat': torch.from_numpy(np.array(self.labels_Ei[index:index + 1])[0]).float(),
            # 'output_Fi_pwmat': torch.from_numpy(np.array(self.labels_Fi[index:index + 1])[0]).float(),
        }
        if self.use_dR_neigh:
            dic['input_dR'] = self.dR[self.ind_img[index]:self.ind_img[index+1]]
            dic['input_dR_neigh_list'] = self.dR_neigh_list[self.ind_img[index]:self.ind_img[index+1]]
            dic['input_davg'] = self.davg
            dic['input_dstd'] = self.dstd
            dic['input_ener_shift'] = self.ener_shift
            # import ipdb;ipdb.set_trace()
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

    def __smooth(self, x, Ri_xyz, mask):

        uu = torch.zeros_like(x)
        vv = torch.zeros_like(x)
        
        res = torch.zeros_like(x)

        # x < rcut_min vv = 1
        mask_min = x < 5.8   #set rcut=25, 10  min=0,max=30
        mask_1 = mask & mask_min  #[2,108,100]
        vv[mask_1] = 1

        # rcut_min< x < rcut_max
        mask_max = x < 6.0
        mask_2 = ~mask_min & mask_max & mask
        # uu = (xx - rmin) / (rmax - rmin) ;
        uu[mask_2] = (x[mask_2] - 5.8)/(6.0 -5.8)
        vv[mask_2] = uu[mask_2] * uu[mask_2] * uu[mask_2] * (-6 * uu[mask_2] * uu[mask_2] + 15 * uu[mask_2] - 10) + 1
        # dd = ( 3 * uu*uu * (-6 * uu*uu + 15 * uu - 10) + uu*uu*uu * (-12 * uu + 15) ) * du;
 
        mask_3 = ~mask_max & mask
        vv[mask_3] = 0

        res[mask] = vv[mask] / x[mask]
        vv_copy = vv.unsqueeze(-1).repeat(1, 1, 1, 3)
        Ri_xyz[mask] *= vv_copy[mask]
        Ri = torch.cat((res.unsqueeze(-1), Ri_xyz), dim=-1)
        return Ri

    def __compute_stat(self, image_num = 10):
        # only for one atom type
        if image_num > self.__len__():
            image_num = self.__len__()
        image_dR = self.dR[self.ind_img[0]:self.ind_img[image_num]]
        list_neigh = self.dR_neigh_list[self.ind_img[0]:self.ind_img[image_num]]
        image_dR = np.reshape(image_dR, (-1, self.natoms[0], pm.maxNeighborNum, 3))
        list_neigh = np.reshape(list_neigh, (-1, self.natoms[0], pm.maxNeighborNum))

        image_dR = torch.tensor(image_dR, device=self.device)
        list_neigh = torch.tensor(list_neigh, device=self.device)

        # image_dR = torch.tensor(np.load("rij.npy"), device=self.device)
        # list_neigh = torch.tensor(np.load("nblist.npy"), device=self.device)
        # image_dR = image_dR.reshape(1, 108, 100, 3)
        # list_neigh = list_neigh.reshape(1, 108, 100)
        # list_neigh = list_neigh + 1

        # deepmd neighbor id 从 0 开始，MLFF从1开始
        mask = list_neigh > 0

        dR2 = torch.zeros_like(list_neigh, dtype=torch.double)
        Rij = torch.zeros_like(list_neigh, dtype=torch.double)
        dR2[mask] = torch.sum(image_dR[mask] * image_dR[mask], -1) 
        Rij[mask] = torch.sqrt(dR2[mask])
        
        nr = torch.zeros_like(dR2)
        Ri_xyz = torch.zeros((image_num, self.natoms[0], pm.maxNeighborNum, 3), device=self.device)
        dR2_copy = dR2.unsqueeze(-1).repeat(1, 1, 1, 3)

        nr[mask] = dR2[mask] / Rij[mask]
        Ri_xyz[mask] = image_dR[mask] / dR2_copy[mask]
        Ri = self.__smooth(nr, Ri_xyz, mask)

        Ri = Ri.reshape((-1, 4))

        np.save("torch_stat_Ri.npy", Ri.cpu().numpy())
        Ri2 = Ri * Ri
        
        sum_Ri = Ri.sum(axis=0).tolist()
        sum_Ri_r = sum_Ri[0]
        sum_Ri_a = np.average(sum_Ri[1:])
        sum_Ri2 = Ri2.sum(axis=0).tolist()
        sum_Ri2_r = sum_Ri2[0]
        sum_Ri2_a = np.average(sum_Ri2[1:])
        sum_n = Ri.shape[0]

        davg_unit = [sum_Ri[0] / (sum_n + 1e-15), 0, 0, 0]
        dstd_unit = [
            self.__compute_std(sum_Ri2_r, sum_Ri_r, sum_n),
            self.__compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
            self.__compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
            self.__compute_std(sum_Ri2_a, sum_Ri_a, sum_n)
        ]

        self.davg = np.tile(davg_unit, pm.maxNeighborNum).reshape(-1, 4)
        self.dstd = np.tile(dstd_unit, pm.maxNeighborNum).reshape(-1, 4)

        np.save("torch_davg.npy", self.davg)
        np.save("torch_dstd.npy", self.dstd)
    
    def __compute_stat_output(self, image_num = 10,  rcond = 1e-3):
        # only for one atom type
        if image_num > self.__len__():
            image_num = self.__len__()
        energy = self.energy[self.ind_img[0]:self.ind_img[image_num]]
        energy = np.reshape(energy, (-1, self.natoms[0], 1))
        etot = energy.sum(axis=1)
        natoms = np.ones_like(etot) * self.natoms[0]
        ener_shift, _, _, _ = np.linalg.lstsq(natoms, etot, rcond = rcond)
        self.ener_shift = ener_shift[0, 0]
    

    def get_stat(self, image_num=20, rcond=1e-3):
        # image_num = batch_size * batch_stat_num
        self.__compute_stat(image_num)
        self.__compute_stat_output(image_num, rcond)
        return self.davg, self.dstd, self.ener_shift





def get_pwmat_data(movement_filename, train_data_file_frompwmat, test_data_file_frompwmat):
    movement_filename = pm.trainSetDir+'/MOVEMENT'
    train_data_file_frompwmat = './train_data/final_train/train_data.csv'
    test_data_file_frompwmat = './train_data/final_test/valid_data.csv'
    atoms_number_in_one_image, train_samples = get_data_property(filename, pm.test_ratio)
    if not os.path.exists(train_data_file):
        print("==========generate train and valid data from PWmat============")
        get_data(filename, atoms_number_in_one_image, 
        train_data_file, valid_data_file, train_samples)   # 读取PWmat数据的etot，force
    else:
        print("==========train_data.csv and test_data.csv exist!==========")



def get_torch_data(atoms_number_in_one_image, examplespath):
    '''
    input para:
    atoms_number_in_one_image : can be read from parameter file, e.g. cu case 108
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

    f_energy = os.path.join(examplespath+'/engy_scaled.npy')
    f_force = os.path.join(examplespath+'/fors_scaled.npy') 

    torch_data = MovementDataset(atoms_number_in_one_image, f_feat, f_dfeat,
                 f_egroup, f_egroup_weight, f_divider, 
                 f_itype, f_nblist, f_weight_all,
                 f_energy, f_force, ind_img, f_dR_neigh)
    return torch_data

def main():
    batch_size = pm.batch_size   #40
    train_data_path=pm.train_data_path
    torch_train_data = get_torch_data(pm.natoms, train_data_path)
    loader_train = Data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=True) 
    #是否打乱顺序，多线程读数据num_workers=4

    for i, value in enumerate(loader_train):
        print(i)
        print(value['input_feat'].size())   #  torch.size([batchsize, 42])
        print(value['input_feat'])
        print(value['output_force'].size())
        print(value['input_itype'])
        # print(value['output_Etot_pwmat'].shape)
        print(len(loader_train))     #135864/108/40=31.45=>32

if __name__ == "__main__":
    main()
