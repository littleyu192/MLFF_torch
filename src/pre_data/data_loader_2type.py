#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import re
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
            self.dR_neigh = np.load(dR_neigh_path)

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
            dic['input_dR_neigh'] = self.dR_neigh[self.ind_img[index]:self.ind_img[index+1]]
        return dic

    def __len__(self):
        image_number = self.ind_img.shape[0] - 1 
        return image_number

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
    f_dR_neigh = os.path.join(examplespath+'/dR_neigh.npy')
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
