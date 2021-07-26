# -*- coding: utf-8 -*-
import torch
import numbers
import os,sys
import random
import time
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
from torch.autograd import Variable
import math
sys.path.append("../data")
import parameters as pm 
# dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data_process_path=os.path.join(dir_mytest+'\data')
# print(sys.path.insert(0,data_process_path))
# print(dir_mytest)
from data_loader_2type import MovementDataset, get_torch_data

#import写好的不同类型的网络
from FC import FCNet  


torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

grad1=torch.from_numpy(np.ones((40,42))).float().to(device)
# grad2=torch.rand([40,42], device='cuda:0')
grad3=torch.rand([40,42])
# print(grad2)

# ==========================part1:数据读取==========================
batch_size = 40   #40
train_data_path='../data/train_data/final_train'
train_data_file_frompwmat = '/train_data.csv'
torch_train_data = get_torch_data(pm.natoms, train_data_path, train_data_file_frompwmat)
test_data_path='../data/train_data/final_test'
test_data_file_frompwmat = '/test_data.csv'
torch_test_data = get_torch_data(pm.natoms, test_data_path, test_data_file_frompwmat)

loader_train = Data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=True)
loader_test = Data.DataLoader(torch_test_data, batch_size=batch_size, shuffle=True)

for iter, value in enumerate(loader_train):
    print(value['input_dfeat'].size())   #  torch.size([batchsize, image, neigh, feature, 3direc])
    print(value['input_dfeat'])
    dfeat=value['input_dfeat']
    force=value['output_force']    #[40,108,3]
    # treat grad1 as Ei backward gradient [40*42]
    for i in range(40):
        atomi_dx, atomi_dy, atomi_dz = [0,0,0]
        for j in range(42):   #一个原子的受力
            atomi_dx = atomi_dx + grad1[i,j]*dfeat[i,0,:,j,0].sum() 
            atomi_dy = atomi_dy + grad1[i,j]*dfeat[i,0,:,j,1].sum()
            atomi_dz = atomi_dz + grad1[i,j]*dfeat[i,0,:,j,2].sum()
        atomi_force = torch.tensor([atomi_dx, atomi_dy, atomi_dz])  #反向计算出的力
        force_deviation = force[i,0,:] - atomi_force   #和labelF的偏差
        force_ABS_error = force.norm(1) / 3
        force_RMSE_error = math.sqrt(1/3) * force.norm(2)
        print(atomi_force)
        print(force_RMSE_error)
        import ipdb;ipdb.set_trace()

            

    print(value['output_force'].size())
    print(value['input_itype'])
    # print(value['output_Etot_pwmat'].shape)
    print(len(loader_train))     #135864/108/40=31.45=>32
    import ipdb;ipdb.set_trace()


