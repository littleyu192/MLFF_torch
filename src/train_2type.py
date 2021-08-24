# -*- coding: utf-8 -*-
import torch
import numbers
import os,sys
codepath=os.path.abspath(sys.path[0])
import random
import time
import math
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
from torch.nn.modules import loss
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
writer = SummaryWriter()

sys.path.append(os.getcwd())
import parameters as pm 
sys.path.append(codepath+'/pre_data')
from data_loader_2type import MovementDataset, get_torch_data

sys.path.append(codepath+'/model')
from FC import FCNet  

torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================part1:数据读取==========================
batch_size = pm.batch_size   #40
train_data_path=pm.train_data_path
train_data_file_frompwmat = pm.train_data_path + './train_data.csv'
torch_train_data = get_torch_data(pm.natoms, train_data_path, train_data_file_frompwmat)
test_data_path= pm.test_data_path
test_data_file_frompwmat = pm.test_data_path + './test_data.csv'
torch_test_data = get_torch_data(pm.natoms, test_data_path, test_data_file_frompwmat)

loader_train = Data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=True)
loader_test = Data.DataLoader(torch_test_data, batch_size=batch_size, shuffle=True)

# for i, value in enumerate(loader_train):
#     print(i)
#     print(value['input_feat'].size())   #  torch.size([batchsize, 42])
#     print(value['input_feat'])
#     print(value['output_force'].size())
#     print(value['input_itype'])
#     # print(value['output_Etot_pwmat'].shape)
#     print(len(loader_train))     #135864/108/40=31.45=>32
#     import ipdb;ipdb.set_trace()

# ==========================part2:train和valid==========================
def train(sample_batches, models, optimizers, criterion):
    error=0
    # backward_force=torch.tensor
    atom_type = Variable(sample_batches['input_itype'].int().to(device))   #[40,64] CuO  32个29,Cu 32个8,O
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1]
    len_batch = sample_batches["input_feat"].shape[0]
    out_atoms_energy = torch.zeros(len_batch, 1).to(device)
    atom_index_temp = 0
    for itype in range(len(pm.atomType)): 
        for i in range(pm.natoms[itype]):
            input_data = Variable(sample_batches['input_feat'][:,atom_index_temp + i,:].float().to(device),requires_grad=True)
            label = Variable(sample_batches['output_energy'][:,atom_index_temp + i,:].float().to(device))
            model = models[itype]
            model.to(device)
            model.train()
            x, out = model(input_data)
            # out_sum = out.mean()
            # out_sum.backward(retain_graph=True)
            # input_grad = input_data.grad  #input_data.grad.shape --> torch.size([40,42]) 
            out_atoms_energy = out_atoms_energy + out  # 加入Etot，重新计算此处的loss
        atom_index_temp = atom_index_temp + pm.natoms[itype]
    Etot_deviation = out_atoms_energy - Etot_label
    Etot_shape = Etot_label.shape[0]  #40
    Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)
    optimizer = optimizers
    optimizer.zero_grad()
    loss=Etot_RMSE_error
    loss.backward()
    optimizer.step()
    error = error+float(loss.item())
    return error
    
def valid(sample_batches, models, criterion):
    error=0
    len_batch = sample_batches["input_feat"].shape[0]
    out_atoms_energy = torch.zeros(len_batch, 1).to(device)
    atom_index_temp = 0
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
    Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))
    Etot_label = torch.sum(Etot_label, dim=1)
    Etot_predict = 0
    for itype in range(len(pm.atomType)):
        for i in range(pm.natoms[itype]):
            input_data = Variable(sample_batches['input_feat'][:,atom_index_temp + i,:].float().to(device))
            label = Variable(sample_batches['output_energy'][:,atom_index_temp + i,:].float().to(device))
            model = models[itype]
            model.to(device)
            model.eval()
            x, out = model(input_data)
            out_atoms_energy = out_atoms_energy + out
        atom_index_temp = atom_index_temp + pm.natoms[itype]
    Etot_deviation = out_atoms_energy - Etot_label
    Etot_shape = Etot_label.shape[0]  #40
    Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)
    loss = Etot_RMSE_error
    error = error+float(loss.item())
    return error

def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

def train_finetuning(sample_batches, models, optimizers, criterion):
    error=0
    # backward_force=torch.tensor
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
    Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))   #[40,108,3]
    Neighbor = Variable(sample_batches['input_nblist'][:,:,:].to(device))  # [40,108,100]
    dfeat=Variable(sample_batches['input_dfeat'][:,:,:,:,:].float().to(device))  #[40,108,100,42,3]
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1] 
    len_batch = sample_batches["input_feat"].shape[0]
    # out_atoms_energy = torch.zeros(len_batch, 1)
    atom_index_temp = 0
    for itype in range(len(pm.atomType)):
        for i in range(pm.natoms[itype]):
            input_data = Variable(sample_batches['input_feat'][:,atom_index_temp + i,:].float().to(device), requires_grad=True)
            label = Variable(sample_batches['output_energy'][:,atom_index_temp + i,:].float().to(device))
            model = models[itype]
            model.to(device)
            model.train()
            _, out = model(input_data)   # out [40,1]
            #===================每个Ei对输入的导数====================
            out_sum = out.sum()
            out_sum.backward(retain_graph=True)
            input_grad = input_data.grad  #input_data.grad.shape --> torch.size([40,42])
            if(atom_index_temp + i ==0):
                input_grad_allatoms=torch.unsqueeze(input_grad,1)
            else: 
                input_grad_allatoms=torch.cat([input_grad_allatoms, torch.unsqueeze(input_grad,1)], dim=1)  #[40,108,42]
            #===================每个Ei的预测值====================
            if(atom_index_temp + i == 0):
                out_atoms_energy = out     #[40,1]
            else:
                out_atoms_energy = torch.cat([out_atoms_energy, out], dim=1)    # [40,108]
        atom_index_temp = atom_index_temp + pm.natoms[itype]
    out_atoms_energy = torch.sum(out_atoms_energy, dim=1)    # [40,1]
    
    #===================每个Fi的预测值====================
    for idx, natom in enumerate(pm.natoms):  #[32,32]
        atom_index_temp = 0
        for i in range(natom):
            force=Variable(sample_batches['output_force'][:,atom_index_temp + i,:].float().to(device))
            neighbor = Variable(sample_batches['input_nblist'][:,atom_index_temp + i,:].to(device))    #[40,100]
            neighbor_number = len(neighbor[0,:])  #100
            for batch in range(len_batch):
                force_dx = torch.zeros([1, input_grad_allatoms.shape[2]]).to(device)
                force_dy = torch.zeros([1, input_grad_allatoms.shape[2]]).to(device)
                force_dz = torch.zeros([1, input_grad_allatoms.shape[2]]).to(device)
                for nei in range(neighbor_number):
                    nei_index = int(neighbor[batch,nei])
                    if(nei_index==0):
                        break
                    sub_force_dx=input_grad_allatoms[batch,nei_index-1,:]*dfeat[batch,atom_index_temp + i,nei,:,0]  #[1,42] * [1,42]
                    sub_force_dy=input_grad_allatoms[batch,nei_index-1,:]*dfeat[batch,atom_index_temp + i,nei,:,1]
                    sub_force_dz=input_grad_allatoms[batch,nei_index-1,:]*dfeat[batch,atom_index_temp + i,nei,:,2]
                    force_dx += sub_force_dx       #[1,42]
                    force_dy += sub_force_dy
                    force_dz += sub_force_dz
                force_dx_reduced = torch.sum(force_dx, dim=1)  #[1,42]->[1]
                force_dy_reduced = torch.sum(force_dy, dim=1)
                force_dz_reduced = torch.sum(force_dz, dim=1)
                atomi_neighbori_force = torch.tensor([force_dx_reduced, force_dy_reduced, force_dz_reduced])  #[3]  Ei的一个neighbor的力
                atomi_neighbori_force = atomi_neighbori_force.unsqueeze(0).to(device)   #[1,3]
                if(batch==0):
                    Force_atomi_predict=atomi_neighbori_force  #[1,3]
                else:
                    Force_atomi_predict=torch.cat((Force_atomi_predict, atomi_neighbori_force), dim=0)  #[40,3]不同的batch的一个中心原子的force
            if(atom_index_temp + i==0):
                Force_predict = Force_atomi_predict.unsqueeze(1)  #[40,1,3]
            else:
                Force_predict = torch.cat((Force_predict, Force_atomi_predict.unsqueeze(1)), dim=1)   #[40,108,3]
        atom_index_temp += natom
        if(idx == 0):
            Forces_predict =  Force_predict  #[40,32,3]
        else:
            Forces_predict = torch.cat((Forces_predict, Force_predict), dim=1)   #[40,108,3]
        
    #===================反向传播更新参数====================
    
    Etot_deviation = out_atoms_energy - Etot_label     # [40,1]
    Etot_shape = Etot_label.shape[0]  #40
    Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)

    Force_deviation = Forces_predict - Force_label
    Force_shape = Force_deviation.shape[0] * Force_deviation.shape[1] * Force_deviation.shape[2]   #40*108*3
    Force_ABS_error = Force_deviation.norm(1) / Force_shape
    Force_RMSE_error = math.sqrt(1/Force_shape) * Force_deviation.norm(2)
        
    optimizer.zero_grad()
    loss = Force_RMSE_error + Etot_RMSE_error
    loss.backward()
    optimizer.step()
    error = error+float(loss.item())
    return error

# ==========================part3:模型训练==========================
n_epoch = 2000
learning_rate = 0.1
weight_decay = 0.9
weight_decay_epoch = 50
direc = './FC3model_minimize_Etot'
if not os.path.exists(direc):
    os.makedirs(direc)

#model的可选项: model=FCNet()  model=FCNet(BN=True)  model=FCNet(Dropout=True)

#==========================Cu，单一元素时==========================
models = [FCNet().to(device)]
optimizer = optim.Adam(models[0].parameters(), lr=learning_rate)
#optimizers = [optim.Adam(models[0].parameters(), lr=learning_rate)]

resume=False  # resume:恢复
if resume:  # 中断的时候恢复训练
    path=r"./FC3model_minimize_Etot/3layers0type1321.pt"
    checkpoint = torch.load(path)
    models[0].load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch=checkpoint['epoch']+1

'''
#==========================CuO，两种元素时==========================
models = [FCNet().to(device), FCNet(itype=1).to(device)]   # BN=False, Dropout=False
optimizer = optim.Adam(models[0].parameters(), lr=learning_rate)
#optimizers = []
#optimizers.append(optim.Adam(models[0].parameters(), lr=learning_rate))
#optimizers.append(optim.Adam(models[1].parameters(), lr=learning_rate))  #当有两种元素时，以此类推
resume=False  # resume:恢复
if resume:  # 中断的时候恢复训练
    path0=r"./FC3model_minimize_Etot/3layers0type466.pt"
    path1=r"./FC3model_minimize_Etot/3layers1type466.pt"
    checkpoint0 = torch.load(path0)
    checkpoint1 = torch.load(path1)
    models[0].load_state_dict(checkpoint0['model'])
    models[1].load_state_dict(checkpoint0['model'])
    optimizer.load_state_dict(checkpoint0['optimizer'])
    start_epoch=checkpoint0['epoch']+1
'''

# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)

# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=weight_decay)
scheduler = optimizer
start = time.time()

min_loss = np.inf
start_epoch=1

if pm.flag_plt:
    fig, ax=plt.subplots()
    line_train,=ax.plot([],[], label='train_RMSE_loss')
    line_test,=ax.plot([],[], label='test_MAE')
    ax.set_yscale('log')
    ax.legend()
    #plt.show(block=False)

for epoch in range(start_epoch, n_epoch + 1):
    print("epoch " + str(epoch))
    # start = time.time()
    if epoch > weight_decay_epoch:   # 学习率衰减
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    
    train_epoch_loss = 0
    j = 0
    for i_batch, sample_batches in enumerate(loader_train):
        train_error = train(sample_batches, models, optimizer, nn.MSELoss())   #预训练
        # train_error = train_finetuning(sample_batches, models, optimizer, nn.MSELoss())
        # Log train/loss to TensorBoard at every iteration
        n_iter = (epoch - 1) * len(loader_train) + i_batch + 1
        writer.add_scalar('Train/loss', train_error, n_iter)
        train_epoch_loss += train_error
    train_loss = train_epoch_loss/len(loader_train)
    end = time.time()
    time_cost = sec_to_hms(int(end-start))    #到当前epoch累计时间
    print('epoch = {}, training loss = {:.8f}, lr = {}, time cost = {}'.format(epoch, train_loss, lr, time_cost))

    val_epoch_loss = 0
    with torch.no_grad():
        for i_batch, sample_batches in enumerate(loader_test):
            val_error = valid(sample_batches, models, nn.MSELoss())
            n_iter = (epoch - 1) * len(loader_test) + i_batch + 1
            writer.add_scalar('Val/loss', val_error, n_iter)
            val_epoch_loss += val_error
    val_loss = val_epoch_loss/len(loader_test)
    print('validation loss = {:.8f}'.format(val_loss))   

    iprint = 1 #隔几个epoch记录一次误差
    f_err_log=pm.dir_work+'out_err.dat'
    if epoch // iprint == 1:
        fid_err_log = open(f_err_log, 'w')
    else:
        fid_err_log = open(f_err_log, 'a')
    fid_err_log.write('%d %e %e %s %e\n' % (epoch, train_loss, lr, time_cost, val_loss))
    fid_err_log.close()
  
    if pm.flag_plt:
        line_train.set_xdata(np.append(line_train.get_xdata(),epoch))
        line_train.set_ydata(np.append(line_train.get_ydata(),train_loss))
        line_test.set_xdata(np.append(line_test.get_xdata(),epoch))
        line_test.set_ydata(np.append(line_test.get_ydata(),val_loss))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    # direc = '../FC3model/'
    if val_loss < min_loss:
        min_loss = val_loss
        for i in range(len(pm.atomType)):
            name = direc + '/3layers_'+ str(i) + 'type_' + str(epoch)+'.pt'
            state = {'model': models[i].state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
            torch.save(state, name)
            print('saving model to {}'.format(name))
writer.close()
