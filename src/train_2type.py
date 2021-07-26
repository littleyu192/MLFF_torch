# -*- coding: utf-8 -*-
import torch
import numbers
import os,sys
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
def train(sample_batches, model, optimizer, criterion):
    error=0
    for i in range(pm.natoms):
        input_data = Variable(sample_batches['input_feat'][:,i,:].float().to(device),requires_grad=True)
        label = Variable(sample_batches['output_energy'][:,i,:].float().to(device))
        model.to(device)
        model.train()
        x, out = model(input_data)
        optimizer.zero_grad()
        loss = criterion(out, label)
        loss.backward()
        # import ipdb;ipdb.set_trace()
        optimizer.step()
        error = error+float(loss.item())
    return error

def valid(sample_batches, model, criterion):
    error=0
    len_batch = sample_batches["input_feat"].shape[0]
    out_atoms_energy = torch.zeros(len_batch, 1)
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
    Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))
    Etot_label = torch.sum(Etot_label, dim=1)
    Etot_predict = 0
    for i in range(pm.natoms):
        input_data = Variable(sample_batches['input_feat'][:,i,:].float().to(device))
        label = Variable(sample_batches['output_energy'][:,i,:].float().to(device))
        model.to(device)
        model.eval()
        x, out = model(input_data)
        out_atoms_energy = out_atoms_energy + out
 
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

def train_finetuning(sample_batches, model, optimizer, criterion):
    error=0
    # backward_force=torch.tensor
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
    Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1]
    len_batch = sample_batches["input_feat"].shape[0]
    out_atoms_energy = torch.zeros(len_batch, 1)
    for i in range(pm.natoms):
        input_data = Variable(sample_batches['input_feat'][:,i,:].float().to(device),requires_grad=True)
        label = Variable(sample_batches['output_energy'][:,i,:].float().to(device))
        dfeat=Variable(sample_batches['input_dfeat'][:,i,:,:,:].float().to(device))
        force=Variable(sample_batches['output_force'][:,i,:].float().to(device))
        model.to(device)
        model.train()
        x, out = model(input_data)
        out_sum = out.sum()
        out_sum.backward(retain_graph=True)
        # gradients=torch.from_numpy(np.ones((40, 1)).float().to(device))
        # out.backward(gradient=None)
        input_grad = input_data.grad  #input_data.grad.shape --> torch.size([40,42])

        # 加入力和Etot，重新计算此处的loss
        for batch_index in range(len_batch):
            atomi_dx, atomi_dy, atomi_dz = [0,0,0]
            for input_index in range(42):   #一个原子的受力
                atomi_dx = atomi_dx + input_grad[batch_index, input_index]*dfeat[batch_index,:,input_index,0].sum() 
                atomi_dy = atomi_dy + input_grad[batch_index, input_index]*dfeat[batch_index,:,input_index,1].sum()
                atomi_dz = atomi_dz + input_grad[batch_index, input_index]*dfeat[batch_index,:,input_index,2].sum()
            atomi_force = torch.tensor([atomi_dx, atomi_dy, atomi_dz])  #反向计算出一个原子的力
            # force_deviation = force[i,0,:] - atomi_force   #和labelF的偏差
            # force_ABS_error = force.norm(1) / 3
            # force_RMSE_error = math.sqrt(1/3) * force.norm(2)
            # print(force_RMSE_error)
            atomi_force = atomi_force.unsqueeze(0)
            # print(atomi_force)
            if(batch_index==0):
                batchi_atomi_force=atomi_force         #初始化atoms_force
            else:
                batchi_atomi_force = torch.cat((batchi_atomi_force, atomi_force), dim=0)  #(40,3)
        # print(batchi_atomi_force.shape)
        if(i==0):
            Force_predict=batchi_atomi_force.unsqueeze(1)
        else:
            Force_predict=torch.cat((Force_predict, batchi_atomi_force.unsqueeze(1)), dim=1)
        out_atoms_energy = out_atoms_energy + out

    Force_deviation = Force_predict - Force_label  
    Etot_deviation = out_atoms_energy - Etot_label

    Force_shape = Force_deviation.shape[0] * Force_deviation.shape[1] * Force_deviation.shape[2] #40*108*3
    Force_ABS_error = Force_deviation.norm(1) / Force_shape
    Force_RMSE_error = math.sqrt(1/Force_shape) * Force_deviation.norm(2)
    Etot_shape = Etot_label.shape[0]  #40
    Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)
    
    optimizer.zero_grad()
    loss=Force_RMSE_error+Etot_RMSE_error
    loss.backward()
    optimizer.step()
    error = error+float(loss.item())
    return error


# ==========================part3:模型训练==========================
n_epoch = 2000
learning_rate = 0.1
weight_decay = 0.9
weight_decay_epoch = 50
direc = '../FC3model'
if not os.path.exists(direc):
    os.makedirs(direc)


model=FCNet()   # BN=False, Dropout=False
# model=FCNet(BN=True)
# model=FCNet(Dropout=True)


# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=weight_decay)
scheduler = optimizer
start = time.time()

min_loss = np.inf
start_epoch=1

resume=False   # resume:恢复
if resume:  # 中断的时候恢复训练
    path=r"../FC3model"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch=checkpoint['epoch']+1

for epoch in range(start_epoch, n_epoch + 1):
    print("epoch " + str(epoch))
    if epoch > weight_decay_epoch:   # 学习率衰减
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    
    train_epoch_loss = 0
    j = 0
    for i_batch, sample_batches in enumerate(loader_train):
        train_error = train_finetuning(sample_batches, model, optimizer, nn.MSELoss())
        # train_error = train(sample_batches, model, optimizer, nn.MSELoss())
        # Log train/loss to TensorBoard at every iteration
        n_iter = (epoch - 1) * len(loader_train) + i_batch + 1
        writer.add_scalar('Train/loss', train_error, n_iter)
        train_epoch_loss += train_error
    train_loss = train_epoch_loss/len(loader_train)
    end = time.time()
    time_cost = sec_to_hms(int(end-start))
    print('epoch = {}, training loss = {:.8f}, lr = {}, time cost = {}'.format(epoch, train_loss, lr, time_cost))

    val_epoch_loss = 0
    with torch.no_grad():
        for i_batch, sample_batches in enumerate(loader_test):
            val_error = valid(sample_batches, model, nn.MSELoss())
            n_iter = (epoch - 1) * len(loader_test) + i_batch + 1
            writer.add_scalar('Val/loss', val_error, n_iter)
            val_epoch_loss += val_error
    val_loss = val_epoch_loss/len(loader_test)
    print('validation loss = {:.8f}'.format(val_loss))   
    
    # direc = '../FC3model/'
    if val_loss < min_loss:
        min_loss = val_loss
        # name = direc + '/3layers'+str(epoch)+'.pt'
        name = direc + '/3layers'+'.pt'
        print('saving model to {}'.format(name))
        state = {'model': model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, name)
writer.close()