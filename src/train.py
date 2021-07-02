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
import torchvision
from torchvision import transforms
import torch.utils.data as Data

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


sys.path.append("../data")
# dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data_process_path=os.path.join(dir_mytest+'\data')
# print(sys.path.insert(0,data_process_path))
# print(dir_mytest)
from data_loader import MovementDataset

from FC3 import simpleNet, FC3Net, FC3BN_Net  #import写好的module或类

torch.manual_seed(2018)
torch.cuda.manual_seed(2018)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_file='../data/train_data.csv'
valid_data_file='../data/valid_data.csv'
train_data = MovementDataset(train_data_file)
valid_data = MovementDataset(valid_data_file)
batch_size = 40   #40
loader_train = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True) #是否打乱顺序，多线程读数据num_workers=4
loader_valid = Data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

# for i, value in enumerate(loader_train):
#     print(i)
#     print(value['input'].shape)   #  torch.size([batchsize, 324])
#     print(value['output_Fi'].shape)
#     print(value['output_Etot'].shape)
#     print(len(loader_train))
#     import ipdb;ipdb.set_trace()

def train(sample_batches, model, optimizer, criterion):
    input_data = Variable(sample_batches['input'].float().to(device))
    label = Variable(sample_batches['output_Etot'].float().to(device))
    model.to(device)
    model.train()
    out = model(input_data)
    optimizer.zero_grad()
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
    error = float(loss.item())
    return error

def valid(sample_batches, model, criterion):
    input_data = Variable(sample_batches['input'].float().to(device))
    label = Variable(sample_batches['output_Etot'].float().to(device))
    model.to(device)
    model.eval()
    out = model(input_data)
    loss = criterion(out, label)
    error = float(loss.item())
    return error

def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

input_dim = 324
hidden_size = [256, 128]
output_dim = 1     #1  Etot ; 324  Fi
n_epoch = 2000
learning_rate = 0.01
weight_decay = 0.9
weight_decay_epoch = 50
direc = '../FC3model'
if not os.path.exists(direc):
    os.makedirs(direc)

model=FC3Net(in_dim=input_dim,
             n_hidden_1=hidden_size[0],
             n_hidden_2=hidden_size[1],
             out_dim=output_dim)
if torch.cuda.device_count() > 1:
    air = nn.DataParallel(model)
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
    if epoch > weight_decay_epoch:   # 学习率衰减
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    
    train_epoch_loss = 0
    j = 0
    for i_batch, sample_batches in enumerate(loader_train):
        train_error = train(sample_batches, model, optimizer, nn.MSELoss())
        # Log train/loss to TensorBoard at every iteration
        n_iter = (epoch - 1) * len(loader_train) + i_batch + 1
        writer.add_scalar('Train/loss', train_error, n_iter)
        train_epoch_loss += train_error
    train_loss = train_epoch_loss//len(loader_train)
    end = time.time()
    time_cost = sec_to_hms(int(end-start))
    print('epoch = {}, training loss = {}, lr = {}, time cost = {}'.format(epoch, train_loss, lr, time_cost))

    val_epoch_loss = 0
    with torch.no_grad():
        for i_batch, sample_batches in enumerate(loader_valid):
            val_error = valid(sample_batches, model, nn.MSELoss())
            n_iter = (epoch - 1) * len(loader_valid) + i_batch + 1
            writer.add_scalar('Val/loss', val_error, n_iter)
            val_epoch_loss += val_error
    val_loss = val_epoch_loss//len(loader_valid)
    print('validation loss = {}'.format(val_loss))   
    
    if val_loss < min_loss:
        min_loss = val_loss
        name = direc + '3layers'+str(epoch)+'.pt'
        print('saving model to {}'.format(name))
        state = {'model': model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, name)
writer.close()
