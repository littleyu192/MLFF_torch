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

# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()
from torchsummary import summary

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
torch_train_data = get_torch_data(108, train_data_path, train_data_file_frompwmat)
test_data_path='../data/train_data/final_test'
test_data_file_frompwmat = '/test_data.csv'
torch_test_data = get_torch_data(108, test_data_path, test_data_file_frompwmat)

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
    input_data = Variable(sample_batches['input_feat'].float().to(device))
    label = Variable(sample_batches['output_energy'].float().to(device))
    model.to(device)
    model.train()
    x, out = model(input_data)
    optimizer.zero_grad()
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
    error = float(loss.item())
    return error

def valid(sample_batches, model, criterion):
    input_data = Variable(sample_batches['input_feat'].float().to(device))
    label = Variable(sample_batches['output_energy'].float().to(device))
    model.to(device)
    model.eval()
    x, out = model(input_data)
    loss = criterion(out, label)
    error = float(loss.item())
    return error

def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


# ==========================part3:模型训练==========================
n_epoch = 2000
learning_rate = 0.01
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
    if epoch > weight_decay_epoch:   # 学习率衰减
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    
    train_epoch_loss = 0
    j = 0
    for i_batch, sample_batches in enumerate(loader_train):
        train_error = train(sample_batches, model, optimizer, nn.MSELoss())
        # Log train/loss to TensorBoard at every iteration
        n_iter = (epoch - 1) * len(loader_train) + i_batch + 1
        # writer.add_scalar('Train/loss', train_error, n_iter)
        train_epoch_loss += train_error
    train_loss = train_epoch_loss//len(loader_train)
    end = time.time()
    time_cost = sec_to_hms(int(end-start))
    print('epoch = {}, training loss = {}, lr = {}, time cost = {}'.format(epoch, train_loss, lr, time_cost))

    val_epoch_loss = 0
    with torch.no_grad():
        for i_batch, sample_batches in enumerate(loader_test):
            val_error = valid(sample_batches, model, nn.MSELoss())
            n_iter = (epoch - 1) * len(loader_test) + i_batch + 1
            # writer.add_scalar('Val/loss', val_error, n_iter)
            val_epoch_loss += val_error
    val_loss = val_epoch_loss//len(loader_test)
    print('validation loss = {}'.format(val_loss))   
    
    if val_loss < min_loss:
        min_loss = val_loss
        name = direc + '3layers'+str(epoch)+'.pt'
        print('saving model to {}'.format(name))
        state = {'model': model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, name)
# writer.close()