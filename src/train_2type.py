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

torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
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
    if torch.cuda.is_available():
        out_atoms_energy = out_atoms_energy.cuda()
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
    Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))   #[40,108,3]
    Neighbor = Variable(sample_batches['input_nblist'][:,:,:].to(device))  # [40,108,100]
    dfeat=Variable(sample_batches['input_dfeat'][:,:,:,:,:].float().to(device))  #[40,108,100,42,3]
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1] 
    len_batch = sample_batches["input_feat"].shape[0]
    # out_atoms_energy = torch.zeros(len_batch, 1)
    for i in range(pm.natoms):
        input_data = Variable(sample_batches['input_feat'][:,i,:].float().to(device),requires_grad=True)
        label = Variable(sample_batches['output_energy'][:,i,:].float().to(device))
        model.to(device)
        model.train()
        x, out = model(input_data)   # out [40,1]
        #===================每个Ei对输入的导数====================
        out_sum = out.mean()
        out_sum.backward(retain_graph=True)
        input_grad = input_data.grad  #input_data.grad.shape --> torch.size([40,42])
        if(i==0):
            input_grad_allatoms=torch.unsqueeze(input_grad,1)
        else: 
            input_grad_allatoms=torch.cat([input_grad_allatoms, torch.unsqueeze(input_grad,1)], dim=1)  #[40,108,42]
        #===================每个Ei的预测值====================
        if(i==0):
            out_atoms_energy = out     #[40,1]
        else:
            out_atoms_energy = torch.cat([out_atoms_energy, out], dim=1)    # [40,108]
    out_atoms_energy = torch.sum(out_atoms_energy, dim=1)    # [40,1]
    Etot_deviation = out_atoms_energy - Etot_label     # [40,1]
    Etot_shape = Etot_label.shape[0]  #40
    Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)
    #===================每个Fi的预测值====================
    for i in range(pm.natoms):
        force=Variable(sample_batches['output_force'][:,i,:].float().to(device))
        neighbor = Variable(sample_batches['input_nblist'][:,i,:].to(device))    #[40,100]
        neighbor_number = len(neighbor[2])
        force_dx = torch.zeros([1, input_grad_allatoms.shape[2]])
        force_dy = torch.zeros([1, input_grad_allatoms.shape[2]])
        force_dz = torch.zeros([1, input_grad_allatoms.shape[2]])
        if torch.cuda.is_available():
            force_dx = force_dx.cuda()
            force_dy = force_dy.cuda()
            force_dz = force_dz.cuda()
        for batch in range(pm.batch_size):
            for nei in range(neighbor_number):
                nei_index = int(neighbor[batch,nei])
                if(nei_index==0):
                    break
                sub_force_dx=input_grad_allatoms[batch,nei_index-1,:]*dfeat[batch,i,nei,:,0]  #[1,42] * [1,42]
                sub_force_dy=input_grad_allatoms[batch,nei_index-1,:]*dfeat[batch,i,nei,:,1]
                sub_force_dz=input_grad_allatoms[batch,nei_index-1,:]*dfeat[batch,i,nei,:,2]
                force_dx = force_dx + sub_force_dx       #[1,42]
                force_dy = force_dy + sub_force_dy
                force_dz = force_dz + sub_force_dz
                # import ipdb;ipdb.set_trace()
                force_dx_reduced = torch.sum(force_dx, dim=1)  #[1,42]->[1]
                force_dy_reduced = torch.sum(force_dy, dim=1)
                force_dz_reduced = torch.sum(force_dz, dim=1)
                atomi_neighbori_force = torch.tensor([force_dx_reduced, force_dy_reduced, force_dz_reduced])  #[3]  Ei的一个neighbor的力
                atomi_neighbori_force = atomi_neighbori_force.unsqueeze(0)   #[1,3]
                if torch.cuda.is_available():
                    atomi_neighbori_force = atomi_neighbori_force.cuda()
                if(nei==0):
                    atomi_neighbors_force=atomi_neighbori_force
                else:
                    atomi_neighbors_force=torch.cat((atomi_neighbors_force, atomi_neighbori_force), dim=0)   #[100,3] Ei的多个neighbor的力
            atomi_neighbors_force=torch.sum(atomi_neighbors_force, dim=0)   #[3]
            if(batch==0):
                Force_atomi_predict=atomi_neighbors_force.unsqueeze(0)   #[1,3]
            else:
                Force_atomi_predict=torch.cat((Force_atomi_predict, atomi_neighbors_force.unsqueeze(0)), dim=0)  #[40,3]不同的batch的一个中心原子的force
        if(i==0):
            Force_predict =  Force_atomi_predict.unsqueeze(1)  #[40,1,3]
        else:
            Force_predict = torch.cat((Force_predict, Force_atomi_predict.unsqueeze(1)), dim=1)   #[40,108,3]
    Force_deviation = Force_predict - Force_label
    Force_shape = Force_deviation.shape[0] * Force_deviation.shape[1] * Force_deviation.shape[2]   #40*108*3
    Force_ABS_error = Force_deviation.norm(1) / Force_shape
    Force_RMSE_error = math.sqrt(1/Force_shape) * Force_deviation.norm(2)
    #===================反向传播更新参数====================
    optimizer.zero_grad()
    loss=Force_RMSE_error+Etot_RMSE_error
    loss.backward()
    optimizer.step()
    error = error+float(loss.item())
    return error

    '''
    out_image = torch.sum(out_image, dim=1)   #[40]
    # out = torch.unsqueeze(out_image, 1)     #[40,1]
    out_sum = out_image.mean()
    out_sum.backward(retain_graph=True)
    
    
        
        # 加入力和Etot，重新计算此处的loss
        for batch_index in range(len_batch):
            atomi_dx, atomi_dy, atomi_dz = [0,0,0]
            for input_index in range(42):   #一个原子的受力
                atomi_dx = atomi_dx + input_grad[batch_index, input_index]*dfeat[batch_index,:,input_index,0].sum()
                atomi_dy = atomi_dy + input_grad[batch_index, input_index]*dfeat[batch_index,:,input_index,1].sum()
                atomi_dz = atomi_dz + input_grad[batch_index, input_index]*dfeat[batch_index,:,input_index,2].sum()
            atomi_force = torch.tensor([atomi_dx, atomi_dy, atomi_dz])  #反向计算出一个原子的力
            if torch.cuda.is_available():
                atomi_force = atomi_force.cuda()
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
    
    
'''

# ==========================part3:模型训练==========================
n_epoch = 2000
learning_rate = 0.1
weight_decay = 0.9
weight_decay_epoch = 50
direc = './FC3model'
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
    path=r"./FC3model"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch=checkpoint['epoch']+1

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
        train_error = train_finetuning(sample_batches, model, optimizer, nn.MSELoss())
        # train_error = train(sample_batches, model, optimizer, nn.MSELoss())
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
            val_error = valid(sample_batches, model, nn.MSELoss())
            n_iter = (epoch - 1) * len(loader_test) + i_batch + 1
            writer.add_scalar('Val/loss', val_error, n_iter)
            val_epoch_loss += val_error
    val_loss = val_epoch_loss/len(loader_test)
    print('validation loss = {:.8f}'.format(val_loss))   
    
    iprint = 1 #隔几个epoch记录一次误差
    f_err_log=pm.dir_work+'out_err_for.dat'
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
        # name = direc + '/3layers'+str(epoch)+'.pt'
        name = direc + '/3layers'+'.pt'
        print('saving model to {}'.format(name))
        state = {'model': model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, name)
writer.close()
