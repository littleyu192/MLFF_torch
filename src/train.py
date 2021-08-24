#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os,sys
import random
import time
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from model.FC import MLFFNet
import torch.utils.data as Data
from torch.autograd import Variable
import math
sys.path.append(os.getcwd())
import parameters as pm 
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/pre_data')
from data_loader_2type import MovementDataset, get_torch_data
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
writer = SummaryWriter()
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(sample_batches, model, optimizer, criterion):
    error=0
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
    atom_number = Etot_label.shape[1]
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1]
    Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))   #[40,108,3]
    Egroup_label = Variable(sample_batches['input_egroup'].float().to(device))

    input_data = Variable(sample_batches['input_feat'].float().to(device), requires_grad=True)
    neighbor = Variable(sample_batches['input_nblist'].int().to(device))  # [40,108,100]
    dfeat = Variable(sample_batches['input_dfeat'].float().to(device))  #[40,108,100,42,3]
    egroup_weight = Variable(sample_batches['input_egroup_weight'].float().to(device))
    ind_img = Variable(sample_batches['ind_image'].int().to(device))
    divider = Variable(sample_batches['input_divider'].float().to(device))
    model = model.to(device)
    model.train()
    force_predict, Etot_predict, Ei_predict = model(input_data, dfeat, neighbor)
    Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider)   #[40,108,1]
    
    Etot_deviation = Etot_predict - Etot_label     # [40,1]
    Etot_square_deviation = Etot_deviation ** 2
    Etot_shape = Etot_label.shape[0]  #40
    Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)
    Etot_L2 = (1/Etot_shape) * Etot_square_deviation.sum()   #L2-->tf.reduce_mean(tf.square())
    Ei_L2 = Etot_L2 / atom_number

    Force_deviation = force_predict - Force_label
    Force_square_deviation = Force_deviation ** 2
    Force_shape = Force_deviation.shape[0] * Force_deviation.shape[1] * Force_deviation.shape[2]   #40*108*3
    Force_ABS_error = Force_deviation.norm(1) / Force_shape
    Force_RMSE_error = math.sqrt(1/Force_shape) * Force_deviation.norm(2)
    Force_L2 = (1/Force_shape) * Force_square_deviation.sum()

    Egroup_deviation = Egroup_predict - Egroup_label
    Egroup_square_deviation = Egroup_deviation ** 2
    Egroup_shape = Egroup_label.shape[0] * Egroup_label.shape[1]
    Egroup_ABS_error = Egroup_deviation.norm(1) / Egroup_shape
    Egroup_RMSE_error = math.sqrt(1/Egroup_shape) * Egroup_deviation.norm(2)
    Egroup_L2 = (1/Egroup_shape) * Egroup_square_deviation.sum()
    
    optimizer.zero_grad()
    # w_e = torch.sum(Force_square_deviation) / (torch.sum(Force_square_deviation)+torch.sum(Etot_square_deviation))
    # w_f = 1 - w_e
    # loss = w_f * torch.sum(Force_square_deviation) + w_e * torch.sum(Etot_square_deviation)
    # loss = torch.sum(Force_square_deviation) + torch.sum(Etot_square_deviation)
    # loss = torch.sum(Etot_square_deviation)   #30个epoch，etot就会变成0.*
    # loss = error / Etot_shape
    # loss = 0.8 * torch.sum(Egroup_square_deviation) + 0.2 * torch.sum(Force_square_deviation)
    force_square_loss = torch.sum(Force_square_deviation) / Force_shape
    etot_square_loss = torch.sum(Etot_square_deviation) / Etot_shape
    egroup_square_loss = torch.sum(Egroup_square_deviation) / Egroup_shape
    loss = pm.rtLossF * force_square_loss + pm.rtLossEtot * etot_square_loss + pm.rtLossE * egroup_square_loss

    loss.backward()
    optimizer.step()
    error = error+float(loss.item())

    return loss, force_square_loss, etot_square_loss, egroup_square_loss, \
        Force_RMSE_error, Force_ABS_error, Etot_RMSE_error, Etot_ABS_error, Egroup_RMSE_error, Egroup_ABS_error, \
            Etot_L2, Ei_L2, Force_L2, Egroup_L2

def valid(sample_batches, model, criterion):
    error=0
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
    atom_number = Etot_label.shape[1]
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1]
    Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))   #[40,108,3]
    Egroup_label = Variable(sample_batches['input_egroup'].float().to(device))

    input_data = Variable(sample_batches['input_feat'].float().to(device), requires_grad=True)
    neighbor = Variable(sample_batches['input_nblist'].int().to(device))  # [40,108,100]
    dfeat=Variable(sample_batches['input_dfeat'].float().to(device))  #[40,108,100,42,3]
    egroup_weight = Variable(sample_batches['input_egroup_weight'].float().to(device))
    ind_img = Variable(sample_batches['ind_image'].int().to(device))
    divider = Variable(sample_batches['input_divider'].float().to(device))
    # label = Variable(sample_batches['output_energy'].float().to(device))
    model.to(device)
    model.train()
    force_predict, Etot_predict, Ei_predict = model(input_data, dfeat, neighbor)
    Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider)

    Etot_deviation = Etot_predict - Etot_label     # [40,1]
    Etot_square_deviation = Etot_deviation ** 2
    Etot_shape = Etot_label.shape[0]  #40
    Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)
    Etot_L2 = (1/Etot_shape) * Etot_square_deviation.sum()   #L2-->tf.reduce_mean(tf.square())
    Ei_L2 = Etot_L2 / atom_number

    Force_deviation = force_predict - Force_label
    Force_square_deviation = Force_deviation ** 2
    Force_shape = Force_deviation.shape[0] * Force_deviation.shape[1] * Force_deviation.shape[2]   #40*108*3
    Force_ABS_error = Force_deviation.norm(1) / Force_shape
    Force_RMSE_error = math.sqrt(1/Force_shape) * Force_deviation.norm(2)
    Force_L2 = (1/Force_shape) * Force_square_deviation.sum()

    Egroup_deviation = Egroup_predict - Egroup_label
    Egroup_square_deviation = Egroup_deviation ** 2
    Egroup_shape = Egroup_label.shape[0] * Egroup_label.shape[1]
    Egroup_ABS_error = Egroup_deviation.norm(1) / Egroup_shape
    Egroup_RMSE_error = math.sqrt(1/Egroup_shape) * Egroup_deviation.norm(2)
    Egroup_L2 = (1/Egroup_shape) * Egroup_square_deviation.sum()

    force_square_loss = torch.sum(Force_square_deviation) / Force_shape
    etot_square_loss = torch.sum(Etot_square_deviation) / Etot_shape
    egroup_square_loss = torch.sum(Egroup_square_deviation) / Egroup_shape
    # loss = torch.sum(Force_square_deviation) + torch.sum(Etot_square_deviation)
    # loss = torch.sum(Etot_square_deviation)
    # loss = error / Etot_shape
    # loss = 0.8 * torch.sum(Egroup_square_deviation) + 0.2 * torch.sum(Force_square_deviation)
    loss = pm.rtLossF * force_square_loss + pm.rtLossEtot * etot_square_loss + pm.rtLossE * egroup_square_loss
    error = error+float(loss.item())
    return error, force_square_loss, etot_square_loss, egroup_square_loss, \
        Force_RMSE_error, Force_ABS_error, Etot_RMSE_error, Etot_ABS_error, Egroup_RMSE_error, Egroup_ABS_error,\
             Etot_L2, Ei_L2, Force_L2, Egroup_L2

def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


# ==========================part1:数据读取==========================
batch_size = pm.batch_size   #40
train_data_path=pm.train_data_path
torch_train_data = get_torch_data(pm.natoms, train_data_path)
loader_train = Data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=True)

valid_data_path=pm.test_data_path
torch_valid_data = get_torch_data(pm.natoms, valid_data_path)
loader_valid = Data.DataLoader(torch_valid_data, batch_size=1, shuffle=True)

# ==========================part2:指定模型参数==========================
n_epoch = 2000
learning_rate = 0.1
weight_decay = 0.9
weight_decay_epoch = 10
direc = './FC3model_minimize_8Eg_2force_tanh_dropout3'
if not os.path.exists(direc):
    os.makedirs(direc)
model = MLFFNet()
# if torch.cuda.device_count() > 1:
    # model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=weight_decay)
start = time.time()
min_loss = np.inf
start_epoch=1

patience = 50	# 当验证集损失在连续50次没有降低时，停止模型训练，防止模型过拟合

resume=False  # resume:恢复
if resume:    # 中断的时候恢复训练
    path=r"./FC3model_minimize_8Eg_2force_tanh/3layers_MLFFNet_8epoch.pt"
    checkpoint = torch.load(path, map_location={'cpu':'cuda:0'})
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # import ipdb; ipdb.set_trace()
    start_epoch=checkpoint['epoch']+1
# import ipdb; ipdb.set_trace()


# ==========================part3:模型训练==========================
if pm.flag_plt:
    fig, ax=plt.subplots()
    line_train_force,=ax.plot([],[], label='train_RMSE_force_loss')
    line_train_etot,=ax.plot([],[], label='train_RMSE_etot_loss')
    line_train_total,=ax.plot([],[], label='train_RMSE_total_loss')
    ax.set_yscale('log')
    ax.legend()
    #plt.show(block=False)

for epoch in range(start_epoch, n_epoch + 1):
    print("epoch " + str(epoch))
    start = time.time()
    
    lr = optimizer.param_groups[0]['lr']
    
    train_epoch_force_RMSE_loss = 0
    train_epoch_etot_RMSE_loss = 0
    train_epoch_egroup_RMSE_loss = 0
    loss_function_err = 0
    train_epoch_force_square_loss = 0
    train_epoch_etot_square_loss = 0
    train_epoch_egroup_square_loss = 0
    
    for i_batch, sample_batches in enumerate(loader_train):
        square_error, force_square_loss, etot_square_loss, egroup_square_loss, \
            Force_RMSE_error, Force_ABS_error, Etot_RMSE_error, Etot_ABS_error, Egroup_RMSE_error, Egroup_ABS_error, \
                Etot_L2, Ei_L2, Force_L2, Egroup_L2 = train(sample_batches, model, optimizer, nn.MSELoss())
        # Log train/loss to TensorBoard at every iteration
        n_iter = (epoch - 1) * len(loader_train) + i_batch + 1
        # writer.add_scalar('Train/loss', Force_RMSE_error, n_iter)
        loss_function_err += square_error
        train_epoch_force_square_loss += force_square_loss
        train_epoch_etot_square_loss += etot_square_loss
        train_epoch_egroup_square_loss += egroup_square_loss

        train_epoch_force_RMSE_loss += Force_RMSE_error
        train_epoch_etot_RMSE_loss += Etot_RMSE_error
        train_epoch_egroup_RMSE_loss += Egroup_RMSE_error
  
    train_function_err_avg = loss_function_err / len(loader_train)
    train_epoch_force_square_loss = train_epoch_force_square_loss/len(loader_train)
    train_epoch_etot_square_loss = train_epoch_etot_square_loss/len(loader_train)
    train_epoch_egroup_square_loss = train_epoch_egroup_square_loss/len(loader_train)

    train_force_rmse_loss = train_epoch_force_RMSE_loss/len(loader_train)
    train_etot_rmse_loss = train_epoch_etot_RMSE_loss/len(loader_train)
    train_egroup_rmse_loss = train_epoch_egroup_RMSE_loss/len(loader_train)

    train_force_L2 = Force_L2/len(loader_train)
    train_etot_L2 = Etot_L2/len(loader_train)
    train_ei_L2 = Ei_L2/len(loader_train)
    train_egroup_L2 = Egroup_L2/len(loader_train)

    end = time.time()
    time_cost = sec_to_hms(int(end-start))    #每个epoch的训练时间
    print('epoch = {}, step = {}, train_function_err_avg = {:.8f}, train_epoch_force_square_loss = {:.8f}, \
        train_epoch_etot_square_loss = {:.8f}, train_epoch_egroup_square_loss = {:.8f}, \
          training force L2 = {:.8f}, training ei L2 = {:.8f}, training etot L2 = {:.8f}, training egroup L2 = {:.8f}, \
            lr = {}, time cost = {}'.format(epoch, n_iter, train_function_err_avg, train_epoch_force_square_loss, \
            train_epoch_etot_square_loss, train_epoch_egroup_square_loss, \
                train_force_L2, train_ei_L2, train_etot_L2, train_egroup_L2, lr, time_cost)) 

    valid_loss_function_err = 0
    valid_epoch_force_square_loss = 0
    valid_epoch_etot_square_loss = 0
    valid_epoch_egroup_square_loss = 0

    valid_epoch_force_RMSE_loss = 0
    valid_epoch_etot_RMSE_loss = 0
    valid_epoch_egroup_RMSE_loss = 0
    
    for i_batch, sample_batches in enumerate(loader_valid):
        square_error, force_square_loss, etot_square_loss, egroup_square_loss, \
            Force_RMSE_error, Force_ABS_error, Etot_RMSE_error, Etot_ABS_error, Egroup_RMSE_error, Egroup_ABS_error, \
                 Etot_L2, Ei_L2, Force_L2, Egroup_L2 = valid(sample_batches, model, nn.MSELoss())
        n_iter = (epoch - 1) * len(loader_valid) + i_batch + 1
        # writer.add_scalar('Val/loss', square_error, n_iter)
        valid_loss_function_err += square_error
        valid_epoch_force_square_loss += force_square_loss
        valid_epoch_etot_square_loss += etot_square_loss
        valid_epoch_egroup_square_loss += egroup_square_loss 

        valid_epoch_force_RMSE_loss += Force_RMSE_error
        valid_epoch_etot_RMSE_loss += Etot_RMSE_error
        valid_epoch_egroup_RMSE_loss += Egroup_RMSE_error

    valid_loss_function_err = valid_loss_function_err/len(loader_valid)
    valid_epoch_force_square_loss = valid_epoch_force_square_loss/len(loader_valid)
    valid_epoch_etot_square_loss = valid_epoch_etot_square_loss/len(loader_valid)
    valid_epoch_egroup_square_loss = valid_epoch_egroup_square_loss/len(loader_valid)

    valid_force_rmse_loss = valid_epoch_force_RMSE_loss/len(loader_valid)
    valid_etot_rmse_loss = valid_epoch_etot_RMSE_loss/len(loader_valid)
    valid_egroup_rmse_loss = valid_epoch_egroup_RMSE_loss/len(loader_valid)

    valid_force_L2 = Force_L2/len(loader_valid)
    valid_etot_L2 = Etot_L2/len(loader_valid)
    valid_ei_L2 = Ei_L2/len(loader_valid)
    valid_egroup_L2 = Egroup_L2/len(loader_valid)
    # valid_total_loss = valid_force_rmse_loss + valid_etot_rmse_loss
    print('valid_loss_function_err = {:.8f}, valid_epoch_force_square_loss = {:.8f}, \
        valid_epoch_etot_square_loss = {:.8f}, valid_epoch_egroup_square_loss = {:.8f}, \
         valid force L2 = {:.8f}, valid ei L2 = {:.8f}, valid etot L2 = {:.8f}, valid egroup L2 = {:.8f}'\
             .format(valid_loss_function_err, valid_epoch_force_square_loss, valid_epoch_etot_square_loss, \
             valid_epoch_egroup_square_loss, valid_force_L2, valid_ei_L2, valid_etot_L2, valid_egroup_L2))
    
    if epoch > weight_decay_epoch:   # 学习率衰减
        scheduler.step()
    iprint = 1 #隔几个epoch记录一次误差
    f_err_log=pm.dir_work+'out_miminize_8eg_2force_tanh_dropout3.dat'
    if epoch // iprint == 1:
        fid_err_log = open(f_err_log, 'w')
    else:
        fid_err_log = open(f_err_log, 'a')
    fid_err_log.write('%d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %s %e %e %e %e %e %e %e %e\n'       \
    % (epoch, train_function_err_avg, train_epoch_force_square_loss, train_epoch_etot_square_loss, train_epoch_egroup_square_loss, \
        train_force_rmse_loss, train_etot_rmse_loss, train_egroup_rmse_loss, lr, \
        valid_loss_function_err, valid_epoch_force_square_loss, valid_epoch_etot_square_loss, valid_epoch_egroup_square_loss,\
            valid_force_rmse_loss, valid_etot_rmse_loss, valid_egroup_rmse_loss, time_cost,   \
            train_force_L2, train_ei_L2, train_etot_L2, train_egroup_L2,   \
                valid_force_L2, valid_ei_L2, valid_etot_L2, valid_egroup_L2))
    fid_err_log.close()
  
    if pm.flag_plt:
        line_train_force.set_xdata(np.append(line_train_force.get_xdata(),epoch))
        line_train_force.set_ydata(np.append(line_train_force.get_ydata(),train_force_rmse_loss))
        line_train_etot.set_xdata(np.append(line_train_etot.get_xdata(),epoch))
        line_train_etot.set_ydata(np.append(line_train_etot.get_ydata(),train_force_rmse_loss))
        line_train_total.set_xdata(np.append(line_train_total.get_xdata(),epoch))
        line_train_total.set_ydata(np.append(line_train_total.get_ydata(),train_function_err_avg))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    if valid_loss_function_err < min_loss:
        min_loss = valid_loss_function_err
        works_epoch = 0
        name = direc + '/3layers_' + 'MLFFNet_' + str(epoch)+'epoch.pt'
        # state = {'model': model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
        state = {'model': model.state_dict(), 'epoch': epoch}
        torch.save(state, name)
        print('saving model to {}'.format(name))
    else:
        works_epoch += 1
        if works_epoch > patience:
            name = direc + '/3layers_' + 'MLFFNet.pt'
            state = {'model': model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
            torch.save(state, name)
            print("Early stopping")
            break
# writer.close()
