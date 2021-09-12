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
from torch.nn.modules import loss
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from model.FC import preMLFFNet, MLFFNet
from model.LN import LNNet
from model.MLFF_dmirror import MLFF_dmirror
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
#import torchviz



writer = SummaryWriter()
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pretrain(sample_batches, premodel, optimizer, criterion):
    error=0
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1]
    print("==========Etot label==========")
    print(Etot_label[0])
    input_data = Variable(sample_batches['input_feat'].float().to(device), requires_grad=True)
    neighbor = Variable(sample_batches['input_nblist'].int().to(device))  # [40,108,100]
    dfeat = Variable(sample_batches['input_dfeat'].float().to(device))  #[40,108,100,42,3]
   
    optimizer.zero_grad()
    model = premodel.to(device)
    model.train()
    Etot_predict, Ei_predict = model(input_data, dfeat, neighbor)
    print("==========Etot predict==========")
    print(Etot_predict[0])
    Etot_deviation = Etot_predict - Etot_label     # [40,1]
    Etot_square_deviation = Etot_deviation ** 2
    Etot_shape = Etot_label.shape[0]  #40
    Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)
    Etot_L2 = (1/Etot_shape) * Etot_square_deviation.sum()   #L2-->tf.reduce_mean(tf.square())
    
    etot_square_loss = torch.sum(Etot_square_deviation) / Etot_shape
    # ===========loss 选取etot==========
    # loss = etot_square_loss

    # ===========loss 选取torch.nn的函数==========
    loss = criterion(Etot_predict, Etot_label)

    loss.backward()
    optimizer.step()

    return etot_square_loss, Etot_RMSE_error, Etot_ABS_error, Etot_L2

def train(sample_batches, model, optimizer, criterion):
    error=0
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
    # print("==========Etot==========")
    # print(Etot_label)
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
    # import ipdb; ipdb.set_trace()
    # model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)
    model.train()
    # force_predict, Etot_predict, Ei_predict, Egroup_predict = model(input_data, dfeat, neighbor, egroup_weight, divider)
    Etot_predict, force_predict = model(input_data, dfeat, neighbor, egroup_weight, divider)
    # import ipdb; ipdb.set_trace()
    
    optimizer.zero_grad()
    # Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider)   #[40,108,1]
    # print("==========Etot predict==========")
    # print(Etot_predict)

    # Etot_deviation = Etot_predict - Etot_label     # [40,1]
    #print("etot predict: " + str(Etot_predict[0]) +  "etot label: " + str(Etot_label[0]))
    # Etot_square_deviation = Etot_deviation ** 2
    # Etot_shape = Etot_label.shape[0]  #40
    # Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    # Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)
    # Etot_L2 = (1/Etot_shape) * Etot_square_deviation.sum()   #L2-->tf.reduce_mean(tf.square())
    # Ei_L2 = Etot_L2 / atom_number

    # Force_deviation = force_predict - Force_label
    #print("force predict: " + str(force_predict[0,0]) +  "force label: " + str(Force_label[0,0]))
    # print(force_predict[0,0])
    # print("==========force label==========")
    # print(Force_label[0,0])
    # Force_square_deviation = Force_deviation ** 2
    # Force_shape = Force_deviation.shape[0] * Force_deviation.shape[1] * Force_deviation.shape[2]   #40*108*3
    # Force_ABS_error = Force_deviation.norm(1) / Force_shape
    # Force_RMSE_error = math.sqrt(1/Force_shape) * Force_deviation.norm(2)
    # Force_L2 = (1/Force_shape) * Force_square_deviation.sum()
    # import ipdb; ipdb.set_trace()

    # Egroup_deviation = Egroup_predict - Egroup_label
    # Egroup_square_deviation = Egroup_deviation ** 2
    # Egroup_shape = Egroup_label.shape[0] * Egroup_label.shape[1]
    # Egroup_ABS_error = Egroup_deviation.norm(1) / Egroup_shape
    # Egroup_RMSE_error = math.sqrt(1/Egroup_shape) * Egroup_deviation.norm(2)
    # Egroup_L2 = (1/Egroup_shape) * Egroup_square_deviation.sum()
    
    # force_square_loss = torch.sum(Force_square_deviation) / Force_shape
    # etot_square_loss = torch.sum(Etot_square_deviation) / Etot_shape
    # egroup_square_loss = torch.sum(Egroup_square_deviation) / Egroup_shape

    # ===========loss 只选 etot==========
    # loss = etot_square_loss

    # ===========loss 对etot和egroup配平==========
    # w_e = torch.sum(Egroup_square_loss) / (torch.sum(Egroup_square_loss)+torch.sum(Etot_square_loss))
    # w_eg = 1 - w_e
    # loss = w_eg * torch.sum(Egroup_square_loss) + w_e * torch.sum(Etot_square_loss)

    # ===========loss 选取linear的权重==========
    # loss =  pm.rtLossF * force_square_loss + pm.rtLossEtot * etot_square_loss + pm.rtLossE * egroup_square_loss
    
    # ===========loss 选取torch.nn的函数==========
    # loss = pm.rtLossF * criterion(force_predict, Force_label) + pm.rtLossEtot * criterion(Etot_predict, Etot_label) + pm.rtLossE * criterion(Egroup_predict, Egroup_label)
    loss = criterion(force_predict, Force_label) + criterion(Etot_predict, Etot_label)
    print("loss: " + str(loss))

    loss.backward()
    optimizer.step()
    # error = error + float(loss.item())
    # return loss, force_square_loss, etot_square_loss, egroup_square_loss, \
    #     Force_RMSE_error, Force_ABS_error, Force_L2, Etot_RMSE_error, Etot_ABS_error, Etot_L2, \
    #          Egroup_RMSE_error, Egroup_ABS_error, Egroup_L2

    # torchviz.make_dot(force_predict.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    # torchviz.make_dot(Etot_predict.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    return loss

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
    
    # loss = pm.rtLossF * force_square_loss + pm.rtLossEtot * etot_square_loss + pm.rtLossE * egroup_square_loss
    # ===========loss 选取torch.nn的函数==========
    loss = pm.rtLossF * criterion(force_predict, Force_label) + pm.rtLossEtot * criterion(Etot_predict, Etot_label) + pm.rtLossE * criterion(Egroup_predict, Egroup_L2)

    error = error+float(loss.item())
    return error, force_square_loss, etot_square_loss, egroup_square_loss, \
        Force_RMSE_error, Force_ABS_error, Force_L2, Etot_RMSE_error, Etot_ABS_error, Etot_L2, \
            Egroup_RMSE_error, Egroup_ABS_error, Egroup_L2

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
learning_rate = 0.8
weight_decay = 1.0
weight_decay_epoch = 100
direc = './FC3model_mini_force'
if not os.path.exists(direc):
    os.makedirs(direc) 

# if torch.cuda.device_count() > 1:
    # model = nn.DataParallel(model)

# ==========================part3:模型预训练==========================
if pm.isNNpretrain == True:
    premodel = preMLFFNet()           #预训练
    optimizer = optim.Adam(premodel.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=weight_decay)
    start = time.time()
    min_loss = np.inf
    start_epoch=1
    patience = 50	# 当验证集损失在连续50次没有降低时，停止模型训练，防止模型过拟合

    resume=False  # resume:恢复
    if resume:    # 中断的时候恢复训练
        path=r"./FC3model/3layers_MLFFNet_34epoch.pt"
        checkpoint = torch.load(path, map_location={'cpu':'cuda:0'})
        premodel.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint['epoch']+1
    # import ipdb; ipdb.set_trace()
    for epoch in range(start_epoch, n_epoch + 1):
        print("epoch " + str(epoch))
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        
        loss_function_err = 0
        train_epoch_etot_RMSE_loss = 0

        for i_batch, sample_batches in enumerate(loader_train):
            etot_square_loss, Etot_RMSE_error, Etot_ABS_error, Etot_L2 = pretrain(sample_batches, premodel, optimizer, nn.MSELoss())
            loss_function_err += etot_square_loss
            train_epoch_etot_RMSE_loss += Etot_RMSE_error
        train_function_err_avg = loss_function_err / len(loader_train)
        train_etot_rmse_loss = train_epoch_etot_RMSE_loss/len(loader_train)

        end = time.time()
        time_cost = sec_to_hms(int(end-start))    #每个epoch的训练时间
        print('Pretraining stage: epoch = {}, train_function_err_avg = {:.8f}, lr = {}, time cost = {}, \
            training etot rmse = {:.8f}'.format(epoch, train_function_err_avg,  \
                lr, time_cost, train_etot_rmse_loss)) 

        if epoch > weight_decay_epoch:   # 学习率衰减
            scheduler.step()
            print("ddddddddddd")
        iprint = 1               #隔几个epoch记录一次误差
        f_err_log=pm.dir_work+'pretraining.dat'
        if epoch // iprint == 1:
            fid_err_log = open(f_err_log, 'w')
        else:
            fid_err_log = open(f_err_log, 'a')
        fid_err_log.write('%d %e %e %e %s\n'       \
        % (epoch, train_function_err_avg, lr, train_etot_rmse_loss, time_cost))
        fid_err_log.close()

        if train_function_err_avg < min_loss:
            min_loss = train_function_err_avg
            works_epoch = 0
            name = direc + '/3layers_' + 'preMLFFNet_' + str(epoch)+'epoch.pt'
            # state = {'model': model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
            state = {'model': premodel.state_dict(), 'epoch': epoch}
            torch.save(state, name)
            print('saving model to {}'.format(name))
        else:
            works_epoch += 1
            if works_epoch > patience:
                name = direc + '/3layers_' + 'preMLFFNet.pt'
                state = {'model': premodel.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
                torch.save(state, name)
                print("Early stopping")
                break
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23457', rank=0, world_size=1)


# ==========================part4:模型finetuning==========================
if pm.isNNfinetuning == True:
    # model = MLFFNet()
    #model = LNNet()
    model = MLFF_dmirror()

    # if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=weight_decay)
    start = time.time()
    min_loss = np.inf
    start_epoch=1
    patience = 50

    if pm.isNNpretrain == True:   # True时表示load预训练的模型， False表示直接fine tuning
        path=r"./FC3model/3layers_preMLFFNet.pt"
        checkpoint = torch.load(path, map_location={'cpu':'cuda:0'})
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint['epoch']+1

    resume = False     #模型中断时重新训练
    if resume:
        path=r"./FC3model/3layers_MLFFNet_11epoch.pt"
        checkpoint = torch.load(path, map_location={'cpu':'cuda:0'})
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint['epoch']+1

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(start_epoch, n_epoch + 1):
            print("epoch " + str(epoch))
            start = time.time()
            lr = optimizer.param_groups[0]['lr']
            print("learning rate:" + str(lr))
            loss_function_err = 0
            train_epoch_force_square_loss = 0
            train_epoch_etot_square_loss = 0
            train_epoch_egroup_square_loss = 0
            train_epoch_force_RMSE_loss = 0
            train_epoch_etot_RMSE_loss = 0
            train_epoch_egroup_RMSE_loss = 0
            for i_batch, sample_batches in enumerate(loader_train):
                # loss, force_square_loss, etot_square_loss, egroup_square_loss, \
                # Force_RMSE_error, Force_ABS_error, Force_L2, Etot_RMSE_error, Etot_ABS_error, Etot_L2, \
                #     Egroup_RMSE_error, Egroup_ABS_error, Egroup_L2 = train(sample_batches, model, optimizer, nn.MSELoss())
                loss = train(sample_batches, model, optimizer, nn.MSELoss())
                # import ipdb; ipdb.set_trace()
                # Log train/loss to TensorBoard at every iteration
                n_iter = (epoch - 1) * len(loader_train) + i_batch + 1
                # writer.add_scalar('Train/loss', Force_RMSE_error, n_iter)
                # loss_function_err += loss
                # import ipdb; ipdb.set_trace()
                # train_epoch_force_square_loss += force_square_loss
                # train_epoch_etot_square_loss += etot_square_loss
                # train_epoch_egroup_square_loss += egroup_square_loss

                # train_epoch_force_RMSE_loss += Force_RMSE_error
                # train_epoch_etot_RMSE_loss += Etot_RMSE_error
                # train_epoch_egroup_RMSE_loss += Egroup_RMSE_error
        
            # train_function_err_avg = loss_function_err / len(loader_train)
            # train_epoch_force_square_loss = train_epoch_force_square_loss/len(loader_train)
            # train_epoch_etot_square_loss = train_epoch_etot_square_loss/len(loader_train)
            # train_epoch_egroup_square_loss = train_epoch_egroup_square_loss/len(loader_train)

            # train_force_rmse_loss = train_epoch_force_RMSE_loss/len(loader_train)
            # train_etot_rmse_loss = train_epoch_etot_RMSE_loss/len(loader_train)
            # train_egroup_rmse_loss = train_epoch_egroup_RMSE_loss/len(loader_train)

            end = time.time()
            time_cost = sec_to_hms(int(end-start))    #每个epoch的训练时间
            # print('Finetuning stage: epoch = {}, step = {}, train_function_err_avg = {:.8f}, train_epoch_force_square_loss = {:.8f}, \
            #     train_epoch_etot_square_loss = {:.8f}, train_epoch_egroup_square_loss = {:.8f}, lr = {}, time cost = {}, \
            #     training force rmse = {:.8f}, training etot rmse = {:.8f}, training egroup rmse = {:.8f}'.format(epoch, n_iter, \
            #         train_function_err_avg, train_epoch_force_square_loss, train_epoch_etot_square_loss, train_epoch_egroup_square_loss, \
            #             lr, time_cost, train_force_rmse_loss, train_etot_rmse_loss, train_egroup_rmse_loss)) 
            # print("loss :" + )

            # valid_loss_function_err = 0
            # valid_epoch_force_square_loss = 0
            # valid_epoch_etot_square_loss = 0
            # valid_epoch_egroup_square_loss = 0

            # valid_epoch_force_RMSE_loss = 0
            # valid_epoch_etot_RMSE_loss = 0
            # valid_epoch_egroup_RMSE_loss = 0
            
            # for i_batch, sample_batches in enumerate(loader_valid):
            #     error, force_square_loss, etot_square_loss, egroup_square_loss, \
            #     Force_RMSE_error, Force_ABS_error, Force_L2, Etot_RMSE_error, Etot_ABS_error, Etot_L2, \
            #         Egroup_RMSE_error, Egroup_ABS_error, Egroup_L2 = valid(sample_batches, model, nn.MSELoss())
            #     n_iter = (epoch - 1) * len(loader_valid) + i_batch + 1
            #     # writer.add_scalar('Val/loss', square_error, n_iter)
            #     valid_loss_function_err += error
            #     valid_epoch_force_square_loss += force_square_loss
            #     valid_epoch_etot_square_loss += etot_square_loss
            #     valid_epoch_egroup_square_loss += egroup_square_loss 

            #     valid_epoch_force_RMSE_loss += Force_RMSE_error
            #     valid_epoch_etot_RMSE_loss += Etot_RMSE_error
            #     valid_epoch_egroup_RMSE_loss += Egroup_RMSE_error

            # valid_loss_function_err = valid_loss_function_err/len(loader_valid)
            # valid_epoch_force_square_loss = valid_epoch_force_square_loss/len(loader_valid)
            # valid_epoch_etot_square_loss = valid_epoch_etot_square_loss/len(loader_valid)
            # valid_epoch_egroup_square_loss = valid_epoch_egroup_square_loss/len(loader_valid)

            # valid_force_rmse_loss = valid_epoch_force_RMSE_loss/len(loader_valid)
            # valid_etot_rmse_loss = valid_epoch_etot_RMSE_loss/len(loader_valid)
            # valid_egroup_rmse_loss = valid_epoch_egroup_RMSE_loss/len(loader_valid)

            # print('valid_loss_function_err = {:.8f}, valid_epoch_force_square_loss = {:.8f}, \
            #     valid_epoch_etot_square_loss = {:.8f}, valid_epoch_egroup_square_loss = {:.8f}, \
            #     valid force rmse = {:.8f}, valid etot rmse = {:.8f}, valid egroup rmse = {:.8f}'\
            #         .format(valid_loss_function_err, valid_epoch_force_square_loss, valid_epoch_etot_square_loss, \
            #         valid_epoch_egroup_square_loss, valid_force_rmse_loss, valid_etot_rmse_loss, valid_egroup_rmse_loss))
            
            if epoch % weight_decay_epoch == 0:   # 学习率衰减
                scheduler.step()
                print("eeeeeeeeeeeeeeeeeeee")
            # iprint = 1 #隔几个epoch记录一次误差
            # f_err_log=pm.dir_work+'finetuning.dat'
            # if epoch // iprint == 1:
            #     fid_err_log = open(f_err_log, 'w')
            # else:
            #     fid_err_log = open(f_err_log, 'a')
            # fid_err_log.write('%d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %s\n'       \
            # % (epoch, train_function_err_avg, train_epoch_force_square_loss, train_epoch_etot_square_loss, train_epoch_egroup_square_loss, \
            #     train_force_rmse_loss, train_etot_rmse_loss, train_egroup_rmse_loss, lr, \
            #     valid_loss_function_err, valid_epoch_force_square_loss, valid_epoch_etot_square_loss, valid_epoch_egroup_square_loss,\
            #         valid_force_rmse_loss, valid_etot_rmse_loss, valid_egroup_rmse_loss, time_cost))
            # fid_err_log.close()

            # if valid_loss_function_err < min_loss:
            #     min_loss = valid_loss_function_err
            #     works_epoch = 0
            #     name = direc + '/3layers_' + 'MLFFNet_' + str(epoch)+'epoch.pt'
            #     # state = {'model': model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
            #     state = {'model': model.state_dict(), 'epoch': epoch}
            #     torch.save(state, name)
            #     print('saving model to {}'.format(name))
            # else:
            #     works_epoch += 1
            #     if works_epoch > patience:
            #         name = direc + '/3layers_' + 'MLFFNet.pt'
            #         state = {'model': model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
            #         torch.save(state, name)
            #         print("Early stopping")
            #         break
# writer.close()
