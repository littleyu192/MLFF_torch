#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.serialization import load
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import sys
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
codepath = os.path.abspath(sys.path[0])
sys.path.append(codepath+'/pre_data')
from data_loader_2type import MovementDataset, get_torch_data

# from tensorboardX import SummaryWriter
writer = SummaryWriter()
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(sample_batches, model):
    Etot_label = Variable(
        sample_batches['output_energy'][:, :, :].float().to(device))
    atom_number = Etot_label.shape[1]
    Etot_label = torch.sum(Etot_label, dim=1)  # [40,108,1]-->[40,1]
    Force_label = Variable(
        sample_batches['output_force'][:, :, :].float().to(device))  # [40,108,3]
    Egroup_label = Variable(sample_batches['input_egroup'].float().to(device))

    input_data = Variable(
        sample_batches['input_feat'].float().to(device), requires_grad=True)
    neighbor = Variable(
        sample_batches['input_nblist'].int().to(device))  # [40,108,100]
    dfeat = Variable(sample_batches['input_dfeat'].float().to(
        device))  # [40,108,100,42,3]
    egroup_weight = Variable(sample_batches['input_egroup_weight'].float().to(device))
    ind_img = Variable(sample_batches['ind_image'].int().to(device))
    divider = Variable(sample_batches['input_divider'].float().to(device))
    label = Variable(sample_batches['output_energy'].float().to(device))
    model.to(device)
    model.train()
    force_predict, Etot_predict, Ei_predict = model(input_data, dfeat, neighbor)
    Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider) 

    Etot_deviation = Etot_predict - Etot_label     # [40,1]
    Etot_square_deviation = Etot_deviation ** 2
    Etot_shape = Etot_label.shape[0]  # 40
    Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)
    Etot_L2 = (1/Etot_shape) * Etot_square_deviation.sum()   #L2-->tf.reduce_mean(tf.square())
    Ei_L2 = Etot_L2 / atom_number

    Force_deviation = force_predict - Force_label
    Force_square_deviation = Force_deviation ** 2
    Force_shape = Force_deviation.shape[0] * \
        Force_deviation.shape[1] * Force_deviation.shape[2]  # 40*108*3
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
    loss = pm.rtLossF * force_square_loss + pm.rtLossEtot * etot_square_loss + pm.rtLossE * egroup_square_loss
    error = float(loss.item())
    return error, force_square_loss, etot_square_loss, egroup_square_loss, \
        Force_RMSE_error, Force_ABS_error, Etot_RMSE_error, Etot_ABS_error, Egroup_RMSE_error, Egroup_ABS_error, \
        Etot_L2, Ei_L2, Force_L2, Egroup_L2


def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


# ==========================part1:数据读取==========================
batch_size = 1
test_data_path = pm.train_data_path
torch_test_data = get_torch_data(pm.natoms, test_data_path)
loader_test = Data.DataLoader(
    torch_test_data, batch_size=batch_size, shuffle=True)

# ==========================part2:load模型==========================
model = MLFFNet()
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
model.to(device)
path = r"./FC3model_minimize_Etot_force_tanh_dropout3/3layers_MLFFNet.pt"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'])

all_model_loss = []
avgModelError = 0
start = time.time()

# ==========================part3:测试集==========================
test_square_err = []
Force_square_loss = []
Force_RMSE_error = []
Force_ABS_error = []
Force_L2_loss = []
Etot_square_loss = []
Etot_RMSE_error = []
Etot_ABS_error = []
Etot_L2_loss = []
Egroup_square_loss = []
Egroup_RMSE_error = []
Egroup_ABS_error = []
Egroup_L2_loss = []
Ei_L2_loss = []

# with torch.no_grad():
start = time.time()
for i_batch, sample_batches in enumerate(loader_test):
    temp_error, force_square_loss, etot_square_loss, egroup_square_loss, \
        force_RMSE_error, force_ABS_error, etot_RMSE_error, etot_ABS_error, egroup_RMSE_error, egroup_ABS_error, \
            Etot_L2, Ei_L2, Force_L2, Egroup_L2 = test(sample_batches, model)
    test_square_err.append(temp_error)
    Force_square_loss.append(force_square_loss)
    Etot_square_loss.append(etot_square_loss)
    Egroup_square_loss.append(egroup_square_loss)

    Force_RMSE_error.append(force_RMSE_error)
    Force_ABS_error.append(force_ABS_error)
    Force_L2_loss.append(Force_L2)
    Etot_RMSE_error.append(etot_RMSE_error)
    Etot_ABS_error.append(etot_ABS_error)
    Etot_L2_loss.append(Etot_L2)
    Egroup_RMSE_error.append(egroup_RMSE_error)
    Egroup_ABS_error.append(egroup_ABS_error)
    Egroup_L2_loss.append(Egroup_L2)
    Ei_L2_loss.append(Ei_L2)

test_square_err = sum(test_square_err)/len(loader_test)
Force_square_loss = sum(Force_square_loss)/len(loader_test)
Etot_square_loss = sum(Etot_square_loss)/len(loader_test)
Egroup_square_loss = sum(Egroup_square_loss)/len(loader_test)

Force_RMSE_error = sum(Force_RMSE_error)/len(loader_test)
Force_ABS_error = sum(Force_ABS_error)/len(loader_test)
Force_L2_loss = sum(Force_L2_loss)/len(loader_test)
Etot_RMSE_error = sum(Etot_RMSE_error)/len(loader_test)
Etot_ABS_error = sum(Etot_ABS_error)/len(loader_test)
Etot_L2_loss = sum(Etot_L2_loss)/len(loader_test)
Egroup_RMSE_error = sum(Egroup_RMSE_error)/len(loader_test)
Egroup_ABS_error = sum(Egroup_ABS_error)/len(loader_test)
Egroup_L2_loss = sum(Egroup_L2_loss)/len(loader_test)
Ei_L2_loss = sum(Ei_L2_loss)/len(loader_test)

end = time.time()
time_cost = sec_to_hms(int(end-start))
print('averaged test square loss = {:.8f}, test f square loss = {:.8f}, test etot square loss = {:.8f}, test egroup square loss = {:.8f}, \
    testing force L2 loss = {:.8f}, testing ei L2 loss = {:.8f}, testing etot L2 loss = {:.8f}, time cost = {}'.format(
    test_square_err, Force_square_loss, Etot_square_loss, Egroup_square_loss, Force_L2_loss, Ei_L2_loss, Etot_L2_loss, time_cost))

f_err_log = pm.dir_work+'out_test_err.dat'
fid_err_log = open(f_err_log, 'w')
fid_err_log.write('%e %e %e %e %e %e %e %e %e %e %s %e %e %e %e\n' % (test_square_err, Force_square_loss, Etot_square_loss, Egroup_square_loss, \
    Force_RMSE_error, Force_ABS_error, Etot_RMSE_error, Etot_ABS_error, Egroup_RMSE_error, Egroup_ABS_error, time_cost, \
        Force_L2_loss, Ei_L2_loss, Etot_L2_loss, Egroup_L2_loss))
fid_err_log.close()