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
#from model.LN import LNNet
from model.MLFF_dmirror import MLFF_dmirror
import torch.utils.data as Data
from torch.autograd import Variable
import math
sys.path.append(os.getcwd())
import parameters as pm 
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/pre_data')
from data_loader_2type import MovementDataset, get_torch_data
from scalers import DataScalers
from torch.utils.tensorboard import SummaryWriter
import torchviz
import time
import getopt

# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# parse optional parameters
opt_force_cpu = False
opt_magic = False
opt_epochs = 0
opt_lr = float(0)
opt_gamma = float(0)
opt_step = 0
opt_act = 'sigmoid'
opt_dtype = pm.training_dtype
opt_net_cfg = 'default'
opt_regular_wd = float(0)
opt_rseed = 2021
opt_batch_size = pm.batch_size
opt_log_level = logging.INFO
opt_file_log_level = logging.DEBUG
# session related options
opt_session_name = ''
opt_session_dir = ''
opt_logging_file = ''
opt_tensorboard_dir = ''

opts,args = getopt.getopt(sys.argv[1:],
    '-h-c-m-e:-l:-g:-t:-a:-d:-n:-w:-r:-b:-s:-o:-i:',
    ['help','cpu','magic','epochs=','lr=','gamma=',
     'step=','act=','dtype=','net_cfg=','weight_decay=','rseed=','batch_size=',
     'session=', 'log_level=', 'file_log_level='])

for opt_name,opt_value in opts:
    if opt_name in ('-h','--help'):
        print("")
        print("Available parameters:")
        print("     -h, --help                  :  print help info")
        print("     -c, --cpu                   :  force training run on cpu")
        print("     -m, --magic                 :  a magic flag for your testing code")
        print("     -e epochs, --epochs=epochs  :  specify training epochs")
        print("     -l lr, --lr=lr              :  specify initial training lr")
        print("     -g gamma, --gamma=gamma     :  specify gamma of StepLR scheduler")
        print("     -t step, --step=step        :  specify step_size of StepLR scheduler")
        print("     -a act, --act=act           :  specify activation_type of MLFF_dmirror")
        print("                                    current supported: [sigmoid, softplus]")
        print("     -d dtype, --dtype=dtype     :  specify default dtype: [float64, float32]")
        print("     -n cfg, --net_cfg=cfg       :  specify network cfg variable in parameters.py")
        print("                                    eg: -n MLFF_dmirror_cfg1")
        print("     -w val, --weight_decay=val  :  specify weight decay regularization value")
        print("     -r seed, --rseed=seed       :  specify random seed used in training")
        print("     -b size, --batch_size=size  :  specify batch size")
        print("     -s name, --session=name     :  specify the session name, log files, tensorboards and models")
        print("                                    will be saved to subdirectory named by this session name")
        print("     -o level, --log_level=level :  specify logging level of console")
        print("                                    available: DUMP < DEBUG < SUMMARY < [INFO] < WARNING < ERROR")
        print("                                    logging msg with level >= logging_level will be displayed")
        print("     -i L, --file_log_level=L    :  specify logging level of log file")
        print("                                    available: DUMP < [DEBUG] < SUMMARY < INFO < WARNING < ERROR")
        print("                                    logging msg with level >= logging_level will be recoreded")
        print("")
        exit()
    elif opt_name in ('-c','--cpu'):
        opt_force_cpu = True
    elif opt_name in ('-m','--magic'):
        opt_magic = True
    elif opt_name in ('-e','--epochs'):
        opt_epochs = int(opt_value)
    elif opt_name in ('-l','--lr'):
        opt_lr = float(opt_value)
    elif opt_name in ('-g','--gamma'):
        opt_gamma = float(opt_value)
    elif opt_name in ('-t','--step'):
        opt_step = int(opt_value)
    elif opt_name in ('-a','--act'):
        opt_act = opt_value
    elif opt_name in ('-d','--dtype'):
        opt_dtype = opt_value
    elif opt_name in ('-n','--net_cfg'):
        opt_net_cfg = opt_value
    elif opt_name in ('-w','--weight_decay'):
        opt_regular_wd = float(opt_value)
    elif opt_name in ('-r','--rseed'):
        opt_rseed = int(opt_value)
    elif opt_name in ('-b','--batch_size'):
        opt_batch_size = int(opt_value)
    elif opt_name in ('-s','--session'):
        opt_session_name = opt_value
        opt_session_dir = './'+opt_session_name+'/'
        opt_logging_file = opt_session_dir+'train.log'
        opt_tensorboard_dir = opt_session_dir+'/tensorboard/'
        if not os.path.exists(opt_session_dir):
            os.makedirs(opt_session_dir) 
        if not os.path.exists(opt_tensorboard_dir):
            os.makedirs(opt_tensorboard_dir) 
    elif opt_name in ('-o','--log_level'):
        if (opt_value == 'DUMP'):
            opt_log_level = logging_level_DUMP
        elif (opt_value == 'SUMMARY'):
            opt_log_level = logging_level_SUMMARY
        else:
            opt_log_level = 'logging.'+opt_value
            opt_log_level = eval(opt_log_level)
    elif opt_name in ('-i','--file_log_level'):
        if (opt_value == 'DUMP'):
            opt_file_log_level = logging_level_DUMP
        elif (opt_value == 'SUMMARY'):
            opt_file_log_level = logging_level_SUMMARY
        else:
            opt_file_log_level = 'logging.'+opt_value
            opt_file_log_level = eval(opt_file_log_level)

# setup logging module
logging.addLevelName(logging_level_DUMP, 'DUMP')
logging.addLevelName(logging_level_SUMMARY, 'SUMMARY')
logger = logging.getLogger('train')
logger.setLevel(logging_level_DUMP)

formatter = logging.Formatter("\33[0m\33[34;49m[%(name)s]\33[0m.\33[33;49m[%(levelname)s]\33[0m: %(message)s")
handler1 = logging.StreamHandler()
handler1.setLevel(opt_log_level)
handler1.setFormatter(formatter)
logger.addHandler(handler1)

if (opt_logging_file != ''):
    formatter = logging.Formatter("\33[0m\33[32;49m[%(asctime)s]\33[0m.\33[34;49m[%(name)s]\33[0m.\33[33;49m[%(levelname)s]\33[0m: %(message)s")
    handler2 = logging.FileHandler(filename = opt_logging_file)
    handler2.setLevel(opt_file_log_level)
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)

def dump(msg, *args, **kwargs):
    logger.log(logging_level_DUMP, msg, *args, **kwargs)
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)
def summary(msg, *args, **kwargs):
    logger.log(logging_level_SUMMARY, msg, *args, **kwargs)
def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)
def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs, exc_info=True)

# show start logging banner to logging file
#
summary("")
summary("#########################################################################################")
summary("#            ___          __                         __      __  ___       __  ___      #")
summary("#      |\ | |__  |  |    |__) |  | |\ | |\ | | |\ | / _`    /__`  |   /\  |__)  |       #")
summary("#      | \| |___ |/\|    |  \ \__/ | \| | \| | | \| \__>    .__/  |  /~~\ |  \  |       #")
summary("#                                                                                       #")
summary("#########################################################################################")
summary("")
summary(' '.join(sys.argv))
summary("")


# set default training dtype
#
# 1) dtype of model parameters during training
# 2) feature data will be casted to this dtype before using
#
if (opt_dtype == 'float64'):
    info("Training: set default dtype to float64")
    torch.set_default_dtype(torch.float64)
elif (opt_dtype == 'float32'):
    info("Training: set default dtype to float32")
    torch.set_default_dtype(torch.float32)
else:
    error("Training: unsupported dtype: %s" %opt_dtype)
    raise RuntimeError("Training: unsupported dtype: %s" %opt_dtype)

# set training device
if (opt_force_cpu == True):
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
info("Training: device = %s" %device)

# set random seed
torch.manual_seed(opt_rseed)
torch.cuda.manual_seed(opt_rseed)
info("Training: rseed = %s" %opt_rseed)

# set print precision
torch.set_printoptions(precision = 16)

# set tensorboard
if (opt_tensorboard_dir != ''):
    writer = SummaryWriter(opt_tensorboard_dir)
else:
    writer = None


def pretrain(sample_batches, premodel, optimizer, criterion):
    error=0
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].double().to(device))
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1]
    dump("==========Etot label==========")
    dump(Etot_label[0])
    input_data = Variable(sample_batches['input_feat'].double().to(device), requires_grad=True)
    neighbor = Variable(sample_batches['input_nblist'].int().to(device))  # [40,108,100]
    dfeat = Variable(sample_batches['input_dfeat'].double().to(device))  #[40,108,100,42,3]
   
    optimizer.zero_grad()
    model = premodel.to(device)
    model.train()
    Etot_predict, Ei_predict = model(input_data, dfeat, neighbor)
    dump("==========Etot predict==========")
    dump(Etot_predict[0])
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

def train(sample_batches, model, optimizer, criterion, last_epoch):
    error=0

    # floating part of sample_batches, cast to specified opt_dtype
    #
    if (opt_dtype == 'float64'):
        Etot_label = Variable(sample_batches['output_energy'][:,:,:].double().to(device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].double().to(device))   #[40,108,3]
        Egroup_label = Variable(sample_batches['input_egroup'].double().to(device))
        input_data = Variable(sample_batches['input_feat'].double().to(device), requires_grad=True)
        dfeat = Variable(sample_batches['input_dfeat'].double().to(device))  #[40,108,100,42,3]
        egroup_weight = Variable(sample_batches['input_egroup_weight'].double().to(device))
        divider = Variable(sample_batches['input_divider'].double().to(device))
    elif (opt_dtype == 'float32'):
        Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))   #[40,108,3]
        Egroup_label = Variable(sample_batches['input_egroup'].float().to(device))
        input_data = Variable(sample_batches['input_feat'].float().to(device), requires_grad=True)
        dfeat = Variable(sample_batches['input_dfeat'].float().to(device))  #[40,108,100,42,3]
        egroup_weight = Variable(sample_batches['input_egroup_weight'].float().to(device))
        divider = Variable(sample_batches['input_divider'].float().to(device))
    else:
        error("train(): unsupported opt_dtype %s" %opt_dtype)
        raise RuntimeError("train(): unsupported opt_dtype %s" %opt_dtype)

    # non-floating or derived part of sample_batches
    #
    atom_number = Etot_label.shape[1]
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1]
    neighbor = Variable(sample_batches['input_nblist'].int().to(device))  # [40,108,100]
    ind_img = Variable(sample_batches['ind_image'].int().to(device))

    # dumping what you want here
    #
    dump("defat.shape= %s" %(dfeat.shape,))
    dump("neighbor.shape = %s" %(neighbor.shape,))
    dump("dump dfeat ------------------->")
    dump(dfeat)
    dump("dump neighbor ------------------->")
    dump(neighbor)

    model = model.to(device)
    # model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)
    # model.train()
    # force_predict, Etot_predict, Ei_predict, Egroup_predict = model(input_data, dfeat, neighbor, egroup_weight, divider)
    Etot_predict, Force_predict = model(input_data, dfeat, neighbor, egroup_weight, divider)
    
    optimizer.zero_grad()
    # Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider)   #[40,108,1]

    # Etot_deviation = Etot_predict - Etot_label     # [40,1]
    dump("etot predict =============================================>")
    dump(Etot_predict)
    dump("etot label ===============================================>")
    dump(Etot_label)
    dump("force predict ============================================>")
    dump(Force_predict)
    dump("force label ==============================================>")
    dump(Force_label)

    if (last_epoch):
        summary("etot predict =============================================>")
        summary(Etot_predict)
        summary("etot label ===============================================>")
        summary(Etot_label)
        summary("force predict ============================================>")
        summary(Force_predict)
        summary("force label ==============================================>")
        summary(Force_label)

    # Etot_square_deviation = Etot_deviation ** 2
    # Etot_shape = Etot_label.shape[0]  #40
    # Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    # Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)
    # Etot_L2 = (1/Etot_shape) * Etot_square_deviation.sum()   #L2-->tf.reduce_mean(tf.square())
    # Ei_L2 = Etot_L2 / atom_number

    # Force_deviation = force_predict - Force_label
    # import ipdb;ipdb.set_trace()
    # print(force_predict[0,0])
    # print("==========force label==========")
    # print(Force_label[0,0])
    # Force_square_deviation = Force_deviation ** 2
    # Force_shape = Force_deviation.shape[0] * Force_deviation.shape[1] * Force_deviation.shape[2]   #40*108*3
    # Force_ABS_error = Force_deviation.norm(1) / Force_shape
    # Force_RMSE_error = math.sqrt(1/Force_shape) * Force_deviation.norm(2)
    # Force_L2 = (1/Force_shape) * Force_square_deviation.sum()

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
    #loss = pm.rtLossF * criterion(force_predict, Force_label) + pm.rtLossEtot * criterion(Etot_predict, Etot_label) + pm.rtLossE * criterion(Egroup_predict, Egroup_label)
    #
    # Etot_label.shape = [batch_size, 1], while Etot_predict.shape = [batch_size], so squeeze Etot_label to match
    #
    loss = pm.rtLossF * criterion(Force_predict, Force_label) + pm.rtLossEtot * criterion(Etot_predict, Etot_label)
    loss_F = criterion(Force_predict, Force_label)
    loss_Etot = criterion(Etot_predict, Etot_label)
    loss_Egroup = 0
    info("loss = %f (loss_etot = %f, loss_force = %f, RMSE_etot = %f, RMSE_force = %f)" %(loss, loss_Etot, loss_F, loss_Etot ** 0.5, loss_F ** 0.5))
    #w_f = loss_Etot / (loss_Etot + loss_F)
    #w_e = 1 - w_f
    #w_f = pm.rtLossF
    #w_e = pm.rtLossE
    #w_f = 0
    #w_e = 1
    #loss = w_e * criterion(Etot_predict, Etot_label) + w_f * criterion(force_predict, Force_label)
    #print("etot MSE loss: " + str(loss_Etot))
    #print("force MSE loss: " + str(loss_F))
    
    # print("weighted loss: " + str(loss))
    #time_start = time.time()
    loss.backward()                             # FIXME: need to try grad normalization with minibatch size

    for name, parameter in model.named_parameters():
        dump("dump model parameter (%s : %s) ------------------------>" %(name, parameter.size()))
        dump(parameter)
        dump("dump grad of model parameter (%s : %s) (not applied)--->" %(name, parameter.size()))
        dump(parameter.grad)
        if (last_epoch):
            summary("dump model parameter (%s : %s) ------------------------>" %(name, parameter.size()))
            summary(parameter)
            summary("dump grad of model parameter (%s : %s) (not applied)--->" %(name, parameter.size()))
            summary(parameter.grad)

    optimizer.step()
    #time_end = time.time()
    #print("update grad time:", time_end - time_start, 's')

    # error = error + float(loss.item())
    # return loss, force_square_loss, etot_square_loss, egroup_square_loss, \
    #     Force_RMSE_error, Force_ABS_error, Force_L2, Etot_RMSE_error, Etot_ABS_error, Etot_L2, \
    #          Egroup_RMSE_error, Egroup_ABS_error, Egroup_L2

    return loss, loss_Etot, loss_Egroup, loss_F

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
    Etot_predict, force_predict = model(input_data, dfeat, neighbor, egroup_weight, divider)
    # force_predict, Etot_predict, Ei_predict = model(input_data, dfeat, neighbor)
    # Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider)
    loss_F = criterion(force_predict, Force_label)
    loss_Etot = criterion(Etot_predict, Etot_label)
    '''
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
    '''
    # loss = pm.rtLossF * force_square_loss + pm.rtLossEtot * etot_square_loss + pm.rtLossE * egroup_square_loss
    # ===========loss 选取torch.nn的函数==========
    loss_F = criterion(force_predict, Force_label)
    loss_Etot = criterion(Etot_predict, Etot_label)
    w_f = loss_Etot / (loss_Etot + loss_F)
    w_e = 1 - w_f
    w_f = pm.rtLossF
    w_e = pm.rtLossE
    w_f = 0
    w_e = 1
    loss = w_f * criterion(force_predict, Force_label) + w_e * criterion(Etot_predict, Etot_label)

    error = error+float(loss.item())
    # return error, force_square_loss, etot_square_loss, egroup_square_loss, \
    #     Force_RMSE_error, Force_ABS_error, Force_L2, Etot_RMSE_error, Etot_ABS_error, Etot_L2, \
    #         Egroup_RMSE_error, Egroup_ABS_error, Egroup_L2
    return error, loss_F, loss_Etot

def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


# ==========================part1:数据读取==========================
batch_size = opt_batch_size 
train_data_path=pm.train_data_path
torch_train_data = get_torch_data(pm.natoms, train_data_path)
loader_train = Data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=False)

valid_data_path=pm.test_data_path
torch_valid_data = get_torch_data(pm.natoms, valid_data_path)
loader_valid = Data.DataLoader(torch_valid_data, batch_size=1, shuffle=False)

# ==========================part2:指定模型参数==========================

n_epoch = 25
if (opt_epochs != 0):
    n_epoch = opt_epochs
learning_rate = 0.1
weight_decay = 0.9
weight_decay_epoch = 10
direc = './FC3model_mini_pm_loss'

# for Scheduler
LR_base = 0.1
if (opt_lr != 0.):
    LR_base = opt_lr
LR_gamma = 0.9
if (opt_gamma != 0.):
    LR_gamma = opt_gamma
LR_step = 100
if (opt_step != 0):
    LR_step = opt_step

# for Regularization
REGULAR_wd = 0.0000001
REGULAR_wd = 0.
if (opt_regular_wd != 0.):
    REGULAR_wd = opt_regular_wd

info("Training: batch_size = %d" %batch_size)
info("Training: n_epoch = %d" %n_epoch)
info("Training: LR_base = %.16f" %LR_base)
info("Training: LR_gamma = %.16f" %LR_gamma)
info("Training: LR_step = %d" %LR_step)
info("Training: REGULAR_wd = %.16f" %REGULAR_wd)

#LR_milestones = [1000, 1500, 1700, 1800, 1900, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
#LR_tmax=200
#LR_eta_min=0.05
#LR_factor1 = 1.1
#LR_max = 0.8
#LR_steps = 1
#LR_epochs = 2000

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

# implement a linear scheduler
def LinearLR(optimizer, lr, total_epoch, cur_epoch):
    lr *= (1.0 - (float(cur_epoch) / float(total_epoch)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if pm.isNNfinetuning == True:
    data_scalers = DataScalers(f_ds=pm.f_data_scaler, f_feat=pm.f_train_feat, load=True)
    # import ipdb; ipdb.set_trace()
    # model = MLFFNet(data_scalers)
    # model = LNNet()
    model = MLFF_dmirror(opt_net_cfg, opt_act, device, opt_magic)

    # if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model)
    optimizer = optim.Adam(
                [
                    {'params': (p for name, p in model.named_parameters() if 'bias' not in name)},
                    {'params': (p for name, p in model.named_parameters() if 'bias' in name), 'weight_decay': 0.}
                ],
                lr=LR_base, weight_decay = REGULAR_wd)
    #optimizer = optim.SGD(model.parameters(), lr=LR_base, momentum=0.5)
    #scheduler = optim.lr_scheduler.OneCycleLR(
    #                            optimizer, max_lr=LR_max, 
    #                            steps_per_epoch=LR_steps, epochs=LR_epochs)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=LR_factor)
    #scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=LR_factor1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_milestones, gamma=LR_gamma)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_step, gamma=LR_gamma)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LR_tmax, eta_min=LR_eta_min)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_gamma)
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

    
    for epoch in range(start_epoch, n_epoch + 1):
        if (epoch == n_epoch):
            last_epoch = True
        else:
            last_epoch = False
        lr = optimizer.param_groups[0]['lr']
        info("<-------------------------  epoch %d (lr=%.16f) ------------------------->" %(epoch, lr))
        nr_total_sample = 0
        loss = 0.
        loss_Etot = 0.
        loss_Egroup = 0.
        loss_F = 0.
        for i_batch, sample_batches in enumerate(loader_train):
            batch_loss, batch_loss_Etot, batch_loss_Egroup, batch_loss_F = \
                train(sample_batches, model, optimizer, nn.MSELoss(), last_epoch)
            nr_batch_sample = sample_batches['input_feat'].shape[0]
            debug("nr_batch_sample = %s" %nr_batch_sample)
            loss += batch_loss * nr_batch_sample
            loss_Etot += batch_loss_Etot * nr_batch_sample
            loss_Egroup += batch_loss_Egroup * nr_batch_sample
            loss_F += batch_loss_F * nr_batch_sample
            nr_total_sample += nr_batch_sample

        # epoch loss update
        loss /= nr_total_sample
        loss_Etot /= nr_total_sample
        loss_Egroup /= nr_total_sample
        loss_F /= nr_total_sample
        RMSE_Etot = loss_Etot ** 0.5
        RMSE_Egroup = loss_Egroup ** 0.5
        RMSE_F = loss_F ** 0.5
        info("epoch_loss = %f (loss_Etot = %f, loss_F = %f, RMSE_Etot = %f, RMSE_F = %f)" %(loss, loss_Etot, loss_F, RMSE_Etot, RMSE_F))
        # update tensorboard
        if (writer is not None):
            writer.add_scalar('train_loss', loss, epoch)
            writer.add_scalar('train_loss_Etot', loss_Etot, epoch)
            writer.add_scalar('train_loss_Egroup', loss_Egroup, epoch)
            writer.add_scalar('train_loss_F', loss_F, epoch)
            writer.add_scalar('train_RMSE_Etot', RMSE_Etot, epoch)
            writer.add_scalar('train_RMSE_Egroup', RMSE_Egroup, epoch)
            writer.add_scalar('train_RMSE_F', RMSE_F, epoch)
            
        # print('Finetuning stage: epoch = {}, step = {}, train_function_err_avg = {:.8f}, train_epoch_force_square_loss = {:.8f}, \
        #     train_epoch_etot_square_loss = {:.8f}, train_epoch_egroup_square_loss = {:.8f}, lr = {}, time cost = {}, \
        #     training force rmse = {:.8f}, training etot rmse = {:.8f}, training egroup rmse = {:.8f}'.format(epoch, n_iter, \
        #         train_function_err_avg, train_epoch_force_square_loss, train_epoch_etot_square_loss, train_epoch_egroup_square_loss, \
        #             lr, time_cost, train_force_rmse_loss, train_etot_rmse_loss, train_egroup_rmse_loss)) 
        # print("loss :" + )

        #valid_loss_function_err = 0
        # valid_epoch_force_square_loss = 0
        # valid_epoch_etot_square_loss = 0
        # valid_epoch_egroup_square_loss = 0

        # valid_epoch_force_RMSE_loss = 0
        # valid_epoch_etot_RMSE_loss = 0
        # valid_epoch_egroup_RMSE_loss = 0

        #scheduler.step()
        LinearLR(optimizer=optimizer, lr=LR_base, total_epoch=n_epoch, cur_epoch=epoch)
        
        """
        for i_batch, sample_batches in enumerate(loader_valid):
            # error, force_square_loss, etot_square_loss, egroup_square_loss, \
            # Force_RMSE_error, Force_ABS_error, Force_L2, Etot_RMSE_error, Etot_ABS_error, Etot_L2, \
            #     Egroup_RMSE_error, Egroup_ABS_error, Egroup_L2 = valid(sample_batches, model, nn.MSELoss())
            error, loss_F, loss_Etot = valid(sample_batches, model, nn.MSELoss())
            n_iter = (epoch - 1) * len(loader_valid) + i_batch + 1
            # writer.add_scalar('Val/loss', square_error, n_iter)
            valid_loss_function_err += error
        #     valid_epoch_force_square_loss += force_square_loss
        #     valid_epoch_etot_square_loss += etot_square_loss
        #     valid_epoch_egroup_square_loss += egroup_square_loss 

        #     valid_epoch_force_RMSE_loss += Force_RMSE_error
        #     valid_epoch_etot_RMSE_loss += Etot_RMSE_error
        #     valid_epoch_egroup_RMSE_loss += Egroup_RMSE_error

        valid_loss_function_err = valid_loss_function_err/len(loader_valid)
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
        
        if epoch > weight_decay_epoch:   # 学习率衰减
            scheduler.step()
        iprint = 1 #隔几个epoch记录一次误差
        f_err_log=pm.dir_work+'mini_pm.dat'
        if epoch // iprint == 1:
            fid_err_log = open(f_err_log, 'w')
        else:
            fid_err_log = open(f_err_log, 'a')
        fid_err_log.write('%d %e %e \n'%(epoch, train_function_err_avg, valid_loss_function_err))
        # fid_err_log.write('%d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %s\n'       \
        # % (epoch, train_function_err_avg, train_epoch_force_square_loss, train_epoch_etot_square_loss, train_epoch_egroup_square_loss, \
        #     train_force_rmse_loss, train_etot_rmse_loss, train_egroup_rmse_loss, lr, \
        #     valid_loss_function_err, valid_epoch_force_square_loss, valid_epoch_etot_square_loss, valid_epoch_egroup_square_loss,\
        #         valid_force_rmse_loss, valid_etot_rmse_loss, valid_egroup_rmse_loss, time_cost))
        fid_err_log.close()

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
"""
if (writer is not None):
    writer.close()
