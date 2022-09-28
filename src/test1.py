#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from statistics import mode
from turtle import Turtle
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
from loss.AutomaticWeightedLoss import AutomaticWeightedLoss
from model.MLFF_v1 import MLFF
from model.MLFF import MLFFNet

from optimizer.kalmanfilter import GKalmanFilter, LKalmanFilter, SKalmanFilter

from model.dp import DP
import torch.utils.data as Data
from torch.autograd import Variable
import math
sys.path.append(os.getcwd())
import parameters as pm 
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/pre_data')
sys.path.append(codepath+'/..')

#import use_para as pm
#import parse_input
#parse_input.parse_input()
#
from data_loader_2type import MovementDataset, get_torch_data
from scalers import DataScalers
from utils import get_weight_grad
#from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import time
import getopt
import getpass

from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from sklearn.feature_selection import VarianceThreshold

import matplotlib
import matplotlib.pyplot as plt

# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# parse optional parameters
opt_force_cpu = False
opt_magic = False
opt_follow_mode = False
opt_recover_mode = False
opt_net_cfg = 'default'
# opt_act = 'tanh'
opt_act = 'sigmoid'
opt_optimizer = 'ADAM'
opt_momentum = float(0)
opt_regular_wd = float(0)
opt_scheduler = 'NONE'
opt_epochs = pm.n_epoch
opt_lr = float(0.001)
opt_gamma = float(0.99)
opt_step = 100
opt_batch_size = pm.batch_size
opt_dtype = pm.training_dtype
opt_rseed = 2021
# session and related options
opt_session_name = ''
opt_session_dir = ''
opt_logging_file = ''
opt_tensorboard_dir = ''
opt_model_dir = ''
opt_model_file = ''
opt_run_id = ''
# end session related
opt_log_level = logging.INFO
opt_file_log_level = logging.DEBUG
opt_journal_cycle = 1
# wandb related
opt_wandb = False
opt_wandb_entity = 'moleculenn'
opt_wandb_project = 'MLFF_torch'
# end wandb related
opt_init_b = False
opt_save_model = False

# scheduler specific options
opt_LR_milestones = None
opt_LR_patience = 0
opt_LR_cooldown = 0
opt_LR_total_steps = None
opt_LR_max_lr = 1.
opt_LR_min_lr = 0.
opt_LR_T_max = None

opt_autograd = True

opt_dp = False


opts,args = getopt.getopt(sys.argv[1:],
    '-h-c-m-f-R-n:-a:-z:-v:-w:-u:-e:-l:-g:-t:-b:-d:-r:-s:-o:-i:-j:',
    ['help','cpu','magic','follow','recover','net_cfg=','act=','optimizer=','momentum',
     'weight_decay=','scheduler=','epochs=','lr=','gamma=','step=',
     'batch_size=','dtype=','rseed=','session=','log_level=',
     'file_log_level=','j_cycle=','init_b','save_model',
     'milestones=','patience=','cooldown=','eps=','total_steps=',
     'max_lr=','min_lr=','T_max=',
     'wandb','wandb_entity=','wandb_project=',
     'auto_grad=', 'dmirror=', 'dp='])

for opt_name,opt_value in opts:
    if opt_name in ('-h','--help'):
        print("")
        print("Generic parameters:")
        print("     -h, --help                  :  print help info")
        print("     -c, --cpu                   :  force training run on cpu")
        print("     -m, --magic                 :  a magic flag for your testing code")
        print("     -f, --follow                :  follow a previous trained model file")
        print("     -R, --recover               :  breakpoint training")
        print("     -n cfg, --net_cfg=cfg       :  if -f/--follow is not set, specify network cfg in parameters.py")
        print("                                    eg: -n MLFF_dmirror_cfg1")
        print("                                    if -f/--follow is set, specify the model image file name")
        print("                                    eg: '-n best1' will load model image file best1.pt from session dir")
        print("     -a act, --act=act           :  specify activation_type of MLFF_dmirror")
        print("                                    current supported: [sigmoid, softplus]")
        print("     -z name, --optimizer=name   :  specify optimizer")
        print("                                    available : SGD ASGD RPROP RMSPROP ADAG")
        print("                                                ADAD ADAM ADAMW ADAMAX LBFGS")
        print("     -v val, --momentum=val      :  specify momentum parameter for optimizer")
        print("     -w val, --weight_decay=val  :  specify weight decay regularization value")
        print("     -u name, --scheduler=name   :  specify learning rate scheduler")
        print("                                    available  : LAMBDA STEP MSTEP EXP COS PLAT OC LR_INC LR_DEC")
        print("                                    LAMBDA     : lambda scheduler")
        print("                                    STEP/MSTEP : Step/MultiStep scheduler")
        print("                                    EXP/COS    : Exponential/CosineAnnealing") 
        print("                                    PLAT/OC/LR : ReduceLROnPlateau/OneCycle")
        print("                                    LR_INC     : linearly increase to max_lr")
        print("                                    LR_DEC     : linearly decrease to min_lr")
        print("     -e epochs, --epochs=epochs  :  specify training epochs")
        print("     -l lr, --lr=lr              :  specify initial training lr")
        print("     -g gamma, --gamma=gamma     :  specify gamma of StepLR scheduler")
        print("     -t step, --step=step        :  specify step_size of StepLR scheduler")
        print("     -b size, --batch_size=size  :  specify batch size")
        print("     -d dtype, --dtype=dtype     :  specify default dtype: [float64, float32]")
        print("     -r seed, --rseed=seed       :  specify random seed used in training")
        print("     -s name, --session=name     :  specify the session name, log files, tensorboards and models")
        print("                                    will be saved to subdirectory named by this session name")
        print("     -o level, --log_level=level :  specify logging level of console")
        print("                                    available: DUMP < DEBUG < SUMMARY < [INFO] < WARNING < ERROR")
        print("                                    logging msg with level >= logging_level will be displayed")
        print("     -i L, --file_log_level=L    :  specify logging level of log file")
        print("                                    available: DUMP < [DEBUG] < SUMMARY < INFO < WARNING < ERROR")
        print("                                    logging msg with level >= logging_level will be recoreded")
        print("     -j val, --j_cycle=val       :  specify journal cycle for tensorboard and data dump")
        print("                                    0: disable journaling")
        print("                                    1: record on every epoch [default]")
        print("                                    n: record on every n epochs")
        print("")
        print("scheduler specific parameters:")
        print("     --milestones=int_list       :  milestones for MultiStep scheduler")
        print("     --patience=int_val          :  patience for ReduceLROnPlateau")
        print("     --cooldown=int_val          :  cooldown for ReduceLROnPlateau")
        print("     --total_steps=int_val       :  total_steps for OneCycle scheduler")
        print("     --max_lr=float_val          :  max learning rate for OneCycle scheduler")
        print("     --min_lr=float_val          :  min learning rate for CosineAnnealing/ReduceLROnPlateau")
        print("     --T_max=int_val             :  T_max for CosineAnnealing scheduler")
        print("")
        print("     --dmirror                   :  calculate dE/dx layer by layer explicitly")
        print("     --auto_grad                 :  calculate dE/dx by autograd func")
        print("                                    using --dmirror or --auto_grad")
        print("                                    default: --auto_grad")
        print("")
        print("     --dp                    :  use dp method(emdedding net + fitting net)")
        print("                                    using --dp=True enable dp method")
        print("                                    adding -n DeepMD_cfg (see cu/parameters.py)")
        print("                                    defalt: --dp=False (see line 90)")
        print("")
        print("wandb parameters:")
        print("     --wandb                     :  ebable wandb, sync tensorboard data to wandb")
        print("     --wandb_entity=yr_account   :  your wandb entity or account (default is: moleculenn")
        print("     --wandb_project=yr_project  :  your wandb project name (default is: MLFF_torch)")
        print("")
        exit()
    elif opt_name in ('-c','--cpu'):
        opt_force_cpu = True
    elif opt_name in ('-m','--magic'):
        opt_magic = True
    elif opt_name in ('-R','--recover'):
        opt_recover_mode = True
        print(opt_recover_mode)
        # opt_follow_epoch = int(opt_value)
    elif opt_name in ('-f','--follow'):
        opt_follow_mode = True
        # opt_follow_epoch = int(opt_value)
    elif opt_name in ('-n','--net_cfg'):
        opt_net_cfg = opt_value
    elif opt_name in ('-a','--act'):
        opt_act = opt_value
    elif opt_name in ('-z','--optimizer'):
        opt_optimizer = opt_value
    elif opt_name in ('-v','--momentum'):
        opt_momentum = float(opt_value)
    elif opt_name in ('-w','--weight_decay'):
        opt_regular_wd = float(opt_value)
    elif opt_name in ('-u','--scheduler'):
        opt_scheduler = opt_value
    elif opt_name in ('-e','--epochs'):
        opt_epochs = int(opt_value)
    elif opt_name in ('-l','--lr'):
        opt_lr = float(opt_value)
    elif opt_name in ('-g','--gamma'):
        opt_gamma = float(opt_value)
    elif opt_name in ('-t','--step'):
        opt_step = int(opt_value)
    elif opt_name in ('-b','--batch_size'):
        opt_batch_size = int(opt_value)
    elif opt_name in ('-d','--dtype'):
        opt_dtype = opt_value
    elif opt_name in ('-r','--rseed'):
        opt_rseed = int(opt_value)
    elif opt_name in ('-s','--session'):
        opt_session_name = opt_value
        opt_session_dir = './'+opt_session_name+'/'
        opt_logging_file = opt_session_dir+'train.log'
        opt_model_dir = opt_session_dir+'model/'
        tensorboard_base_dir = opt_session_dir+'tensorboard/'
        if not os.path.exists(opt_session_dir):
            os.makedirs(opt_session_dir) 
        if not os.path.exists(opt_model_dir):
            os.makedirs(opt_model_dir)
        for i in range(1000):
            opt_run_id = 'run'+str(i)
            opt_tensorboard_dir = tensorboard_base_dir+opt_run_id
            if (not os.path.exists(opt_tensorboard_dir)):
                os.makedirs(opt_tensorboard_dir)
                break
        else:
            opt_tensorboard_dir = ''
            raise RuntimeError("reaches 1000 run dirs in %s, clean it" %opt_tensorboard_dir)
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
    elif opt_name in ('-j','--j_cycle'):
        opt_journal_cycle = int(opt_value)
    if opt_name in ('--milestones'):
        opt_LR_milestones = list(map(int, opt_value.split(',')))
    if opt_name in ('--patience'):
        opt_LR_patience = int(opt_value)
    if opt_name in ('--cooldown'):
        opt_LR_cooldown = int(opt_value)
    if opt_name in ('--total_steps'):
        opt_LR_total_steps = int(opt_value)
    if opt_name in ('--max_lr'):
        opt_LR_max_lr = float(opt_value)
    if opt_name in ('--min_lr'):
        opt_LR_min_lr = float(opt_value)
    if opt_name in ('--T_max'):
        opt_LR_T_max = int(opt_value)
    if opt_name in ('--wandb'):
        opt_wandb = True
        import wandb
    if opt_name in ('--wandb_entity'):
        opt_wandb_entity = opt_value
    if opt_name in ('--wandb_project'):
        opt_wandb_project = opt_value
    if opt_name in ('--init_b'):
        opt_init_b = True
    if opt_name in ('--save_model'):
        opt_save_model = True
    elif opt_name in ('--dmirror'):
        opt_autograd = False
    elif opt_name in ('--auto_grad'):
        opt_autograd = True
    elif opt_name in ('--dp='):
        opt_dp = eval(opt_value)

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
    if (opt_wandb is True):
        wandb.tensorboard.patch(root_logdir=opt_tensorboard_dir)
        wandb_run = wandb.init(entity=opt_wandb_entity, project=opt_wandb_project, reinit=True)
        wandb_run.name = getpass.getuser()+'-'+opt_session_name+'-'+opt_run_id
        wandb_run.save()
    #writer = SummaryWriter(opt_tensorboard_dir)
else:
    writer = None

#for plots 

dft_Ei = []
dft_Etot = []
dft_Force = [] 

dp_Ei = [] 
dp_Etot = []
dp_Force = []

num_atoms = 0.0 


def test(sample_batches, model, criterion):
    global dft_Ei, dft_Etot, dft_Force, dp_Ei, dp_Force, dp_Etot
    global num_atoms 

    if (opt_dtype == 'float64'):
        Ei_label = Variable(sample_batches['output_energy'][:,:,:].double().to(device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].double().to(device))   #[40,108,3]
        # Ep_label = Variable(sample_batches['output_ep'][:,:,:].double().to(device))
        if pm.dR_neigh:
            dR = Variable(sample_batches['input_dR'].double().to(device), requires_grad=True)
            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(device))
            Ri = Variable(sample_batches['input_Ri'].double().to(device), requires_grad=True)
            Ri_d = Variable(sample_batches['input_Ri_d'].to(device))
        else:
            Egroup_label = Variable(sample_batches['input_egroup'].double().to(device))
            input_data = Variable(sample_batches['input_feat'].double().to(device), requires_grad=True)
            dfeat = Variable(sample_batches['input_dfeat'].double().to(device))  #[40,108,100,42,3]
            egroup_weight = Variable(sample_batches['input_egroup_weight'].double().to(device))
            divider = Variable(sample_batches['input_divider'].double().to(device))
    elif (opt_dtype == 'float32'):
        Ei_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))   #[40,108,3]
        Egroup_label = Variable(sample_batches['input_egroup'].float().to(device))
        input_data = Variable(sample_batches['input_feat'].float().to(device), requires_grad=True)
        dfeat = Variable(sample_batches['input_dfeat'].float().to(device))  #[40,108,100,42,3]
        egroup_weight = Variable(sample_batches['input_egroup_weight'].float().to(device))
        divider = Variable(sample_batches['input_divider'].float().to(device))
        # Ep_label = Variable(sample_batches['output_ep'][:,:,:].float().to(device))
        if pm.dR_neigh:
            dR = Variable(sample_batches['input_dR'].float().to(device), requires_grad=True)
            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(device))
            Ri = Variable(sample_batches['input_Ri'].double().to(device), requires_grad=True)
            Ri_d = Variable(sample_batches['input_Ri_d'].to(device))
    else:
        error("train(): unsupported opt_dtype %s" %opt_dtype)
        raise RuntimeError("train(): unsupported opt_dtype %s" %opt_dtype)

    neighbor = Variable(sample_batches['input_nblist'].int().to(device))  # [40,108,100]
    # non_zero_neighbor = torch.count_nonzero(neighbor, dim=2)
    natoms_img = Variable(sample_batches['natoms_img'].int().to(device))  # [40,108,100]

    error=0
    atom_number = Ei_label.shape[1]
    Etot_label = torch.sum(Ei_label, dim=1)
    


    num_atoms = atom_number 


    model.eval()
    if opt_dp:
        Etot_predict, Ei_predict, Force_predict = model(Ri, Ri_d, dR_neigh_list, natoms_img, None, None)
    else:
        Etot_predict, Ei_predict, Force_predict = model(input_data, dfeat, neighbor, natoms_img, egroup_weight, divider)

    # loading data for plot 
    dft_Etot.append(float(Etot_label[0]))
    dp_Etot.append(float(Etot_predict[0]))

    for i in range(len(Ei_label[0])):
        dft_Ei.append(float(Ei_label[0][i]))
        dp_Ei.append(float(Ei_predict[0][i]))


    for i in range(len(Force_label[0])):
        dft_Force.append([float(item) for item in Force_label[0][i]])
        dp_Force.append([float(item) for item in Force_predict[0][i]])  


    #print (Force_label[0][1])
    #print (len(Force_label[0]))


    # Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider)
    loss_F = criterion(Force_predict, Force_label)
    loss_Etot = criterion(Etot_predict, Etot_label)
    loss_Ei = criterion(Ei_predict, Ei_label)
    
    error = float(loss_F.item()) + float(loss_Etot.item())

    f_err_log =  opt_session_dir + 'test.dat'
    if not os.path.exists(f_err_log):
        fid_err_log = open(f_err_log, 'w')
        fid_err_log.write('Etot_predict\t Force_predict_x\t Force_predict_y\t Force_predict_z\n')
    else:
        fid_err_log = open(f_err_log, 'a')
        fid_err_log.write('%e %e %e %e \n'%(Etot_predict.item(), Force_predict[0,0,0].item(), Force_predict[0,0,1].item(), Force_predict[0,0,2].item()))
        
    return Etot_label, Ei_label, Force_label, error, loss_Etot, loss_Ei, loss_F


# ==========================part1:指定参数 && 数据读取==========================

momentum = opt_momentum
REGULAR_wd = opt_regular_wd
n_epoch = opt_epochs
LR_base = opt_lr
LR_gamma = opt_gamma
LR_step = opt_step
batch_size = opt_batch_size 

if (opt_follow_mode == True):
    opt_model_file = opt_model_dir+opt_net_cfg+'.pt'

info("Training: session = %s" %opt_session_name)
info("Training: run_id = %s" %opt_run_id)
info("Training: journal_cycle = %d" %opt_journal_cycle)
info("Training: follow_mode = %s" %opt_follow_mode)
info("Training: recover_mode = %s" %opt_recover_mode)
info("Training: network = %s" %opt_net_cfg)
info("Training: model_dir = %s" %opt_model_dir)
info("Training: model_file = %s" %opt_model_file)
info("Training: activation = %s" %opt_act)
info("Training: optimizer = %s" %opt_optimizer)
info("Training: momentum = %.16f" %momentum)
info("Training: REGULAR_wd = %.16f" %REGULAR_wd)
info("Training: scheduler = %s" %opt_scheduler)
info("Training: batch_size = %d" %batch_size)



# ==========================part1:数据读取==========================
batch_size = 1
test_data_path=pm.test_data_path
torch_test_data = get_torch_data(test_data_path, False)

if pm.is_scale:
    scaler=load('scaler.pkl')
    torch_test_data.feat = scaler.transform(torch_test_data.feat)
    dfeat_tmp = torch_test_data.dfeat
    dfeat_tmp = dfeat_tmp.transpose(0, 1, 3, 2)
    dfeat_tmp = dfeat_tmp * scaler.scale_
    dfeat_tmp = dfeat_tmp.transpose(0, 1, 3, 2)
    torch_test_data.dfeat = dfeat_tmp

#dir = r"/data/data/husiyu/software/MLFFdataset/CuO20_raodong/"
dir = r"/data/data/husiyu/software/MLFFdataset/Zr200/"
loader_test = Data.DataLoader(torch_test_data, batch_size=batch_size, shuffle=False)
if opt_dp:  # 此处应读training中保存的stat 
    davg = np.load(dir + "train_data/davg.npy")
    dstd = np.load(dir + "/train_data/dstd.npy")
    ener_shift = np.load(dir + "train_data/ener_shift.npy")
    stat = [davg, dstd, ener_shift]

# ==========================part2:load模型==========================

#path = dir + "wlj/model/better.pt"  # cu
path = dir + "gxz/model/better.pt"  # cu



if opt_dp:
    model = DP(opt_net_cfg, opt_act, device, stat, opt_magic)
else:
    model = MLFFNet(device)
model.to(device)

checkpoint = torch.load(path,map_location=device)
model.load_state_dict(checkpoint['model'])

nr_total_sample = 0
test_loss = 0.
test_loss_Etot = 0.
test_loss_Ei = 0.
test_loss_F = 0.

for i_batch, sample_batches in enumerate(loader_test):
    natoms_sum = sample_batches['natoms_img'][0, 0].item()
    nr_batch_sample = sample_batches['output_energy'].shape[0]
    dft_etot_raw, dft_ei_raw, dft_force_raw,test_error_iter, batch_loss_Etot, batch_loss_Ei, batch_loss_F = test(sample_batches, model, nn.MSELoss())
    # n_iter = (epoch - 1) * len(loader_valid) + i_batch + 1
    test_loss += test_error_iter * nr_batch_sample
    test_loss_Etot += batch_loss_Etot.item() * nr_batch_sample
    test_loss_Ei += batch_loss_Ei.item() * nr_batch_sample
    test_loss_F += batch_loss_F.item() * nr_batch_sample
    nr_total_sample += nr_batch_sample

# epoch loss
test_loss /= nr_total_sample
test_loss_Etot /= nr_total_sample
test_loss_Ei /= nr_total_sample
test_loss_F /= nr_total_sample

test_RMSE_Etot = test_loss_Etot ** 0.5
test_RMSE_Ei = test_loss_Ei ** 0.5
test_RMSE_F = test_loss_F ** 0.5

info("test_loss = %.16f (test_RMSE_Etot = %.16f, test_RMSE_Ei = %.16f, test_RMSE_F = %.16f)" \
        %(test_loss, test_RMSE_Etot, test_RMSE_Ei, test_RMSE_F))

"""
    Plot starts 
"""

#********* plot etot ***********
dft_total_energy = np.array(dft_Etot)
nn_total_energy = np.array(dp_Etot)
# import ipdb;ipdb.set_trace()



e_tot_min = min(min(dft_total_energy), min(nn_total_energy))
e_tot_max = max(max(dft_total_energy), max(nn_total_energy))
lim_tot_min = e_tot_min - (e_tot_max - e_tot_min) * 0.1
lim_tot_max = e_tot_max + (e_tot_max - e_tot_min) * 0.1

e_tot_rms = test_RMSE_Etot 

plt.figure(figsize=(8,8))
plt.rcParams['savefig.dpi']=300
plt.rcParams['figure.dpi']=300

#plt.clf()
#plt.subplot(1,2,1)
#plt.title('NaCl Energy')
plt.scatter(dft_total_energy, nn_total_energy, s=10, color='maroon')
plt.plot((lim_tot_min, lim_tot_max), (lim_tot_min, lim_tot_max), ls='--', color='blue',linewidth=2)
plt.xlim(lim_tot_min, lim_tot_max)
plt.ylim(lim_tot_min, lim_tot_max)
#plt.rc('font',family='Times New Roman')
#plt.text(e_tot_min, e_tot_max, "Cu", fontsize=25, fontweight='medium', fontdict={'family':'Times New Roman'})

#plt.text(e_tot_min, e_tot_max,'Energy(eV): %.3E' %(e_tot_rms), fontsize=20)
#plt.text(e_tot_min, e_tot_max - (e_tot_max - e_tot_min)*0.1,'Energy/Natoms(eV): %.3E' %(e_tot_rms/num_atoms), fontsize=20)
plt.xlabel('DFT Energy')
plt.ylabel('DPKF Energy')
# plt.savefig("li_zwt_etot_3.png",format = "png")

#********* plot force **********

f_dft_plot = np.array(dft_Force)
f_nn_plot = np.array(dp_Force)

fmax = max(f_dft_plot.max(), f_nn_plot.max())
fmin = min(f_dft_plot.min(), f_nn_plot.min())
lim_f_min = fmin - (fmax-fmin)*0.1
lim_f_max = fmax + (fmax-fmin)*0.1

f_rms = test_RMSE_F

#plt.title('NaCl Force')
plt.xlim(lim_f_min, lim_f_max)
plt.ylim(lim_f_min, lim_f_max)
#plt.text(fmin, fmax, "Cu", fontsize=25, fontweight='medium', fontdict={'family':'Times New Roman'})

#plt.text(fmin, fmax,'Force(eV/A): %.3E' %(f_rms),fontsize=20)
plt.plot((lim_f_min, lim_f_max), (lim_f_min, lim_f_max), ls='--', color='blue', linewidth=2)
plt.scatter(f_dft_plot, f_nn_plot, s=10, color='green')
plt.ylabel('DPKF Force')
plt.xlabel('DFT Force')
plt.savefig('li_zwt_force.png', format='png')
#*********plot e_i ************


dft_atomic_energy = np.array(dft_Ei)
nn_atomic_energy = np.array(dft_Ei)

e_min = min(min(dft_atomic_energy), min(nn_atomic_energy))
e_max = max(max(dft_atomic_energy), max(nn_atomic_energy))
lim_min = e_min - (e_max - e_min) * 0.1
lim_max = e_max + (e_max - e_min) * 0.1


e_atomic_rms = test_RMSE_Ei

plt.title('atomic energy')
plt.scatter(dft_atomic_energy, nn_atomic_energy, s=0.5)
plt.plot((lim_min, lim_max), (lim_min, lim_max), ls='--', color='red',linewidth=5)
plt.xlim(lim_min, lim_max)
plt.ylim(lim_min, lim_max)
plt.xlabel('DFT energy (eV)')
plt.ylabel('MLFF energy (eV)')
plt.text(lim_min, lim_max-(lim_max-lim_min)*0.1,'Ei, rmse: %.3E' % (e_atomic_rms))
plt.savefig('atomic_energy.png', format='png')
