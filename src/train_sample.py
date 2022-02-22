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
from loss.AutomaticWeightedLoss import AutomaticWeightedLoss
from model.LN import LNNet
from model.MLFF import preMLFF, MLFF
from model.FCold import preMLFFNet, MLFFNet

from model.deepmd import preDeepMD, DeepMD
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
# from tensorboardX import SummaryWriter
import time
import getopt
import getpass

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
opt_act = 'sigmoid'
opt_optimizer = 'ADAM'
opt_momentum = float(0)
opt_regular_wd = float(0)
opt_scheduler = 'NONE'
opt_epochs = 10000
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

opt_deepmd = False
# opt_deepmd = True

opts,args = getopt.getopt(sys.argv[1:],
    '-h-c-m-f-R-n:-a:-z:-v:-w:-u:-e:-l:-g:-t:-b:-d:-r:-s:-o:-i:-j:',
    ['help','cpu','magic','follow','recover','net_cfg=','act=','optimizer=','momentum',
     'weight_decay=','scheduler=','epochs=','lr=','gamma=','step=',
     'batch_size=','dtype=','rseed=','session=','log_level=',
     'file_log_level=','j_cycle=','init_b','save_model',
     'milestones=','patience=','cooldown=','eps=','total_steps=',
     'max_lr=','min_lr=','T_max=',
     'wandb','wandb_entity=','wandb_project=',
     'auto_grad=', 'dmirror=', 'deepmd='])

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
        print("     --deepmd                    :  use deepmd method(emdedding net + fitting net)")
        print("                                    using --deepmd=True enable deepmd method")
        print("                                    adding -n DeepMD_cfg (see cu/parameters.py)")
        print("                                    defalt: --deepmd=False (see line 90)")
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
    elif opt_name in ('--deepmd='):
        opt_deepmd = eval(opt_value)

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
    writer = SummaryWriter(opt_tensorboard_dir)
else:
    writer = None


def get_loss_func(start_lr, real_lr, has_fi, lossFi, has_etot, loss_Etot, has_egroup, loss_Egroup, has_ei, loss_Ei):
    start_pref_egroup = 0.02
    limit_pref_egroup = 1.0
    start_pref_F = 1000  #1000
    limit_pref_F = 1.0
    start_pref_etot = 0.02   
    limit_pref_etot = 1.0  
    start_pref_ei = 0.02
    limit_pref_ei = 1.0
    pref_fi = has_fi * (limit_pref_F + (start_pref_F - limit_pref_F) * real_lr / start_lr)
    pref_etot = has_etot * (limit_pref_etot + (start_pref_etot - limit_pref_etot) * real_lr / start_lr)
    pref_egroup = has_egroup * (limit_pref_egroup + (start_pref_egroup - limit_pref_egroup) * real_lr / start_lr)
    pref_ei = has_ei * (limit_pref_ei + (start_pref_ei - limit_pref_ei) * real_lr / start_lr)
    l2_loss = 0
    if has_fi:
        l2_loss += pref_fi * lossFi
    if has_etot:
        l2_loss += 0.009259259259 * pref_etot * loss_Etot  # 1/natoms
    if has_egroup :
        l2_loss += pref_egroup * loss_Egroup
    if has_ei :
        l2_loss += pref_ei * loss_Ei 
    # data = [learning_rate, loss_Egroup, lossFi, loss_Etot, l2_loss];
    # save_prefactor_file(pm.dir_work+"prefactor_loss.csv", data)
    # print("=====real learning rate=====")
    # print(real_lr)
    l2_loss = torch.sqrt(l2_loss)
    return l2_loss


# 第iter次迭代时进行计算并更新学习率
def adjust_lr(iter, start_lr=0.001, stop_lr=3.51e-8):
    stop_step = 1000000
    decay_step=5000
    decay_rate = np.exp(np.log(stop_lr/start_lr) / (stop_step/decay_step))
    real_lr = start_lr * np.power(decay_rate, (iter//decay_step))
    return real_lr

def train(i_batch, sample_batches, model, optimizer, criterion, last_epoch, real_lr):
    # floating part of sample_batches, cast to specified opt_dtype
    #
    if (opt_dtype == 'float64'):
        Ei_label = Variable(sample_batches['output_energy'][:,:,:].double().to(device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].double().to(device))   #[40,108,3]
        Egroup_label = Variable(sample_batches['input_egroup'].double().to(device))
        input_data = Variable(sample_batches['input_feat'].double().to(device), requires_grad=True)
        dfeat = Variable(sample_batches['input_dfeat'].double().to(device))  #[40,108,100,42,3]
        egroup_weight = Variable(sample_batches['input_egroup_weight'].double().to(device))
        divider = Variable(sample_batches['input_divider'].double().to(device))
        if pm.dR_neigh:
            dR = Variable(sample_batches['input_dR'].double().to(device), requires_grad=True)
            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(device))

    elif (opt_dtype == 'float32'):
        Ei_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))   #[40,108,3]
        Egroup_label = Variable(sample_batches['input_egroup'].float().to(device))
        input_data = Variable(sample_batches['input_feat'].float().to(device), requires_grad=True)
        dfeat = Variable(sample_batches['input_dfeat'].float().to(device))  #[40,108,100,42,3]
        egroup_weight = Variable(sample_batches['input_egroup_weight'].float().to(device))
        divider = Variable(sample_batches['input_divider'].float().to(device))
        if pm.dR_neigh:
            dR = Variable(sample_batches['input_dR'].float().to(device), requires_grad=True)
            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(device))

    else:
        error("train(): unsupported opt_dtype %s" %opt_dtype)
        raise RuntimeError("train(): unsupported opt_dtype %s" %opt_dtype)

    # non-floating or derived part of sample_batches
    #
    atom_number = Ei_label.shape[1]
    Etot_label = torch.sum(Ei_label, dim=1)   #[40,108,1]-->[40,1]
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


    if opt_deepmd:
        Etot_predict, Ei_predict, Force_predict = model(dR, dR_neigh_list)
    else:
        Etot_predict, Ei_predict, Force_predict = model(input_data, dfeat, neighbor, egroup_weight, divider)
    
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

    # iprint = 1 #隔几个epoch记录一次误差
    # f_err_log=pm.dir_work+'images.dat'
    # if not os.path.isfile(f_err_log):
    #     fid_err_log = open(f_err_log, 'w')
    # else:
    #     fid_err_log = open(f_err_log, 'a')
    # fid_err_log.write('%e %e %e %e %e \n'%(Etot_predict[0,0], Etot_label[0,0], Force_predict[0,0,0], Force_predict[0,0,1], Force_predict[0,0,2]))
    # fid_err_log.write('%e %e %e %e %e \n'%(Etot_predict[1,0], Etot_label[1,0], Force_predict[1,0,0], Force_predict[1,0,1], Force_predict[1,0,2]))

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
    
    # shift 之后的 label
    # Ei_label = Ei_label - 5133.92272
    # Etot_label = Etot_label - 554464.1386

    loss_F = criterion(Force_predict, Force_label)
    loss_Etot = criterion(Etot_predict, Etot_label)
    loss_Ei = criterion(Ei_predict, Ei_label)
    # import ipdb;ipdb.set_trace()
    # if i_batch % 2:
    #     loss = loss_Ei + loss_Etot
    # else:
    #     loss = loss_F

    # w_ei = 1
    # loss = pm.rtLossF * loss_F+ pm.rtLossEtot * loss_Etot + w_ei * loss_Ei
    # loss = loss_F   # torch.sqrt(loss_F)   # torch.sqrt(loss_Etot)
    #info("loss = %.16f (loss_etot = %.16f, loss_force = %.16f, RMSE_etot = %.16f, RMSE_force = %.16f)" %(loss, loss_Etot, loss_F, loss_Etot ** 0.5, loss_F ** 0.5))
    
    start_lr = opt_lr
    w_f = 0
    w_e = 1
    w_eg = 0
    loss_Egroup = 0
    w_ei = 1
    loss = get_loss_func(start_lr, real_lr, w_f, loss_F, w_e, loss_Etot, w_eg, loss_Egroup, w_ei, loss_Ei)

    # loss = awl(loss_Etot, loss_Ei, loss_F)
    # print(awl.parameters())

    # w_f = loss_Etot / (loss_Etot + loss_F)
    # w_e = 1 - w_f
    # w_f = pm.rtLossF
    # w_e = pm.rtLossE
    # w_f = 0
    # w_e = 1
    # loss = w_e * criterion(Etot_predict, Etot_label) + w_f * criterion(Force_predict, Force_label) + loss_Ei
    #print("etot MSE loss: " + str(loss_Etot))
    #print("force MSE loss: " + str(loss_F))
    
    # print("weighted loss: " + str(loss))
    #time_start = time.time()
    loss.backward()


    optimizer.step()
    #time_end = time.time()
    #print("update grad time:", time_end - time_start, 's')

    # error = error + float(loss.item())
    # return loss, force_square_loss, etot_square_loss, egroup_square_loss, \
    #     Force_RMSE_error, Force_ABS_error, Force_L2, Etot_RMSE_error, Etot_ABS_error, Etot_L2, \
    #          Egroup_RMSE_error, Egroup_ABS_error, Egroup_L2

    return loss, loss_Etot, loss_Ei, loss_F


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
info("Training: n_epoch = %d" %n_epoch)
info("Training: LR_base = %.16f" %LR_base)
info("Training: LR_gamma = %.16f" %LR_gamma)
info("Training: LR_step = %d" %LR_step)
info("Training: batch_size = %d" %batch_size)

# scheduler specific options
info("Scheduler: opt_LR_milestones = %s" %opt_LR_milestones)
info("Scheduler: opt_LR_patience = %s" %opt_LR_patience)
info("Scheduler: opt_LR_cooldown = %s" %opt_LR_cooldown)
info("Scheduler: opt_LR_total_steps = %s" %opt_LR_total_steps)
info("Scheduler: opt_LR_max_lr = %s" %opt_LR_max_lr)
info("Scheduler: opt_LR_min_lr = %s" %opt_LR_min_lr)
info("Scheduler: opt_LR_T_max = %s" %opt_LR_T_max)
info("scheduler: opt_autograd = %s" %opt_autograd)

train_data_path = pm.train_data_path
torch_train_data = get_torch_data(pm.natoms, train_data_path)
loader_train = Data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=True)

if opt_deepmd:
    davg, dstd, ener_shift = torch_train_data.get_stat()
    stat = [davg, dstd, ener_shift]

valid_data_path=pm.test_data_path
torch_valid_data = get_torch_data(pm.natoms, valid_data_path)
loader_valid = Data.DataLoader(torch_valid_data, batch_size=1, shuffle=True)


# ==========================part2:模型finetuning==========================

# implement a linear scheduler
def LinearLR(optimizer, base_lr, target_lr, total_epoch, cur_epoch):
    lr = base_lr - (base_lr - target_lr) * (float(cur_epoch) / float(total_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if pm.isNNfinetuning == True:
    data_scalers = DataScalers(f_ds=pm.f_data_scaler, f_feat=pm.f_train_feat, load=True)

    # deepmd model test
    if opt_deepmd:
        model = DeepMD(opt_net_cfg, opt_act, device, stat, opt_magic)
    else:
        model = MLFF(opt_net_cfg, opt_act, device, opt_magic, opt_autograd)
        # this is a temp fix for a quick test
        if (opt_init_b == True):
            for name, p in model.named_parameters():
                if ('linear_3.bias' in name):
                    dump(p)
                    p.data.fill_(166.0)
                    dump(p)
        # model = MLFFNet(data_scalers)
    model.to(device)
    # import ipdb; ipdb.set_trace()
    # if opt_follow_mode==True:
    #     checkpoint = torch.load(opt_model_file,map_location=device)
    #     model.load_state_dict(checkpoint['model'],strict=False)
    

    start_epoch = 1
    if (opt_recover_mode == True):
        if (opt_session_name == ''):
            raise RuntimeError("you must run follow-mode from an existing session")
        opt_latest_file = opt_model_dir+'latest.pt'
        checkpoint = torch.load(opt_latest_file,map_location=device)
        '''
        跟deepmd对齐时的模型转换，直接从latest.pt continue时 注释下面一段
        
        # weight_path = "/home/husiyu/software/deepMD/deepmd-kit-gpu/dataset/cu1/data"  # 小样本测试用
        weight_path = "/home/husiyu/software/deepMD/deepmd-kit-gpu/dataset/cu1000/data"
        # dict_keys(['global_step:0', 'descrpt_attr/t_avg:0', 'descrpt_attr/t_std:0', 'filter_type_0/matrix_1_0:0', 'filter_type_0/bias_1_0:0', 'filter_type_0/matrix_2_0:0', 'filter_type_0/bias_2_0:0', 'filter_type_0/matrix_3_0:0', 'filter_type_0/bias_3_0:0', 'layer_0_type_0/matrix:0', 'layer_0_type_0/bias:0', 'layer_1_type_0/matrix:0', 'layer_1_type_0/bias:0', 'layer_1_type_0/idt:0', 'layer_2_type_0/matrix:0', 'layer_2_type_0/bias:0', 'layer_2_type_0/idt:0', 'final_layer_type_0/matrix:0', 'final_layer_type_0/bias:0', 'beta1_power:0', 'beta2_power:0'])
        # odict_keys(['embeding_net.weights.weight0', 'embeding_net.weights.weight1', 'embeding_net.weights.weight2', 'embeding_net.bias.bias0', 'embeding_net.bias.bias1', 'embeding_net.bias.bias2', 'embeding_net.resnet_dt.resnet_dt0', 'embeding_net.resnet_dt.resnet_dt1', 'embeding_net.resnet_dt.resnet_dt2', 'fitting_net.weights.weight0', 'fitting_net.weights.weight1', 'fitting_net.weights.weight2', 'fitting_net.weights.weight3', 'fitting_net.bias.bias0', 'fitting_net.bias.bias1', 'fitting_net.bias.bias2', 'fitting_net.bias.bias3'])
        map_relation = {"embeding_net.weights.weight0" : "filter_type_0/matrix_1_0:0",  #(1,25)
                        "embeding_net.bias.bias0" : "filter_type_0/bias_1_0:0",         #(1,25)
                        "embeding_net.weights.weight1" : "filter_type_0/matrix_2_0:0",  #(25, 50)
                        "embeding_net.bias.bias1" : "filter_type_0/bias_2_0:0",         #(1,50)
                        "embeding_net.weights.weight2" : "filter_type_0/matrix_3_0:0",  #(50,100)
                        "embeding_net.bias.bias2" : "filter_type_0/bias_3_0:0",        #(1,100)
                        "fitting_net.weights.weight0" : "layer_0_type_0/matrix:0",     #(1600, 240)
                        "fitting_net.bias.bias0" : "layer_0_type_0/bias:0",            #(1,240)-->(240,)
                        "fitting_net.weights.weight1" : "layer_1_type_0/matrix:0",     #(240, 240)
                        "fitting_net.bias.bias1" : "layer_1_type_0/bias:0",            #(1,240)-->(240,)
                        "fitting_net.resnet_dt.resnet_dt1":"layer_1_type_0/idt:0",             # (1,240):(240,)
                        "fitting_net.weights.weight2" : "layer_2_type_0/matrix:0",     #(240, 240)
                        "fitting_net.bias.bias2" : "layer_2_type_0/bias:0",            #(1,240)-->(240,)
                        "fitting_net.resnet_dt.resnet_dt2":"layer_2_type_0/idt:0",             #(240,) 
                        "fitting_net.weights.weight3" : "final_layer_type_0/matrix:0", #(240, 1)
                        "fitting_net.bias.bias3" : "final_layer_type_0/bias:0",        #(1,1)-->(1,)
        }
        deepmd_weight = np.load(weight_path + "/weights.npy", allow_pickle='TRUE').item()
        model_weights = checkpoint['model']
        for name, value in model_weights.items():
            copying = torch.from_numpy(deepmd_weight[map_relation[name]])
            if 'fitting_net.bias' in name or 'resnet' in name:
                copying = copying.unsqueeze(0)
            model_weights[name] = copying
       
        # import ipdb; ipdb.set_trace() 
        '''
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint['epoch'] + 1

        # TODO: clean the codes above
        #       1) need to fix opt_net_cfg, the model still need to specify in follow-mode
        #       2) add opt_image_file and it's parameter form
        #       3) model store/load need to handle cpu/gpu
        #       4) handle tensorboard file, can we modify accroding to 'epoch'?

    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)

    # set model parameter properties, do not apply weight decay to bias parameter
    # except for LBFGS, which do not support pre-parameter options
    # model_parameters = [
    #                 {'params': (p for name, p in model.named_parameters() if 'bias' not in name)},
    #                 {'params': (p for name, p in model.named_parameters() if 'bias' in name), 'weight_decay': 0.}]
    
    # awl = AutomaticWeightedLoss(3)	
    # optimizer = optim.Adam([
    #             {'params': model.parameters()},
    #             {'params': awl.parameters(), 'weight_decay': 0}
    #         ])

    model_parameters = model.parameters()
    # import ipdb; ipdb.set_trace()
    

    if (opt_optimizer == 'SGD'):
        optimizer = optim.SGD(model_parameters, lr=LR_base, momentum=momentum, weight_decay=REGULAR_wd)
    elif (opt_optimizer == 'ASGD'):
        optimizer = optim.ASGD(model_parameters, lr=LR_base, weight_decay=REGULAR_wd)
    elif (opt_optimizer == 'RPROP'):
        optimizer = optim.Rprop(model_parameters, lr=LR_base)
    elif (opt_optimizer == 'RMSPROP'):
        optimizer = optim.RMSprop(model_parameters, lr=LR_base, weight_decay=REGULAR_wd, momentum=momentum)
    elif (opt_optimizer == 'ADAG'):
        optimizer = optim.Adagrad(model_parameters, lr=LR_base, weight_decay=REGULAR_wd)
    elif (opt_optimizer == 'ADAD'):
        optimizer = optim.Adadelta(model_parameters, lr=LR_base, weight_decay=REGULAR_wd)
    elif (opt_optimizer == 'ADAM'):
        optimizer = optim.Adam(model_parameters, lr=LR_base, weight_decay = REGULAR_wd)
    elif (opt_optimizer == 'ADAMW'):
        optimizer = optim.AdamW(model_parameters, lr=LR_base, weight_decay = REGULAR_wd)
    elif (opt_optimizer == 'ADAMAX'):
        optimizer = optim.Adamax(model_parameters, lr=LR_base, weight_decay=REGULAR_wd)
    elif (opt_optimizer == 'LBFGS'):
        optimizer = optim.LBFGS(model.parameters(), lr=LR_base)
    else:
        error("unsupported optimizer: %s" %opt_optimizer)
        raise RuntimeError("unsupported optimizer: %s" %opt_optimizer)
    # if (opt_recover_mode == True):
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    # TODO: LBFGS is not done yet
    # FIXME: train process should be better re-arranged to 
    #        support this closure cleanly
    # example code for LBFGS closure()
    # def lbfgs_closure():
    #    optimizer.zero_grad()
    #    output = model(input)
    #    loss = loss_fn(output, target)
    #    loss.backward()
    #    return loss
    #optimizer.step(lbfgs_closure)

    # user specific LambdaLR lambda function
    lr_lambda = lambda epoch: LR_gamma ** epoch

    if (opt_scheduler == 'LAMBDA'):
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif (opt_scheduler == 'STEP'):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_step, gamma=LR_gamma)
    elif (opt_scheduler == 'MSTEP'):
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt_LR_milestones, gamma=LR_gamma)
    elif (opt_scheduler == 'EXP'):
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_gamma)
    elif (opt_scheduler == 'COS'):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_LR_T_max, eta_min=opt_LR_min_lr)
    elif (opt_scheduler == 'PLAT'):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=LR_gamma, 
                        patience=opt_LR_patience, cooldown=opt_LR_cooldown, min_lr=opt_LR_min_lr)
    elif (opt_scheduler == 'OC'):
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt_LR_max_lr, total_steps=opt_LR_total_steps)
    elif (opt_scheduler == 'LR_INC'):
        # do nothing, will direct call LR scheduler at each epoch
        pass
    elif (opt_scheduler == 'LR_DEC'):
        # do nothing, will direct call LR scheduler at each epoch
        pass
    elif (opt_scheduler == 'NONE'):
        pass
    else:
        error("unsupported scheduler: %s" %opt_schedler)
        raise RuntimeError("unsupported scheduler: %s" %opt_scheduler)

    #min_loss = np.inf
    
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
        loss_Ei = 0.
        loss_F = 0.
        for i_batch, sample_batches in enumerate(loader_train):

            global_step = epoch * len(loader_train) + i_batch
            real_lr = adjust_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = real_lr

            batch_loss, batch_loss_Etot, batch_loss_Ei, batch_loss_F = \
                train(i_batch, sample_batches, model, optimizer, nn.MSELoss(), last_epoch, real_lr)
            print("batch loss:" + str(batch_loss.item()))
            print("batch mse ei:" + str(batch_loss_Ei.item()))
            print("batch mse etot:" + str(batch_loss_Etot.item()))
            print("batch mse F:" + str(batch_loss_F.item()))
            print("=============================")
            nr_batch_sample = sample_batches['input_feat'].shape[0]
            debug("nr_batch_sample = %s" %nr_batch_sample)
            loss += batch_loss.item() * nr_batch_sample
            loss_Etot += batch_loss_Etot.item() * nr_batch_sample
            loss_Ei += batch_loss_Ei.item() * nr_batch_sample
            loss_F += batch_loss_F.item() * nr_batch_sample
            nr_total_sample += nr_batch_sample

            # OneCycleLR scheduler steps() at each batch
            if (opt_scheduler == 'OC'):
                scheduler.step()

        # epoch loss update
        loss /= nr_total_sample
        loss_Etot /= nr_total_sample
        loss_Ei /= nr_total_sample
        loss_F /= nr_total_sample
        RMSE_Etot = loss_Etot ** 0.5
        RMSE_Ei = loss_Ei ** 0.5
        RMSE_F = loss_F ** 0.5
        info("epoch_loss = %.16f (loss_Etot = %.16f, loss_F = %.16f, RMSE_Etot = %.16f, RMSE_Ei = %.16f, RMSE_F = %.16f)" %(loss, loss_Etot, loss_F, RMSE_Etot, RMSE_Ei, RMSE_F))
        # update tensorboard
        if ((opt_journal_cycle > 0) and ((epoch) % opt_journal_cycle == 0)):
            if (writer is not None):
                writer.add_scalar('learning_rate', lr, epoch)
                writer.add_scalar('train_loss', loss, epoch)
                writer.add_scalar('train_loss_Etot', loss_Etot, epoch)
                writer.add_scalar('train_loss_Ei', loss_Ei, epoch)
                writer.add_scalar('train_loss_F', loss_F, epoch)
                writer.add_scalar('train_RMSE_Etot', RMSE_Etot, epoch)
                writer.add_scalar('train_RMSE_Ei', RMSE_Ei, epoch)
                writer.add_scalar('train_RMSE_F', RMSE_F, epoch)

        iprint = 10 #隔几个epoch记录一次误差
        f_err_log=opt_session_dir+'loss.dat'
        if epoch // iprint == 1:
            fid_err_log = open(f_err_log, 'w')
        else:
            fid_err_log = open(f_err_log, 'a')
        fid_err_log.write('%d %e %e %e %e %e \n'%(epoch, loss, RMSE_Etot, RMSE_Ei, RMSE_F, real_lr))    

     
        if (opt_scheduler == 'OC'):
            pass
        elif (opt_scheduler == 'PLAT'):
            scheduler.step(loss)
        elif (opt_scheduler == 'LR_INC'):
            LinearLR(optimizer=optimizer, base_lr=LR_base, target_lr=opt_LR_max_lr, total_epoch=n_epoch, cur_epoch=epoch)
        elif (opt_scheduler == 'LR_DEC'):
            LinearLR(optimizer=optimizer, base_lr=LR_base, target_lr=opt_LR_min_lr, total_epoch=n_epoch, cur_epoch=epoch)
        elif (opt_scheduler == 'NONE'):
            pass
        else:
            scheduler.step()

        if opt_save_model == True: 
            state = {'model': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch, 'loss': loss}
            file_name = opt_model_dir + 'latest.pt'
            if epoch % 100 == 0:
                file_name = opt_model_dir + str(epoch) + '.pt'
            torch.save(state, file_name)

        # 打印 dict info
        # for name, parameter in model.named_parameters():
        #     print(name)
        #     print(parameter.shape)
        #     print("===========")
        #     # f_err_log=pm.dir_work+ str(name) +'.txt'
        #     # if not os.path.isfile(f_err_log):
        #     #     fid_err_log = open(f_err_log, 'w')
        #     # else:
        #     #     fid_err_log = open(f_err_log, 'a')
        # import ipdb;ipdb.set_trace()
        #     np.savetxt(fid_err_log, parameter.cpu().detach().numpy())
            

        # if ((opt_journal_cycle > 0) and ((epoch) % opt_journal_cycle == 0)):
        #     for name, parameter in model.named_parameters():
                # param_RMS= parameter.pow(2).mean().pow(0.5)
                # param_ABS= parameter.abs().mean()
                # grad_RMS= parameter.grad.pow(2).mean().pow(0.5)
                # grad_ABS= parameter.grad.abs().mean()
                # param_list = parameter.view(parameter.numel())
                # param_name = [str(x) for x in range(parameter.numel())]
                # param_dict = dict(zip(param_name, param_list))
                # grad_list = parameter.grad.view(parameter.grad.numel())
                # grad_name = [str(x) for x in range(parameter.grad.numel())]
                # grad_dict = dict(zip(grad_name, grad_list))
                # if (writer is not None):
                #     writer.add_scalar(name+'_RMS', param_RMS, epoch)
                #     writer.add_scalar(name+'_ABS', param_ABS, epoch)
                #     writer.add_scalar(name+'.grad_RMS', grad_RMS, epoch)
                #     writer.add_scalar(name+'.grad_ABS', grad_ABS, epoch)
                #     #writer.add_scalars(name, param_dict, epoch)
                #     #writer.add_scalars(name+'.grad', grad_dict, epoch)
                
                # dump("dump parameter statistics of %s -------------------------->" %name)
                # dump("%s : %s" %(name+'_RMS', param_RMS))
                # dump("%s : %s" %(name+'_ABS', param_ABS))
                # dump("%s : %s" %(name+'.grad_RMS', grad_RMS))
                # dump("%s : %s" %(name+'.grad_ABS', grad_ABS))
                # dump("dump model parameter (%s : %s) ------------------------>" %(name, parameter.size()))
                # dump(parameter)
                # dump("dump grad of model parameter (%s : %s) (not applied)--->" %(name, parameter.size()))
                # dump(parameter.grad)

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
        
            #print(epoch_loss)
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
    if (opt_wandb is True):
        wandb_run.finish()