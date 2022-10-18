#!/usr/bin/env python3

"""
    mlff.py 
    seper.py 
    gen_data.py 

    predict.py 
    L. Wang, May 2022
"""
from statistics import mode
from turtle import Turtle

import pandas as pd    
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

#print (codepath)
sys.path.append(codepath+'/pre_data')
sys.path.append(codepath+'/..')

sys.path.append(codepath+'/lib')
from read_all import read_allnn

from data_loader_2type import MovementDataset, get_torch_data
from scalers import DataScalers
from utils import get_weight_grad
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import time
import getopt
import getpass

from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from sklearn.feature_selection import VarianceThreshold


import os
import sys
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../lib')
sys.path.append(os.getcwd())

import prepare as pp 
import seper 

# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# parse optional parameters
opt_outputname = "final.config"
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
     'auto_grad=', 'dmirror=', 'dp=', "outputname="])

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

    elif opt_name in ('--outputname'):
        opt_outputname = opt_value
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

print("temp image name:"+opt_outputname)


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
#summary("")
#summary("#########################################################################################")
#summary("#            ___          __                         __      __  ___       __  ___      #")
#summary("#      |\ | |__  |  |    |__) |  | |\ | |\ | | |\ | / _`    /__`  |   /\  |__)  |       #")
#summary("#      | \| |___ |/\|    |  \ \__/ | \| | \| | | \| \__>    .__/  |  /~~\ |  \  |       #")
#summary("#                                                                                       #")
#summary("#########################################################################################")
#summary("")
#summary(' '.join(sys.argv))
#summary("")


# set default training dtype
#
# 1) dtype of model parameters during training
# 2) feature data will be casted to this dtype before using
#

if (opt_dtype == 'float64'):
    #info("Training: set default dtype to float64")
    torch.set_default_dtype(torch.float64)
elif (opt_dtype == 'float32'):
    #info("Training: set default dtype to float32")
    torch.set_default_dtype(torch.float32)
else:
    error("Training: unsupported dtype: %s" %opt_dtype)
    raise RuntimeError("Training: unsupported dtype: %s" %opt_dtype)

# set training device
if (opt_force_cpu == True):
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#info("Training: device = %s" %device)

# set random seed
#torch.manual_seed(opt_rseed)
#torch.cuda.manual_seed(opt_rseed)
#info("Training: rseed = %s" %opt_rseed)

# set print precision
torch.set_printoptions(precision = 16)

# set tensorboard

def gen_feat():

    """
    mlff.py: gen feature 
    create a tmp folder for a
    """
    #print('auto creating vdw_fitB.ntype, donot use your own vdw_fitB.ntype file')
    #print('please modify parameters.py to specify your vdw parameters')
    
    if not os.path.exists(pm.fitModelDir):
        os.makedirs(pm.fitModelDir)
    strength_rad = 0.0
    if pm.isFitVdw == True:
        strength_rad = 1.0
    vdw_input = {
        'ntypes': pm.ntypes,
        'nterms': 1,
        'atom_type': pm.atomType,
        'rad': [strength_rad for i in range(pm.ntypes)],
        'e_ave': [0.0 for i in range(pm.ntypes)],
        'wp': [ [0.0 for i in range(pm.ntypes*1)] for i in range(pm.ntypes)]
    }
    
    if hasattr(pm, 'vdwInput'):
        vdw_input = pm.vdwInput
    pp.writeVdwInput(pm.fitModelDir, vdw_input)
    
    pp.collectAllSourceFiles()
    pp.savePath()
    pp.combineMovement()
    pp.writeGenFeatInput()
    os.system('cp '+os.path.abspath(pm.fbinListPath)+' ./input/')

    calFeatGrid = False
    for i in range(pm.atomTypeNum):
        if pm.Ftype1_para['iflag_grid'][i] == 3 or pm.Ftype2_para['iflag_grid'][i] == 3:
            calFeatGrid = True
    if calFeatGrid:
        pp.calFeatGrid()
    for i in pm.use_Ftype:
        # import ipdb;ipdb.set_trace()
        command = pm.Ftype_name[i]+".x > ./output/out"+str(i)
        print(command)
        os.system(command)


def seper_dp():
    
    if not os.path.isdir(pm.dir_work):
        os.system("mkdir " + pm.dir_work)
    for dirn in [pm.d_nnEi, pm.d_nnFi]:
        if not os.path.isdir(dirn):
            os.system("mkdir " + dirn)

    seper.write_egroup_input()
    seper.run_write_egroup()
    seper.write_natoms_dfeat()
    if (pm.dR_neigh):
        seper.write_dR_neigh()



def process_data(f_train_feat, f_train_dfeat, f_train_dR_neigh,
                 f_train_natoms, f_train_egroup, nn_data_path): # f_train_ep):
    
    if (pm.feature_dtype == 'float64'):
        from convert_dfeat64 import convert_dfeat
    elif (pm.feature_dtype == 'float32'):
        from convert_dfeat import convert_dfeat
    else:
        raise RuntimeError(
        "unsupported feature_dtype: %s, check feature_dtype in your parameters.py"
        %pm.feature_dtype)

    if not os.path.exists(nn_data_path):
        os.makedirs(nn_data_path)
    # natoms contain all atomnum of each image, format: totnatom, type1n, type2 n
    natoms = np.loadtxt(f_train_natoms, dtype=int)
    natoms = np.atleast_2d(natoms)
    nImg = natoms.shape[0]
    indImg = np.zeros((nImg+1,), dtype=int)
    indImg[0] = 0
    
    for i in range(nImg):
        indImg[i+1] = indImg[i] + natoms[i, 0]
    
    # 设置打印时显示方式：输出数组的时候完全输出
    np.set_printoptions(threshold=np.inf)
    # pd.set_option('display.float_format',lambda x : '%.15f' % x)
    pd.options.display.float_format = '${:,.15f}'.format
    itypes, feat, engy = pp.r_feat_csv(f_train_feat)
    natoms_img = np.zeros((nImg, pm.ntypes + 1), dtype=np.int32)
    for i in range(nImg):
        natoms_img[i][0] = indImg[i+1] - indImg[i]
        tmp = itypes[indImg[i]:indImg[i+1]]
        mask, ind = np.unique(tmp, return_index=True)
        mask = mask[np.argsort(ind)]
        type_id = 1
        for type in mask:
            natoms_img[i][type_id] = np.sum(tmp == type)
            type_id += 1
    
    # 进行scale
    # feat_scaled = scalers.pre_feat(feat, itypes)
    # engy_scaled = scalers.pre_engy(engy, itypes)
    # 不scale
    feat_scaled = feat
    engy_scaled = engy

    egroup, divider, egroup_weight = pp.r_egroup_csv(f_train_egroup)
    if os.path.exists(os.path.join(pm.dir_work, 'weight_for_cases')):
        weight_all = pd.read_csv(os.path.join(pm.dir_work, 'weight_for_cases'),
                                 header=None, encoding= 'unicode_escape').values[:, 0].astype(pm.feature_dtype).reshape(-1, 1)
    else:
        weight_all = np.ones((engy_scaled.shape[0], 1))
    nfeat0m = feat_scaled.shape[1]  # 每个原子特征的维度
    itype_atom = np.asfortranarray(np.array(pm.atomType).transpose())  # 原子类型
    
    # feat_scale_a = np.zeros((nfeat0m, pm.ntypes))
    # for i in range(pm.ntypes):
    #     itype = pm.atomType[i]
    #     feat_scale_a[:, i] = scalers.scalers[itype].feat_scaler.a
    # feat_scale_a = np.asfortranarray(feat_scale_a)  # scaler 的 a参数

    feat_scale_a=np.ones((nfeat0m,pm.ntypes))
    feat_scale_a = np.asfortranarray(feat_scale_a)
    
    init = pm.use_Ftype[0]
    

    dfeatdirs = {}
    energy_all = {}
    force_all = {}
    num_neigh_all = {}
    list_neigh_all = {}
    iatom_all = {}
    dfeat_tmp_all = {}
    num_tmp_all = {}
    iat_tmp_all = {}
    jneigh_tmp_all = {}
    ifeat_tmp_all = {}
    nfeat = {}
    nfeat[0] = 0
    flag = 0
    
    # 读取 dfeat file
    for m in pm.use_Ftype:
        dfeatdirs[m] = np.unique(pd.read_csv(
            f_train_dfeat+str(m), header=None, encoding= 'unicode_escape').values[:, 0])
        for k in dfeatdirs[m]:
            read_allnn.read_dfeat(k, itype_atom, feat_scale_a, nfeat[flag])
            if flag == 0:
                energy_all[k] = np.array(read_allnn.energy_all).astype(pm.feature_dtype)
                force_all[k] = np.array(read_allnn.force_all).transpose(1, 0, 2).astype(pm.feature_dtype)
                list_neigh_all[k] = np.array(read_allnn.list_neigh_all).transpose(1, 0, 2).astype(int)
                iatom_all[k] = np.array(read_allnn.iatom)

            nfeat[flag+1] = np.array(read_allnn.feat_all).shape[0]
            dfeat_tmp_all[k] = np.array(read_allnn.dfeat_tmp_all).astype(pm.feature_dtype)
            num_tmp_all[k] = np.array(read_allnn.num_tmp_all).astype(int)
            iat_tmp_all[k] = np.array(read_allnn.iat_tmp_all).astype(int)
            jneigh_tmp_all[k] = np.array(read_allnn.jneigh_tmp_all).astype(int)
            ifeat_tmp_all[k] = np.array(read_allnn.ifeat_tmp_all).astype(int)
            read_allnn.deallo()
        flag = flag+1
        
    # dfeat_tmp_all_1 = []
    # for k in dfeat_tmp_all.items():
    #     dfeat_tmp_all_1.append(k)
    # dfeat = np.array(dfeat_tmp_all_1)

    # print("========dfeat_tmp_all========")
    # print(dfeat_tmp_all)

    #pm.fitModelDir=./fread_dfeat  
    with open(os.path.join(pm.fitModelDir, "feat.info"), 'w') as f:
        print(os.path.join(pm.fitModelDir, "feat.info"))
        f.writelines(str(pm.iflag_PCA)+'\n')
        f.writelines(str(len(pm.use_Ftype))+'\n')
        for m in range(len(pm.use_Ftype)):
            f.writelines(str(pm.use_Ftype[m])+'\n')

        f.writelines(str(pm.ntypes)+', '+str(pm.maxNeighborNum)+'\n')
        for i in range(pm.ntypes):
            f.writelines(str(pm.atomType[i])+'  ' +
                         str(nfeat0m)+'  '+str(nfeat0m)+'\n')
        for i in range(pm.ntypes):
            for m in range(len(pm.use_Ftype)):
                f.writelines(str(nfeat[m+1])+'  ')
            f.writelines('\n')

    dfeat_names = {}
    image_nums = {}
    pos_nums = {}
    for m in pm.use_Ftype:
        values = pd.read_csv(f_train_dfeat+str(m), header=None, encoding= 'unicode_escape').values
        dfeat_names[m] = values[:, 0]
        image_nums[m] = values[:, 1].astype(int)
        pos_nums[m] = values[:, 2].astype(int)
        nImg = image_nums[m].shape[0]

    fors_scaled = []
    nblist = []
    for ll in range(len(image_nums[init])):
        fors_scaled.append(force_all[dfeat_names[init][ll]][:, :, image_nums[init][ll]-1])
        nblist.append(list_neigh_all[dfeat_names[init][ll]][:, :, image_nums[init][ll]-1])
    fors_scaled = np.concatenate(fors_scaled, axis=0)
    nblist = np.concatenate(nblist, axis=0)
    
# ========================================================================
    img_num = indImg.shape[0] - 1
    convert_dfeat.allo(nfeat0m, indImg[-1], pm.maxNeighborNum)
    for i in range(img_num):
        dfeat_name={}
        image_num={}
        for mm in pm.use_Ftype:
            dfeat_name[mm] = dfeat_names[mm][i]
            image_num[mm] = image_nums[mm][i]
        kk=0
        for mm in pm.use_Ftype:
            dfeat_tmp=np.asfortranarray(dfeat_tmp_all[dfeat_name[mm]][:,:,image_num[mm]-1])
            jneigh_tmp=np.asfortranarray(jneigh_tmp_all[dfeat_name[mm]][:,image_num[mm]-1])
            ifeat_tmp=np.asfortranarray(ifeat_tmp_all[dfeat_name[mm]][:,image_num[mm]-1])
            iat_tmp=np.asfortranarray(iat_tmp_all[dfeat_name[mm]][:,image_num[mm]-1])
            convert_dfeat.conv_dfeat(image_num[mm],nfeat[kk],indImg[i],num_tmp_all[dfeat_name[mm]][image_num[mm]-1],dfeat_tmp,jneigh_tmp,ifeat_tmp,iat_tmp)
            kk=kk+1
    
    dfeat_scaled = np.array(convert_dfeat.dfeat).transpose(1,2,0,3).astype(pm.feature_dtype)
    # print("======dfeat_scaled=======")
    # print(dfeat_scaled)

    convert_dfeat.deallo()
    
    # neighbor 不排序
    if (pm.dR_neigh):
        dR_neigh = pd.read_csv(f_train_dR_neigh, header=None).values.reshape(indImg[-1], len(pm.atomType), pm.maxNeighborNum, 4) # 1 是 ntype
        print("dR neigh shape" + str(dR_neigh.shape))
        np.save(nn_data_path + "/dR_neigh.npy", dR_neigh)



    np.save(nn_data_path + "/feat_scaled.npy", feat_scaled)
    np.save(nn_data_path + "/fors_scaled.npy", fors_scaled)
    np.save(nn_data_path + "/nblist.npy", nblist)
    np.save(nn_data_path + "/engy_scaled.npy", engy_scaled)
    np.save(nn_data_path + "/itypes.npy", itypes)
    np.save(nn_data_path + "/egroup_weight.npy", egroup_weight)
    np.save(nn_data_path + "/weight_all.npy", weight_all)
    np.save(nn_data_path + "/egroup.npy", egroup)
    np.save(nn_data_path + "/divider.npy", divider)
    np.save(nn_data_path + "/dfeat_scaled.npy", dfeat_scaled)
    np.save(nn_data_path + "/ind_img.npy", np.array(indImg).reshape(-1))
    np.save(nn_data_path + "/natoms_img.npy", natoms_img)
    # np.save(nn_data_path + "/ep.npy", ep)

def gendata_dp():

    read_allnn.read_wp(pm.fitModelDir, pm.ntypes)
    assert(pm.test_ratio == 1)

    process_data(pm.f_test_feat,
                 pm.f_test_dfeat,
                 pm.f_test_dR_neigh,
                 pm.f_test_natoms,
                 pm.f_test_egroup,
                 pm.test_data_path)

def test(sample_batches, model, criterion):
    if (opt_dtype == 'float64'):
        Ei_label = Variable(sample_batches['output_energy'][:,:,:].double().to(device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].double().to(device))   #[40,108,3]
        Egroup_label = Variable(sample_batches['input_egroup'].double().to(device))
        input_data = Variable(sample_batches['input_feat'].double().to(device), requires_grad=True)
        dfeat = Variable(sample_batches['input_dfeat'].double().to(device))  #[40,108,100,42,3]
        egroup_weight = Variable(sample_batches['input_egroup_weight'].double().to(device))
        divider = Variable(sample_batches['input_divider'].double().to(device))
        # Ep_label = Variable(sample_batches['output_ep'][:,:,:].double().to(device))
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
        # Ep_label = Variable(sample_batches['output_ep'][:,:,:].float().to(device))
        if pm.dR_neigh:
            dR = Variable(sample_batches['input_dR'].float().to(device), requires_grad=True)
            dR_neigh_list = Variable(sample_batches['input_dR_neigh_list'].to(device))

    else:
        error("train(): unsupported opt_dtype %s" %opt_dtype)
        raise RuntimeError("train(): unsupported opt_dtype %s" %opt_dtype)

    neighbor = Variable(sample_batches['input_nblist'].int().to(device))  # [40,108,100]
    natoms_img = Variable(sample_batches['natoms_img'].int().to(device))  # [40,108,100]

    error=0
    atom_number = Ei_label.shape[1]
    Etot_label = torch.sum(Ei_label, dim=1)

    model.eval()
    
    if opt_dp:
        Etot_predict, Ei_predict, Force_predict = model(dR, dfeat, dR_neigh_list, natoms_img, egroup_weight, divider)
    else:
        Etot_predict, Ei_predict, Force_predict = model(input_data, dfeat, neighbor, natoms_img, egroup_weight, divider)
    
    print() 

    # Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider)
    loss_F = criterion(Force_predict, Force_label)
    loss_Etot = criterion(Etot_predict, Etot_label)
    loss_Ei = criterion(Ei_predict, Ei_label)
    # import ipdb;ipdb.set_trace()
    # print("valid info: force label; force predict")
    # print(Force_label)
    # print(Force_predict)
    error = float(loss_F.item()) + float(loss_Etot.item())

    f_err_log =  opt_session_dir + 'iter_loss_test.dat'
    if not os.path.exists(f_err_log):
        fid_err_log = open(f_err_log, 'w')
        fid_err_log.write('Etot_predict\t Force_predict_x\t Force_predict_y\t Force_predict_z\n')
    else:
        fid_err_log = open(f_err_log, 'a')
        fid_err_log.write('%e %e %e %e \n'%(Etot_predict.item(), Force_predict[0,0,0].item(), Force_predict[0,0,1].item(), Force_predict[0,0,2].item()))
        
    return Etot_predict, Ei_predict, Force_predict, error, loss_Etot, loss_Ei, loss_F


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
"""
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

"""
"""
    single image DP inference starts 
"""

#Clean PWdata and move final.config in 

os.system("rm " + pm.trainSetDir + "/*")
cmd = "cp atom.config "+pm.trainSetDir+"/MOVEMENT"
os.system(cmd)
    
#clean input and output
os.system("rm ./input/*")
os.system("rm ./output/*")

#exit()

gen_feat()

seper_dp()

gendata_dp()


batch_size = 1
test_data_path=pm.test_data_path
torch_test_data = get_torch_data(test_data_path)


if pm.is_scale:
    scaler=load('./scaler.pkl')
    torch_test_data.feat = scaler.transform(torch_test_data.feat)
    dfeat_tmp = torch_test_data.dfeat
    dfeat_tmp = dfeat_tmp.transpose(0, 1, 3, 2)
    dfeat_tmp = dfeat_tmp * scaler.scale_
    dfeat_tmp = dfeat_tmp.transpose(0, 1, 3, 2)
    torch_test_data.dfeat = dfeat_tmp

loader_test = Data.DataLoader(torch_test_data, batch_size=batch_size, shuffle=False)
if opt_dp:  # 此处应读training中保存的stat
    davg, dstd, ener_shift = torch_test_data.get_stat(image_num=10)
    stat = [davg, dstd, ener_shift]

# ==========================part2:load模型==========================
if opt_dp:
    model = DP(opt_net_cfg, opt_act, device, stat, opt_magic)
else:
    model = MLFFNet(device)
model.to(device)

# change the *.pt path
# path = r"/home/husiyu/software/MLFFdataset/cutest_dp/record/model/latest.pt"
# path = r"/home/husiyu/software/MLFFdataset/cutest/record/model/latest.pt"
path = r"/data/home/wlj_pwmat/MLFF_DEV/MLFF_0420/example/cu1646_kfdp/model/latest.pt"

checkpoint = torch.load(path,map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
# if opt_follow_mode==True:
#     checkpoint = torch.load(opt_model_file,map_location=device)
#     model.load_state_dict(checkpoint['model'],strict=False)

# if (opt_recover_mode == True):
#     if (opt_session_name == ''):
#         raise RuntimeError("you must run follow-mode from an existing session")
#     opt_latest_file = opt_model_dir+'latest.pt'
#     checkpoint = torch.load(opt_latest_file,map_location=device)
#     model.load_state_dict(checkpoint['model'])
#     # optimizer.load_state_dict(checkpoint['optimizer'])

nr_total_sample = 0
test_loss = 0.
test_loss_Etot = 0.
test_loss_Ei = 0.
test_loss_F = 0.


for i_batch, sample_batches in enumerate(loader_test):

    if i_batch > 0:
        break 
    
    natoms_sum = sample_batches['natoms_img'][0, 0].item()
    nr_batch_sample = sample_batches['input_feat'].shape[0]

    etot_pred,ei_pred,force_pred,test_error_iter, batch_loss_Etot, batch_loss_Ei, batch_loss_F = test(sample_batches, model, nn.MSELoss())
    # n_iter = (epoch - 1) * len(loader_valid) + i_batch + 1
    
    """
        write files 
    """

    f_etot = "etot.tmp"
    f_ei   = "ei.tmp"
    f_force = "force.tmp"
    
    with open(f_etot,"w") as f:
        f.writelines(str(float(etot_pred[0]))+"\n")

    with open(f_ei,"w") as f:
        size = len(ei_pred[0])
        for i in range(size):
            f.writelines(str(float(ei_pred[0][i]))+"\n")

    with open(f_force,"w") as f:
        size = len(force_pred[0])
        for i in range(size):
            str_tmp = str(float(force_pred[0][i][0])) + ' ' + str(float(force_pred[0][i][1])) + ' ' + str(float(force_pred[0][i][2])) 

            f.writelines(str_tmp+"\n")

    #print (etot_pred)
    #print (ei_pred)
    #print (force_pred)
