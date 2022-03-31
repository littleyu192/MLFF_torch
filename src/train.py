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
import time
import getopt
import getpass

# setup mlff runtime option
# see component/option.py for runtime option details
import component.option as option
opt = option.mlff_runtime_option()
opt.parse(sys.argv)

# setup mlff logger
import component.logger as mlff_logger
mlff_logger.init_mlff_logger(opt.log_level, opt.file_log_level, opt.logging_file)

# setup module logger
logger = mlff_logger.get_module_logger('train')
def dump(msg, *args, **kwargs):
    logger.log(mlff_logger.DUMP, msg, *args, **kwargs)
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)
def summary(msg, *args, **kwargs):
    logger.log(mlff_logger.SUMMARY, msg, *args, **kwargs)
def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)
def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs, exc_info=True)

# show start logging banner to logging file
#
info("New running starts ===>")
info("codebase git revision: %s" %opt.git_revision)
info("commandline: %s" %opt.commandline)

# set default training dtype
#
# 1) dtype of model parameters during training
# 2) feature data will be casted to this dtype before using
#
if (opt.dtype == 'float64'):
    info("Training: set default dtype to float64")
    torch.set_default_dtype(torch.float64)
elif (opt.dtype == 'float32'):
    info("Training: set default dtype to float32")
    torch.set_default_dtype(torch.float32)
else:
    error("Training: unsupported dtype: %s" %opt.dtype)
    raise RuntimeError("Training: unsupported dtype: %s" %opt.dtype)

# set training device
if (opt.force_cpu == True):
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
info("Training: device = %s" %device)

# set random seed
torch.manual_seed(opt.rseed)
torch.cuda.manual_seed(opt.rseed)
info("Training: rseed = %s" %opt.rseed)

# set print precision
torch.set_printoptions(precision = 16)

# set tensorboard
if (opt.tensorboard_dir != ''):
    if (opt.wandb is True):
        wandb.tensorboard.patch(root_logdir=opt.tensorboard_dir)
        wandb_run = wandb.init(entity=opt.wandb_entity, project=opt.wandb_project, reinit=True)
        wandb_run.name = getpass.getuser()+'-'+opt.session_name+'-'+opt.run_id
        wandb_run.save()
    writer = SummaryWriter(opt.tensorboard_dir)
else:
    writer = None


def train(sample_batches, model, optimizer, criterion, last_epoch):
    # floating part of sample_batches, cast to specified opt.dtype
    #
    if (opt.dtype == 'float64'):
        Etot_label = Variable(sample_batches['output_energy'][:,:,:].double().to(device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].double().to(device))   #[40,108,3]
        Egroup_label = Variable(sample_batches['input_egroup'].double().to(device))
        input_data = Variable(sample_batches['input_feat'].double().to(device), requires_grad=True)
        dfeat = Variable(sample_batches['input_dfeat'].double().to(device))  #[40,108,100,42,3]
        egroup_weight = Variable(sample_batches['input_egroup_weight'].double().to(device))
        divider = Variable(sample_batches['input_divider'].double().to(device))
    elif (opt.dtype == 'float32'):
        Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
        Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))   #[40,108,3]
        Egroup_label = Variable(sample_batches['input_egroup'].float().to(device))
        input_data = Variable(sample_batches['input_feat'].float().to(device), requires_grad=True)
        dfeat = Variable(sample_batches['input_dfeat'].float().to(device))  #[40,108,100,42,3]
        egroup_weight = Variable(sample_batches['input_egroup_weight'].float().to(device))
        divider = Variable(sample_batches['input_divider'].float().to(device))
    else:
        error("train(): unsupported opt.dtype %s" %opt.dtype)
        raise RuntimeError("train(): unsupported opt.dtype %s" %opt.dtype)

    # non-floating or derived part of sample_batches
    #
    atom_number = Etot_label.shape[1]
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1]
    neighbor = Variable(sample_batches['input_nblist'].int().to(device))  # [40,108,100]
    ind_img = Variable(sample_batches['ind_image'].int().to(device))

    # dump interested input data
    #
    dump("defat.shape= %s" %(dfeat.shape,))
    dump("neighbor.shape = %s" %(neighbor.shape,))
    dump("dump dfeat ------------------->")
    dump(dfeat)
    dump("dump neighbor ------------------->")
    dump(neighbor)

    # do forward predict
    #
    Etot_predict, Force_predict = model(input_data, dfeat, neighbor, egroup_weight, divider)
    # Egroup_predict = model.get_egroup(Ei_predict, egroup_weight, divider)   #[40,108,1]
    
    # dump predictioin result
    #
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

    # calculate batch loss
    #
    loss = pm.rtLossF * criterion(Force_predict, Force_label) + pm.rtLossEtot * criterion(Etot_predict, Etot_label)
    loss_F = criterion(Force_predict, Force_label)
    loss_Etot = criterion(Etot_predict, Etot_label)
    loss_Egroup = 0
    debug("batch_loss = %.16f (loss_etot = %.16f, loss_force = %.16f, RMSE_etot = %.16f, RMSE_force = %.16f)" %(loss, loss_Etot, loss_F, loss_Etot ** 0.5, loss_F ** 0.5))

    # do backward optimize
    #
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # return batch loss
    #
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

# ==========================part2:指定模型参数==========================
# ==========================part1:数据读取==========================

momentum = opt.momentum
REGULAR_wd = opt.regular_wd
n_epoch = opt.epochs
LR_base = opt.lr
LR_gamma = opt.gamma
LR_step = opt.step
batch_size = opt.batch_size 

#if (opt.follow_mode == True):
#    opt.model_file = opt.model_dir+opt.net_cfg+'.pt'

info("Training: session = %s" %opt.session_name)
info("Training: run_id = %s" %opt.run_id)
info("Training: journal_cycle = %d" %opt.journal_cycle)
info("Training: follow_mode = %s" %opt.follow_mode)
info("Training: recover_mode = %s" %opt.recover_mode)
info("Training: network = %s" %opt.net_cfg)
info("Training: model_dir = %s" %opt.model_dir)
info("Training: model_file = %s" %opt.model_file)
info("Training: activation = %s" %opt.act)
info("Training: optimizer = %s" %opt.optimizer)
info("Training: momentum = %.16f" %momentum)
info("Training: REGULAR_wd = %.16f" %REGULAR_wd)
info("Training: scheduler = %s" %opt.scheduler)
info("Training: n_epoch = %d" %n_epoch)
info("Training: LR_base = %.16f" %LR_base)
info("Training: LR_gamma = %.16f" %LR_gamma)
info("Training: LR_step = %d" %LR_step)
info("Training: batch_size = %d" %batch_size)

# scheduler specific options
info("Scheduler: opt.LR_milestones = %s" %opt.LR_milestones)
info("Scheduler: opt.LR_patience = %s" %opt.LR_patience)
info("Scheduler: opt.LR_cooldown = %s" %opt.LR_cooldown)
info("Scheduler: opt.LR_total_steps = %s" %opt.LR_total_steps)
info("Scheduler: opt.LR_max_lr = %s" %opt.LR_max_lr)
info("Scheduler: opt.LR_min_lr = %s" %opt.LR_min_lr)
info("Scheduler: opt.LR_T_max = %s" %opt.LR_T_max)

train_data_path=pm.train_data_path
torch_train_data = get_torch_data(pm.natoms, train_data_path)
if (opt.shuffle_data == True):
    loader_train = Data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=True)
else:
    loader_train = Data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=False)

valid_data_path=pm.test_data_path
torch_valid_data = get_torch_data(pm.natoms, valid_data_path)
loader_valid = Data.DataLoader(torch_valid_data, batch_size=1, shuffle=False)

# if torch.cuda.device_count() > 1:
    # model = nn.DataParallel(model)

# ==========================part3:模型预训练==========================
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23457', rank=0, world_size=1)


# ==========================part4:模型finetuning==========================

# implement a linear scheduler
def LinearLR(optimizer, base_lr, target_lr, total_epoch, cur_epoch):
    lr = base_lr - (base_lr - target_lr) * (float(cur_epoch) / float(total_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

start_epoch=1
if pm.isNNfinetuning == True:
    model = MLFF_dmirror(opt.net_cfg, opt.act, device, opt.magic)
    model.to(device)
    if opt.follow_mode==True:
        pass
        #checkpoint = torch.load(opt.model_file,map_location=device)
        #model.load_state_dict(checkpoint['model'],strict=False)
        
    # this is a temp fix for a quick test
    if (opt.init_b == True):
        for name, p in model.named_parameters():
            if ('linear_3.bias' in name):
                dump(p)
                p.data.fill_(166.0)
                dump(p)


    data_scalers = DataScalers(f_ds=pm.f_data_scaler, f_feat=pm.f_train_feat, load=True)
    if (opt.recover_mode == True):
        if (opt.session_name == ''):
            raise RuntimeError("you must run follow-mode from an existing session")
        #opt.latest_file = opt.model_dir+'latest.pt'
        #checkpoint = torch.load(opt.latest_file,map_location=device)
        #model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        #start_epoch=checkpoint['epoch'] + 1

        # TODO: clean the codes above
        #       1) need to fix opt.net_cfg, the model still need to specify in follow-mode
        #       2) add opt.image_file and it's parameter form
        #       3) model store/load need to handle cpu/gpu
        #       4) handle tensorboard file, can we modify accroding to 'epoch'?

    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)

    # set model parameter properties, do not apply weight decay to bias parameter
    # except for LBFGS, which do not support pre-parameter options
    model_parameters = [
                    {'params': (p for name, p in model.named_parameters() if 'bias' not in name)},
                    {'params': (p for name, p in model.named_parameters() if 'bias' in name), 'weight_decay': 0.}]


    if (opt.optimizer == 'SGD'):
        optimizer = optim.SGD(model_parameters, lr=LR_base, momentum=momentum, weight_decay=REGULAR_wd)
    elif (opt.optimizer == 'ASGD'):
        optimizer = optim.ASGD(model_parameters, lr=LR_base, weight_decay=REGULAR_wd)
    elif (opt.optimizer == 'RPROP'):
        optimizer = optim.Rprop(model_parameters, lr=LR_base)
    elif (opt.optimizer == 'RMSPROP'):
        optimizer = optim.RMSprop(model_parameters, lr=LR_base, weight_decay=REGULAR_wd, momentum=momentum)
    elif (opt.optimizer == 'ADAG'):
        optimizer = optim.Adagrad(model_parameters, lr=LR_base, weight_decay=REGULAR_wd)
    elif (opt.optimizer == 'ADAD'):
        optimizer = optim.Adadelta(model_parameters, lr=LR_base, weight_decay=REGULAR_wd)
    elif (opt.optimizer == 'ADAM'):
        optimizer = optim.Adam(model_parameters, lr=LR_base, weight_decay = REGULAR_wd)
    elif (opt.optimizer == 'ADAMW'):
        optimizer = optim.AdamW(model_parameters, lr=LR_base, weight_decay = REGULAR_wd)
    elif (opt.optimizer == 'ADAMAX'):
        optimizer = optim.Adamax(model_parameters, lr=LR_base, weight_decay=REGULAR_wd)
    elif (opt.optimizer == 'LBFGS'):
        optimizer = optim.LBFGS(model.parameters(), lr=LR_base)
    else:
        error("unsupported optimizer: %s" %opt.optimizer)
        raise RuntimeError("unsupported optimizer: %s" %opt.optimizer)
    if (opt.recover_mode == True):
        optimizer.load_state_dict(checkpoint['optimizer'])

    # TODO: LBFGS is not done yet
    # FIXME: train process should be better re-arranged to 
    #        support this closure cleanly
    # example code for LBFGS closure()
    #def lbfgs_closure():
    #    optimizer.zero_grad()
    #    output = model(input)
    #    loss = loss_fn(output, target)
    #    loss.backward()
    #    return loss
    #optimizer.step(lbfgs_closure)

    # user specific LambdaLR lambda function
    lr_lambda = lambda epoch: LR_gamma ** epoch

    if (opt.scheduler == 'LAMBDA'):
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif (opt.scheduler == 'STEP'):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_step, gamma=LR_gamma)
    elif (opt.scheduler == 'MSTEP'):
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.LR_milestones, gamma=LR_gamma)
    elif (opt.scheduler == 'EXP'):
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_gamma)
    elif (opt.scheduler == 'COS'):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.LR_T_max, eta_min=opt.LR_min_lr)
    elif (opt.scheduler == 'PLAT'):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=LR_gamma, 
                        patience=opt.LR_patience, cooldown=opt.LR_cooldown, min_lr=opt.LR_min_lr)
    elif (opt.scheduler == 'OC'):
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.LR_max_lr, total_steps=opt.LR_total_steps)
    elif (opt.scheduler == 'LR_INC'):
        # do nothing, will direct call LR scheduler at each epoch
        pass
    elif (opt.scheduler == 'LR_DEC'):
        # do nothing, will direct call LR scheduler at each epoch
        pass
    elif (opt.scheduler == 'NONE'):
        pass
    else:
        error("unsupported scheduler: %s" %opt.schedler)
        raise RuntimeError("unsupported scheduler: %s" %opt.scheduler)

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

            # OneCycleLR scheduler steps() at each batch
            if (opt.scheduler == 'OC'):
                scheduler.step()

        # epoch loss update
        loss /= nr_total_sample
        loss_Etot /= nr_total_sample
        loss_Egroup /= nr_total_sample
        loss_F /= nr_total_sample
        RMSE_Etot = loss_Etot ** 0.5
        RMSE_Egroup = loss_Egroup ** 0.5
        RMSE_F = loss_F ** 0.5
        info("epoch_loss = %.16f (loss_Etot = %.16f, loss_F = %.16f, RMSE_Etot = %.16f, RMSE_F = %.16f)" %(loss, loss_Etot, loss_F, RMSE_Etot, RMSE_F))
        # update tensorboard
        if ((opt.journal_cycle > 0) and ((epoch) % opt.journal_cycle == 0)):
            if (writer is not None):
                writer.add_scalar('learning_rate', lr, epoch)
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

        if (opt.scheduler == 'OC'):
            pass
        elif (opt.scheduler == 'PLAT'):
            scheduler.step(loss)
        elif (opt.scheduler == 'LR_INC'):
            LinearLR(optimizer=optimizer, base_lr=LR_base, target_lr=opt.LR_max_lr, total_epoch=n_epoch, cur_epoch=epoch)
        elif (opt.scheduler == 'LR_DEC'):
            LinearLR(optimizer=optimizer, base_lr=LR_base, target_lr=opt.LR_min_lr, total_epoch=n_epoch, cur_epoch=epoch)
        elif (opt.scheduler == 'NONE'):
            pass
        else:
            scheduler.step()

        if opt.save_model == True: 
            state = {'model': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch, 'loss': loss}
            file_name = opt.model_dir + 'latest.pt'
            if epoch % 10000 == 0:
                file_name = opt.model_dir + str(epoch) + '.pt'
            torch.save(state, file_name)

        
        if ((opt.journal_cycle > 0) and ((epoch) % opt.journal_cycle == 0)):
            for name, parameter in model.named_parameters():
                param_RMS= parameter.pow(2).mean().pow(0.5)
                param_ABS= parameter.abs().mean()
                grad_RMS= parameter.grad.pow(2).mean().pow(0.5)
                grad_ABS= parameter.grad.abs().mean()
                param_list = parameter.view(parameter.numel())
                param_name = [str(x) for x in range(parameter.numel())]
                param_dict = dict(zip(param_name, param_list))
                grad_list = parameter.grad.view(parameter.grad.numel())
                grad_name = [str(x) for x in range(parameter.grad.numel())]
                grad_dict = dict(zip(grad_name, grad_list))
                if (writer is not None):
                    writer.add_scalar(name+'_RMS', param_RMS, epoch)
                    writer.add_scalar(name+'_ABS', param_ABS, epoch)
                    writer.add_scalar(name+'.grad_RMS', grad_RMS, epoch)
                    writer.add_scalar(name+'.grad_ABS', grad_ABS, epoch)
                    #writer.add_scalars(name, param_dict, epoch)
                    #writer.add_scalars(name+'.grad', grad_dict, epoch)
                
                dump("dump parameter statistics of %s -------------------------->" %name)
                dump("%s : %s" %(name+'_RMS', param_RMS))
                dump("%s : %s" %(name+'_ABS', param_ABS))
                dump("%s : %s" %(name+'.grad_RMS', grad_RMS))
                dump("%s : %s" %(name+'.grad_ABS', grad_ABS))
                dump("dump model parameter (%s : %s) ------------------------>" %(name, parameter.size()))
                dump(parameter)
                dump("dump grad of model parameter (%s : %s) (not applied)--->" %(name, parameter.size()))
                dump(parameter.grad)

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
    if (opt.wandb is True):
        wandb_run.finish()
