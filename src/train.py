
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

def train(sample_batches, model, optimizers, criterion):
    error=0
    Etot_label = Variable(sample_batches['output_energy'][:,:,:].float().to(device))
    Etot_label = torch.sum(Etot_label, dim=1)   #[40,108,1]-->[40,1]
    Force_label = Variable(sample_batches['output_force'][:,:,:].float().to(device))   #[40,108,3]
  
    input_data = Variable(sample_batches['input_feat'].float().to(device), requires_grad=True)
    neighbor = Variable(sample_batches['input_nblist'].int().to(device))  # [40,108,100]
    dfeat=Variable(sample_batches['input_dfeat'].float().to(device))  #[40,108,100,42,3]
    label = Variable(sample_batches['output_energy'].float().to(device))
    model.to(device)
    model.train()
    force_predict, Etot_predict, Ei_predict = model(input_data, dfeat, neighbor)
    
    Etot_deviation = Etot_predict- Etot_label     # [40,1]
    Etot_square_deviation = Etot_deviation ** 2
    Etot_shape = Etot_label.shape[0]  #40
    Etot_ABS_error = Etot_deviation.norm(1) / Etot_shape
    Etot_RMSE_error = math.sqrt(1/Etot_shape) * Etot_deviation.norm(2)

    Force_deviation = force_predict - Force_label
    Force_square_deviation = Force_deviation ** 2
    Force_shape = Force_deviation.shape[0] * Force_deviation.shape[1] * Force_deviation.shape[2]   #40*108*3
    Force_ABS_error = Force_deviation.norm(1) / Force_shape
    Force_RMSE_error = math.sqrt(1/Force_shape) * Force_deviation.norm(2)
        
    optimizer.zero_grad()
    loss = torch.sum(Force_square_deviation) + torch.sum(Etot_square_deviation)
    loss.backward()
    optimizer.step()
    error = error+float(loss.item())
    return error, Force_RMSE_error, Force_ABS_error, Etot_RMSE_error, Etot_ABS_error

def sec_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)



# ==========================part1:数据读取==========================
batch_size = pm.batch_size   #40
train_data_path=pm.train_data_path
train_data_file_frompwmat = pm.train_data_path + './train_data.csv'
torch_train_data = get_torch_data(pm.natoms, train_data_path, train_data_file_frompwmat)
loader_train = Data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=True)


# ==========================part2:指定模型参数==========================
n_epoch = 2000
learning_rate = 0.1
weight_decay = 0.9
weight_decay_epoch = 50
direc = './FC3model_minimize_Etot'
if not os.path.exists(direc):
    os.makedirs(direc)
model = MLFFNet()
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optimizer
start = time.time()
min_loss = np.inf
start_epoch=1

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
    if epoch > weight_decay_epoch:   # 学习率衰减
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    
    train_epoch_force_RMSE_loss = 0
    train_epoch_etot_RMSE_loss = 0
    j = 0
    for i_batch, sample_batches in enumerate(loader_train):
        square_error, Force_RMSE_error, Force_ABS_error, Etot_RMSE_error, Etot_ABS_error = train(sample_batches, model, optimizer, nn.MSELoss())
        # Log train/loss to TensorBoard at every iteration
        # n_iter = (epoch - 1) * len(loader_train) + i_batch + 1
        # writer.add_scalar('Train/loss', Force_RMSE_error, n_iter)
        train_epoch_force_RMSE_loss += Force_RMSE_error
        train_epoch_etot_RMSE_loss += Etot_RMSE_error
    train_force_loss = train_epoch_force_RMSE_loss/len(loader_train)
    train_etot_loss = train_epoch_etot_RMSE_loss/len(loader_train)
    train_total_loss = train_force_loss + train_etot_loss
    end = time.time()
    time_cost = sec_to_hms(int(end-start))    #每个epoch的训练时间
    print('epoch = {}, training force RMSE loss = {:.8f}, training etot RMSE loss = {:.8f}, lr = {}, time cost = {}'.format(epoch, train_force_loss, train_etot_loss, lr, time_cost)) 

    iprint = 1 #隔几个epoch记录一次误差
    f_err_log=pm.dir_work+'out_train_err.dat'
    if epoch // iprint == 1:
        fid_err_log = open(f_err_log, 'w')
    else:
        fid_err_log = open(f_err_log, 'a')
    fid_err_log.write('%d %e %e %e %e %s\n' % (epoch, train_force_loss, train_etot_loss, train_total_loss, lr, time_cost))
    fid_err_log.close()
  
    if pm.flag_plt:
        line_train_force.set_xdata(np.append(line_train_force.get_xdata(),epoch))
        line_train_force.set_ydata(np.append(line_train_force.get_ydata(),train_force_loss))
        line_train_etot.set_xdata(np.append(line_train_etot.get_xdata(),epoch))
        line_train_etot.set_ydata(np.append(line_train_etot.get_ydata(),train_force_loss))
        line_train_total.set_xdata(np.append(line_train_total.get_xdata(),epoch))
        line_train_total.set_ydata(np.append(line_train_total.get_ydata(),train_total_loss))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    if train_total_loss < min_loss:
        min_loss = train_total_loss
        name = direc + '/3layers_' + 'MLFFNet_' + str(epoch)+'epoch.pt'
        state = {'model': model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, name)
        print('saving model to {}'.format(name))
# writer.close()