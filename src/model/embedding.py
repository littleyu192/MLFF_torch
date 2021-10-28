import torch
import torch.nn as nn
import collections

# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.DeepMD')

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

class EmbedingNet(nn.Module):
    def __init__(self, cfg, magic=False):
        super(EmbedingNet, self).__init__()
        self.cfg = cfg
        self.weights = nn.ParameterDict()
        if cfg['bias']:
            self.bias = nn.ParameterDict()
        if self.cfg['resnet_dt']:
                self.resnet_dt = nn.ParameterDict()

        self.network_size = [1] + self.cfg['network_size']

        for i in range(1, len(self.network_size)):
            self.weights["weight" + str(i-1)] = nn.Parameter(torch.randn(self.network_size[i-1], self.network_size[i])) 
            if self.cfg['bias']:
                self.bias["bias" + str(i-1)] = nn.Parameter(torch.randn(1, self.network_size[i])) 
            if self.cfg['resnet_dt']:
                self.resnet_dt["resnet_dt" + str(i-1)] = nn.Parameter(torch.randn(1, self.network_size[i]))

    def forward(self, x):
        for i in range(1, len(self.network_size)):
            if self.cfg['bias']:
                hiden = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
            else:
                hiden = torch.matmul(x, self.weights['weight' + str(i-1)])
            
            hiden = self.cfg['activation'](hiden)
            if self.network_size[i] == self.network_size[i-1]:
                if self.cfg['resnet_dt']:
                    x += hiden * self.resnet_dt['resnet_dt' + str(i-1)]
                else:
                    x += hiden
            elif self.network_size[i] == 2 * self.network_size[i-1]:
                if self.cfg['resnet_dt']:
                    x = torch.cat((x, x), dim=-1)  + hiden * self.resnet_dt['resnet_dt' + str(i-1)]
                else:
                    x = torch.cat((x, x), dim=-1)  + hiden
            else:
                x = hiden
        return x

class FittingNet(nn.Module):
    def __init__(self, cfg, input_dim, magic=False):
        super(FittingNet, self).__init__()
        self.cfg = cfg
        self.weights = nn.ParameterDict()
        if cfg['bias']:
            self.bias = nn.ParameterDict()
        self.network_size = [input_dim] + self.cfg['network_size']

        for i in range(1, len(self.network_size)):
            self.weights["weight" + str(i-1)] = nn.Parameter(torch.randn(self.network_size[i-1], self.network_size[i])) 
            if self.cfg['bias']:
                self.bias["bias" + str(i-1)] = nn.Parameter(torch.randn(1, self.network_size[i])) 

    def forward(self, x):
        for i in range(1, len(self.network_size) - 1):
            if self.cfg['bias']:
                x = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
            else:
                x = torch.matmul(x, self.weights['weight' + str(i-1)])
            
            x = self.cfg['activation'](x)
        i = len(self.network_size) - 1
        if self.cfg['bias']:
            x = torch.matmul(x, self.weights['weight' + str(i-1)]) + self.bias['bias' + str(i-1)]
        else:
            x = torch.matmul(x, self.weights['weight' + str(i-1)]) 
        return x