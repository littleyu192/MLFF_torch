#
# stacker: a configurable network model framework, which treats
# a network model as the stack of fundmental network blocks, both
# the network blocks and the whole network model can be described
# by specified configurations
#
# network block description examples:
# 
#       # a plain res-net block with one weight layer, out_dim = 128
#       #
#       res_block_plain = [
#           (input, auto,),         # layer0: input, width=auto
#           (linear, 128, False,),  # layer1: linear_1, width=128, bias=False
#           (batch_norm,),          # layer2: batch_norm_2
#           (func, tanh,),          # layer3: tanh_3, tanh activation
#           (shortcut, 0, True),    # layer4: shortcut_4, shortcut from layer0,
#                                   # use projection if dimension mismatch=True
#       ]
#
#       # a bottleneck building block just like ResNet v1, out_dim = 256 
#       #
#       res_block_bottleneck = [
#           (input, auto,),         # layer0: input, width=auto
#           (linear, 64, False,),   # layer1: linear_1, width=64, bias=False
#           (batch_norm,),          # layer2: batch_norm_2
#           (func, relu,),          # layer3: relu_3, relu activation
#           (linear, 64, False,),   # layer4: linear_4, width=64, bias=False
#           (batch_norm,),          # layer5: batch_norm_5
#           (func, relu,),          # layer6: relu_6, relu activation
#           (linear, 256, False),   # layer7: linear_7, width=256, bias=False
#           (batch_norm,),          # layer8: batch_norm_8
#           (shortcut, 0, True),    # layer9: shortcut_9, shortcut from layer0
#           (func, relu),           # layer10: relu_10, relu activation
#       ]
#
#       # a two-layer res-net block with full pre-activation just like ResNet v2
#       #
#       res_block_FPA = [           
#           (input, auto,),         # layer0: input, width=auto
#           (batch_norm,),          # layer1: batch_norm_1
#           (func, relu,),          # layer2: relu_2, relu activation
#           (linear, 128, False,),  # layer3: linear_3, width=128, bias=False
#           (batch_norm,),          # layer4:
#           (func, relu,),
#           (linear, 128, False,),
#           (shortcut, 0, True,),
#       ]

#                           (layer_norm
# implemented fundmental network blocks:
# 
# resnet_linear
# block_resnext

# basic layers supported by stacker is defined and implemented  here
#
# Basic dmirror layers:
#
#   dmirror_FC(self, cfg, act_func, d_act_func)
#       * cfg describes the base part of a dmirror style network,
#         the whole network (including the mirrored part) will be
#         automaticlly generated. An example:
#
#           cfg = [
#                   (layer_norm,),                   # layer1: layer_norm_1
#                   (linear, 16, 32, True),     # layer: linear_1, bias=True
#                   (func, tanh,),              # layer: tanh activation
#                   (linear, 32, 64, True),     # layer: linear_2, bias=True
#                   (func, tanh,),              # layer: tanh activation
#                   (linear, 64, 8, False),     # layer: linear3, bias=False
#                   (func, tanh,),              # layer: activation_3
#                   (linear, 8, 1, False),      # layer: linear4, bias=False
#                   (func, linear,),
#           ]
#

import torch
import torch.nn as nn
import collections

# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup module logger
import component.logger as mlff_logger
logger = mlff_logger.get_module_logger('stacker')
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

# dmirror implementation
#
class stacker_block(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, magic=False):
        super(dmirror_linear, self).__init__()
        self.bias = None

        # for pesudo random number generator for float64/float32 precision test
        self.rand_a = 25214903917
        self.rand_c = 11
        self.rand_p = 2021

        if (magic == False):
            self.weight = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=True)
            if (bias == True):
                self.bias = nn.Parameter(torch.randn(out_dim), requires_grad=True)
        else:
            warmup_my_rand = self.my_rand_2d(out_dim, in_dim)
            self.weight = nn.Parameter(self.my_rand_2d(out_dim, in_dim), requires_grad=True)
            if (bias == True):
                self.bias = nn.Parameter(self.my_rand_1d(out_dim), requires_grad=True)

    # random number generator, maybe their better place is train.py
    def my_rand_core(self):
        r = (self.rand_a * self.rand_p + self.rand_c) % 10000
        self.rand_p = r
        return r

    def my_rand_2d(self, m, n):
        res = torch.randn(m, n)
        for i in range(m):
            for j in range(n):
                res[i, j] = float(self.my_rand_core() / 10000.0)
        return res

    def my_rand_1d(self, m):
        res = torch.randn(m)
        for i in range(m):
            res[i] = float(self.my_rand_core() / 10000.0)
        return res

    def forward(self, x):
        if (self.bias is not None):
            return torch.matmul(x, self.weight.t()) + self.bias
        else:
            return torch.matmul(x, self.weight.t())

    def forward_r(self, x):
        return torch.matmul(x, self.weight)


class dmirror_activation(nn.Module):
    def __init__(self, func, d_func):
        super(dmirror_activation, self).__init__()
        self.func = func
        self.d_func = d_func
        self.k = torch.tensor([])

    def forward(self, x):
        self.k = x
        return self.func(x)

    def forward_r(self, x):
        return x * self.d_func(self.k)


class MLFF_network_block(nn.Module):
    def __init__(self, cfg):
        super(MLFF_network_block, self).__init__()
        self.cfg = cfg
        self.layers = []

        # parse cfg & generating layers
        #
        idx_linear = 1
        idx_activation = 1
        for idx, item in enumerate(self.cfg):
            layer_type = item[0]
            if (layer_type == 'linear'):
                in_dim = item[1]
                out_dim = item[2]
                bias = item[3]
                self.layers.append((
                    'dmirror_linear_'+str(idx_linear),
                    dmirror_linear(in_dim, out_dim, bias, magic)
                ))
                idx_linear += 1
            elif (layer_type == 'activation'):
                self.layers.append((
                    'dmirror_activation_'+str(idx_activation),
                    dmirror_activation(act_func, d_act_func)
                ))
                idx_activation += 1
            elif (layer_type == 'scale'):
                raise RuntimeError(
                    "Notimplemented for layer_type = %s" %(layer_type)
                )
            else:
                raise ValueError(
                    "Invalid for layer_type = %s" %(layer_type)
                )
        self.layers_r = list(reversed(self.layers))

        # the layer parameters will be registered to nn Module,
        # so optimizer can update the layer parameters.
        #
        self.base_net = nn.Sequential(
            collections.OrderedDict(self.layers)
        )
        self.mirror_net = nn.Sequential(
            collections.OrderedDict(self.layers_r)
        )
        info("dmirror_FC: start of network instance dump ==============>")
        info("<----------------------- base_net ----------------------->")
        info(self.base_net)
        info("<---------------------- mirror_net ---------------------->")
        info(self.mirror_net)
        info("dmirror_FC: end of network instance dump ================>")

    # we can't call forward() of sequentialized module, since
    # we extened the param list of the layers' forward()
    #
    def forward(self, x):
        for name, obj in (self.layers):
            x = obj.forward(x)
        res0 = x

        x = torch.ones_like(res0)
        for name, obj in (self.layers_r):
            x = obj.forward_r(x)
        res1 = x

        return res0, res1

