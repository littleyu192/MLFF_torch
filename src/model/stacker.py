#
# stacker: a configurable network model framework, which treats
# a network model as the stack of network blocks, and a network
# block is a stack of fundmental layers or fundmental blocks,
# so the whole network model can be described by a hierarchical
# of network block descriptors
#
# network block description examples:
# 
#       # a plain residual block with one weight layer
#       #
#       block_res_plain_1 = [
#           (width, 128, 128, 128, 128, 128,), # default out_width of each layer
#                                              # can be overwritten when calling
#           (input,),               # layer0: input, a virtual layer
#           (linear, False,),       # layer1: L1_linear, bias=False
#           (batch_norm,),          # layer2: L2_batch_norm
#           (func, tanh,),          # layer3: L3_func_tanh, tanh activation
#           (shortcut, 0, True,),   # layer4: L4_shortcut_L0, shortcut from layer0,
#                                   # use projection if dimension mismatch=True
#       ]
#
#       # a bottleneck residual block just like ResNet v1
#       #
#       block_res_bottleneck = [
#           (width, 256, 64, 64, 64, 64, 64, 64, 256, 256, 256, 256,),
#           (input,),               # layer0: input, a virtual layer
#           (linear, False,),       # layer1: L1_linear, bias=False
#           (batch_norm,),          # layer2: L2_batch_norm
#           (func, relu,),          # layer3: L3_func_relu, relu activation
#           (linear, False,),       # layer4: L4_linear, bias=False
#           (batch_norm,),          # layer5: L5_batch_norm
#           (func, relu,),          # layer6: L6_func_relu, relu activation
#           (linear, False,),       # layer7: L7_linear, bias=False
#           (batch_norm,),          # layer8: L8_batch_norm
#           (shortcut, 0, True,),   # layer9: L9_shortcut_L0, shortcut from layer0,
#                                   # use projection if dimension mismatch=True
#           (func, relu,),          # layer10: L10_func_relu, relu activation
#       ]
#
#       # a two-layer full pre-activation residual block just like ResNet v2
#       #
#       block_res_FPA = [           
#           (width, 128, 128, 128, 128, 128, 128, 128, 128,),
#           (input,),               # layer0: input, a virtual layer
#           (batch_norm,),          # layer1: L1_batch_norm
#           (func, relu,),          # layer2: L2_func_relu, relu activation
#           (linear, False,),       # layer3: L3_linear, bias=False
#           (batch_norm,),          # layer4: L4_batch_norm
#           (func, relu,),          # layer5: L5_func_relu, relu activation
#           (linear, True,),        # layer6: L6_linear, bias=True
#           (shortcut, 0, True,),   # layer7: L7_shortcut_L0, shortcut from layer0
#                                   # use projection if dimension mismatch=True
#       ]
#
#       # an aggregated residual transformations block just like ResNeXt
#       block_res_ART = [
#           # default out_width and group for each layer
#           # can be overwritten when calling
#           (width, 256, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256,),
#           (group,   1,   1,   1,   1,  32,   1,   1,   1,   1,   1,   1,),
#           (input,),                   # layer0: input, a virtual layer
#           (linear, False,),           # layer1: L1_linear, bias=False
#           (batch_norm,),              # layer2: L2_batch_norm
#           (func, relu,),              # layer3: L3_func_relu, relu activation
#           (linear_group, False,),     # layer4: L4_linear_group, bias=False
#           (batch_norm,),              # layer5: L5_batch_norm
#           (func, relu,),              # layer6: L6_func_relu, relu activation
#           (linear, False,),           # layer7: L7_linear, bias=False
#           (batch_norm,),              # layer8: L8_batch_norm
#           (shortcut, 0, True,),       # layer9: L9_shortcut_L0, shortcut from layer0
#                                       # use projection if dimension mismatch=True
#           (func, relu,),              # layer10: L10_func_relu, relu activation
#       ]
#
#
#
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

