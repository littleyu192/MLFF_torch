#
# Here we implement some bi-direction dmirror layers as the base
# components to build dmirror style networks. The main idea is,
# each layer instance can handle both the "forward" and "d_forward"
# operation, and it collects all the required data for "d_forward"
# phase during the "forward" phase. By doing so, we can implement
# a hidden style weight & variable sharing, we just need to describe
# the network and all the sharing issues will be done automaticlly
#
# To implement the bi-direction operation, a d_order parameter is
# introduced to forward() method of dmirror layers
#   d_order=0:  0 order derivative (normal forward)
#   d_order=1:  1st order derivative (d_forward)
#   d_order=n:  nth order derivative (not implemented)
#
# A network module based on dmirror layers should correctly handle
# the d_order of it's layers, and it's best not to expose d_order
# related issues to higher level modules or callers.
#
# Basic dmirror layers:
#
#   dmirror_linear(self, in_dim, out_dim, bias=True)
#       * similar to nn.linear
#
#   dmirror_activation(self, func, d_func)
#       * a general activation layer
#       * just specify your activation/d_activation function 
#
#   dmirror_scale()
#       * automaticlly scale/normalize input features
#       * not implemented yet
#
#   dmirror_layer_norm()
#   dmirror_batch_norm()
#       * not implemented yet
#
# Basic dmirror network modules:
#
#   dmirror_FC(self, cfg, act_func, d_act_func)
#       * cfg describes the base part of a dmirror style network,
#         the whole network (including the mirrored part) will be
#         automaticlly generated. An example:
#
#           cfg = [
#                   (scale,),                   # layer: scale_1
#                   (linear, 16, 32, True),     # layer: linear_1, bias=True
#                   (activation,),              # layer: activation_1
#                   (linear, 32, 64, True),     # layer: linear_2, bias=True
#                   (activation,),              # layer: activation_2
#                   (linear, 64, 8, False),     # layer: linear3, bias=False
#                   (activation,),              # layer: activation_3
#                   (linear, 8, 1, False),      # layer: linear4, bias=False
#           ]
#
#           the auto generated (virtual) mirrored part should be:
#           [
#                   (d_linear, 1, 8, False),    # layer: linear4, bias=False
#                   (d_activation,),            # layer: activation_3
#                   (d_linear, 8, 64, False),   # layer: linear3, bias=False
#                   (d_activation,),            # layer: activation_2
#                   (d_linear, 64, 32, True),   # layer: linear_2, bias=False
#                   (d_activation,),            # layer: activation_1
#                   (d_linear, 32, 16, True),   # layer: linear_1, bias=False
#                   (d_scale,),                 # layer: scale_1
#           ]
#
#       * act_func/d_act_func is the actual activation/d_activation
#         function which is called by the dmirror_activation layer
#       * forward() returns the last layer's output of both the base
#         part and the mirrored part, the method call should like:
#           y1, y2 = instance_dimirror_FC.forward(x)
#         or
#           Ei, dEi_dFeat = MLFF_dmirror_FC.forward(Feat_atom_i)
#

import torch
import torch.nn as nn
import collections

# TODO: 1) expand to higher than 1-dimensional input
#       2) make sure the code can running on GPU
#

class dmirror_linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(dmirror_linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=True)
        if (bias == True):
            self.bias = nn.Parameter(torch.randn(out_dim), requires_grad=True)

    def forward(self, x, d_order=0):
        if (d_order == 0):
            res = torch.mv(self.weight, x)
            if (self.bias is not None):
                return res + self.bias
            else:
                return res
        elif (d_order == 1):
            return torch.mv(self.weight.t(), x)
        else:
            raise RuntimeError(
                "Notimplemented for d_order = %s" %(d_order)
            )


class dmirror_activation(nn.Module):
    def __init__(self, func, d_func):
        super(dmirror_activation, self).__init__()
        self.func = func
        self.d_func = d_func
        self.k = torch.tensor([])

    def forward(self, x, d_order=0):
        if (d_order == 0):
            self.k = x
            return self.func(x)
        elif (d_order == 1):
            return x * self.d_func(self.k)
        else:
            raise RuntimeError(
                "Notimplemented for d_order = %s" %(d_order)
            )


class dmirror_FC(nn.Module):
    def __init__(self, cfg, act_func, d_act_func):
        super(dmirror_FC, self).__init__()
        self.cfg = cfg
        self.act_func = act_func
        self.d_act_func = d_act_func
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
                    dmirror_linear(in_dim, out_dim, bias)
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

        # the layer parameters will be registered to nn Module,
        # so optimizer can update the layer parameters.
        #
        self.base_net = nn.Sequential(
            collections.OrderedDict(self.layers)
        )
        self.mirror_net = nn.Sequential(
            collections.OrderedDict(reversed(self.layers))
        )
        print("dmirror_FC: start of network instance dump ==============>")
        print("<----------------------- base_net ----------------------->")
        print(self.base_net)
        print("<---------------------- mirror_net ---------------------->")
        print(self.mirror_net)
        print("dmirror_FC: end of network instance dump ================>")

    # we can't call forward() of sequentialized module, since
    # we extened the param list of the layers' forward()
    #
    def forward(self, x):
        for name, obj in (self.layers):
            x = obj.forward(x, 0)
        res0 = x

        x = torch.ones_like(res0)
        for name, obj in (reversed(self.layers)):
            x = obj.forward(x, 1)
        res1 = x

        return res0, res1

