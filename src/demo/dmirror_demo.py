#
#  A demo of how to build a dmirror net to calculate Force from
#  dEnergy/dR in a forward and modulized way, in this way we can
#  design and train a multi-task Energy-Force combination network
#  just like operating an ordinary forward DNN network
#
#  The demo uses square(SQR) and its corresponding derivative(d_SQR)
#  as active functions, and they can be easily replaced with other
#  function/d_function pairs, such as SoftPlus & Sigmod
#
#        Layer1 Layer2 Layer3        Layer4 Layer5 Layer6 d_feat
#         |--|   |--|   |--|          |--|   |--|   |--|   |--|
#         |  |x1 |  |x2 |  |x3        |  |x4 |  |x5 |  |x6 |  |x7
#    x0-->|  |-->|  |-->|  |-->E  1-->|  |-->|  |-->|  |-->|  |-->F
#         |  | | |  |   |  |          |  | ->|  |   |  |   |  |
#         |__| | |__|   |__|          |__| | |__|   |__|   |__|
#              |                           |
#              |------->---------->--------|
#                                           elw-mul      mul-sum
#  Layer_ops:    SQR                         d_SQR       dFeat/dR 
#        linear1       linear2     d_linear1     d_linear2
#
#  Parameters:
#         W_1           W_2          W_2_T         W_1_T   W_3
#        shared        shared       shared        shared  const   
#
#  Dimensions:
#     L1_dim       L2_dim              L2_dim        L1_dim
#            L2_dim          1     1         L2_dim            1
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#TODO: 1) auto check the correctness of W & grad
#      2) fix calc_dlossdw1/calc_dlossdw2 for bias = True
#

class SQR_layer(nn.Module):
    def __init__(self, **kwargs):
        super(SQR_layer, self).__init__()

    def forward(self, x):
        return x**2


class d_SQR_layer(nn.Module):
    def __init__(self, **kwargs):
        super(d_SQR_layer, self).__init__()

    def forward(self, x, k):
        return x*2*k


class dmirror_net(nn.Module):
    def __init__(self, L1_dim, L2_dim):
        super(dmirror_net, self).__init__()

        # layer entities
        #
        self.layer1 = nn.Linear(L1_dim, L2_dim, bias = False)
        self.layer2 = SQR_layer()
        self.layer3 = nn.Linear(L2_dim, 1, bias = False)
        self.layer4 = nn.Linear(1, L2_dim, bias = False)
        self.layer5 = d_SQR_layer()
        self.layer6 = nn.Linear(L2_dim, L1_dim, bias = False)

        # setting shared parameters
        #
        self.W_1 = torch.rand(L2_dim, L1_dim)
        self.W_2 = torch.rand(1, L2_dim)
        self.W_3 = torch.rand(1, L1_dim)
        
        self.W_1 = torch.tensor([
            [0.2, 0.8, 0.3, 0.6],
            [0.7, 0.8, 0.1, 0.9]
        ])
        self.W_2 = torch.tensor([
            [0.3, 0.4]
        ])
        self.W_3 = torch.tensor([[0.5, 0.6, 0.7, 0.8]])
        #self.W_1 = torch.tensor([3.]).unsqueeze(1)
        #self.W_2 = torch.tensor([4.]).unsqueeze(1)
        #self.W_3 = torch.tensor([5.]).unsqueeze(1)
        self.layer1.weight = nn.Parameter(self.W_1)
        self.layer3.weight = nn.Parameter(self.W_2)
        self.layer6.weight = nn.Parameter(self.W_1.transpose(0, 1))
        self.layer4.weight = nn.Parameter(self.W_2.transpose(0, 1))
        #print(self.W_1.shape)
        #print(self.W_2.shape)
        #print(self.W_3.shape)

    def forward(self, x0):
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(torch.ones(1))
        x5 = self.layer5(x4, x1)
        x6 = self.layer6(x5)
        x7 = torch.mv(self.W_3, x6)
        #print("x1 = ", x1)
        #print("x2 = ", x2)
        #print("x3 = ", x3)
        #print("x4 = ", x4)
        #print("x5 = ", x5)
        #print("x6 = ", x6)
        #print("x7 = ", x7)
        return x7

    def calc_dlossdw1(self, x0, dlossdf):
        w1 = self.W_1
        w2t = self.W_2.t()
        w3 = self.W_3
        w3t = self.W_3.t()
        x0 = x0.unsqueeze(1)
        x0t = x0.t()
        dfdw1 = 2 * (((w2t * (w1 @ x0)) @ w3) + (((w1 @ w3t) * w2t) @ x0t))
        return dlossdf * dfdw1

    def calc_dlossdw2(self, x0, dlossdf):
        w1t = self.W_1.t()
        w3 = self.W_3
        x0 = x0.unsqueeze(1)
        x0t = x0.t()
        dfdw2 = 2 * ((w3 @ w1t) * (x0t @ w1t))
        return dlossdf * dfdw2


#input_x = torch.tensor([10., 9., 8., 7., 6., 5., 3., 2.])
input_x = torch.tensor([10., 9., 8., 7.])
label = torch.tensor([500.])
n = dmirror_net(4, 2)
#n = dmirror_net(1, 1)
loss_fn = nn.MSELoss()

learning_rate = 0.00000001
weight_decay = 0.9
weight_decay_round = 10
optimizer = optim.SGD(n.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=weight_decay)


for i in range(200):
    out = n(input_x)
    loss = loss_fn(out, label)
    optimizer.zero_grad()
    loss.backward()

    dlossdw1 = n.calc_dlossdw1(input_x, 2*(out-label))
    dlossdw2 = n.calc_dlossdw2(input_x, 2*(out-label))

    print("<================= turn %s ================>"%(i))
    print("Input    :  ", input_x)
    print("Label    :  ", label)
    print("Output   : ", out)
    print("dlossdw1 : ", dlossdw1)
    print("dlossdw2 : ", dlossdw2)
    print("W_1      : ", n.W_1)
    print("W_2      : ", n.W_2)
    print("W_3      : ", n.W_3)
    print("***************************")

    for name, param in n.named_parameters():
        print(name, ":", param.size())
        print("value   : ", param.data)
        print("grad    : ", param.grad)
        print("***************************")

    optimizer.step()
    if (i % weight_decay_round == 0):
        scheduler.step()
