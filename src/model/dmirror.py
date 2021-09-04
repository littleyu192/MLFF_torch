import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class dmirror_linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(dmirror_linear, self).__init__()
        self.bias = bias
        self.w = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=True)
        if (self.bias == True):
            self.b = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x, d_order=0):
        if (d_order == 0):
            res = torch.mv(self.w, x)
            if (self.bias == True)
                return res + self.b
            else
                return res
        elif (d_order == 1):
            return torch.mv(self.w.t(), x)
        else:
            torch.Error("Unsupported d_order: %d", %(d_order))


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
            torch.Error("Unsupported d_order: %d", %(d_order))


class dmirror_FC(nn.Module):
    def __init__(self, cfg, act_func, d_act_func):
        super(dmirror_FC, self).__init__()
        self.cfg = cfg
        self.act_func = act_func
        self.d_act_func = d_act_func
        self.layers = []

        idx_linear = 1
        idx_activation = 1
        for in_dim, out_dim, bias in enumerate(self.cfg):
            if (isinstance(in_dim, int)):
                if (isinstance(bias, str) and (bias == 'bias')):
                    need_bias = True
                else:
                    need_bias = False
                self.layers.append((
                    'dmirror_linear_'+str(idx_linear),
                    dmirror_linear(in_dim, out_dim, need_bias)
                ))
                idx_linear++
            elif (isinstance(in_dim, str) and (in_dim == 'act')):
                self.layers.append((
                    'dmirror_activation_'+str(idx_activation),
                    dmirror_activation(act_func, d_act_func)
                ))
                idx_activation++
            else:
                torch.Error("Unsupported cfg item")


    def forward(self, x, d_order=0):
        if (d_order == 0):
            layers = self.layers
        elif (d_order == 1):
            layers = reversed(self.layers)
        else:
            torch.Error("Unsupported d_order %d", %(d_order))
     
        for name, obj in enumrate(layers):
            x = obj(x, d_order)



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
        x4 = self.layer4(torch.ones([1]))
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


input_x = torch.tensor([10., 9., 8., 7., 6., 5., 3., 2.])
label = torch.tensor([500.])
n = dmirror_net(8, 16)
loss_fn = nn.MSELoss()

learning_rate = 0.01
weight_decay = 0.9
weight_decay_round = 10
optimizer = optim.Adam(n.parameters(), lr=learning_rate)
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
