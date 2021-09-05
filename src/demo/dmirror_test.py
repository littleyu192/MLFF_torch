import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dmirror import dmirror_FC

def func_SQR(x):
    return x ** 2

def func_d_SQR(x):
    return 2 * x

class dmirror_net(nn.Module):
    def __init__(self, L1_dim, L2_dim):
        super(dmirror_net, self).__init__()

        dmirror_cfg = [
            ('linear', L1_dim, L2_dim, False),
            ('activation',),
            ('linear', L2_dim, 1, False)
        ]
        self.net = dmirror_FC(dmirror_cfg, func_SQR, func_d_SQR)

        self.net.layers[0][1].w.data = torch.tensor([
            [0.2, 0.8, 0.3, 0.6],
            [0.7, 0.8, 0.1, 0.9]
        ])
        self.net.layers[2][1].w.data = torch.tensor([
            [0.3, 0.4]
        ])
        self.dfeat = torch.tensor([[0.5, 0.6, 0.7, 0.8]])

    def forward(self, x0):
        res0, res1 = self.net.forward(x0)
        return torch.mv(self.dfeat, res1)

input_x = torch.tensor([10., 9., 8., 7., 6., 5., 3., 2.])
input_x = torch.tensor([10., 9., 8., 7.])
label = torch.tensor([500.])
n = dmirror_net(4, 2)
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

    #dlossdw1 = n.calc_dlossdw1(input_x, 2*(out-label))
    #dlossdw2 = n.calc_dlossdw2(input_x, 2*(out-label))

    print("<================= turn %s ================>"%(i))
    print("Input    :  ", input_x)
    print("Label    :  ", label)
    print("Output   : ", out)
#    print("dlossdw1 : ", dlossdw1)
#    print("dlossdw2 : ", dlossdw2)
#    print("W_1      : ", n.W_1)
#    print("W_2      : ", n.W_2)
#    print("W_3      : ", n.W_3)
    print("***************************")

    for name, param in n.named_parameters():
        print(name, ":", param.size())
        print("value   : ", param.data)
        print("grad    : ", param.grad)
        print("***************************")

    optimizer.step()
    if (i % weight_decay_round == 0):
        scheduler.step()
