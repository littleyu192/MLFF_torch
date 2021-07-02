# !/usr/bin/python
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

################################################################
# fully connection nn
################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class simpleNet(nn.Module):
    """
     The overall network. It contains 3 layers of fully connection layer.
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class FC3Net(nn.Module):
    """
     The overall network. It contains 3 layers of fully connection layer with RELU activation.
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(FC3Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class FC3BN_Net(nn.Module):
    """
     The overall network. It contains 3 layers of fully connection layer with batch normalization.
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(FC3BN_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x