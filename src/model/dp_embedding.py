import torch
import torch.nn as nn
from torch.nn.init import normal_ as normal
import numpy as np
# from model.op_func import MatmulBiasTanh


class EmbeddingNet(nn.Module):
    def __init__(self, cfg, magic=False):
        super(EmbeddingNet, self).__init__()
        self.cfg = cfg
        self.weights = nn.ParameterList()

        if cfg["bias"]:
            self.bias = nn.ParameterList()

        if self.cfg["resnet_dt"]:
            self.resnet_dt = nn.ParameterList()

        self.network_size = [1] + self.cfg["network_size"]

        if cfg["activation"] == "tanh":
            cfg["activation"] = nn.Tanh()
        else:
            pass

        # 初始化权重 normalization
        for i in range(1, len(self.network_size)):
            wij = torch.Tensor(self.network_size[i - 1], self.network_size[i])
            normal(
                wij,
                mean=0,
                std=(1.0 / np.sqrt(self.network_size[i - 1] + self.network_size[i])),
            )

            self.weights.append(nn.Parameter(wij, requires_grad=True))

            if self.cfg["bias"]:
                bias = torch.Tensor(1, self.network_size[i])
                normal(bias, mean=0, std=1)
                self.bias.append(nn.Parameter(bias, requires_grad=True))

            if self.cfg["resnet_dt"]:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=1, std=0.001)
                self.resnet_dt.append(nn.Parameter(resnet_dt, requires_grad=True))
        self.inter_compute = []

    def forward(self, x):
        if self.is_specialized():
            return self.specialization_forward(x)

        for i in range(1, len(self.network_size)):
            if self.cfg["bias"]:
                hiden = torch.matmul(x, self.weights[i - 1]) + self.bias[i - 1]
            else:
                hiden = torch.matmul(x, self.weights[i - 1])

            hiden = self.cfg["activation"](hiden)

            if self.network_size[i] == self.network_size[i - 1]:
                if self.cfg["resnet_dt"]:
                    x = hiden * self.resnet_dt[i - 2] + x
                else:
                    x = hiden + x
            elif self.network_size[i] == 2 * self.network_size[i - 1]:
                if self.cfg["resnet_dt"]:
                    x = torch.cat((x, x), dim=-1) + hiden * self.resnet_dt[i - 2]
                else:
                    x = torch.cat((x, x), dim=-1) + hiden
            else:
                x = hiden
        return x

    # Specialization for networksize (x, x, x) and resnet_dt == false
    # @torch.compile
    def specialization_forward(self, x):
        self.inter_compute = []
        hiden = torch.matmul(x, self.weights[0]) + self.bias[0]
        hiden = self.cfg["activation"](hiden)
        x = hiden
        self.inter_compute.append(hiden)

        # import ipdb; ipdb.set_trace()
        # hiden = MatmulBiasTanh.apply(x, self.weights[1], self.bias[1])
        hiden = torch.matmul(x, self.weights[1]) + self.bias[1]
        hiden = self.cfg["activation"](hiden)
        self.inter_compute.append(hiden)
        x = hiden + x

        hiden = torch.matmul(x, self.weights[2]) + self.bias[2]
        hiden = self.cfg["activation"](hiden)
        # hiden = MatmulBiasTanh.apply(x, self.weights[2], self.bias[2])
        self.inter_compute.append(hiden)
        x = hiden + x
        

        return x
    
    def compute_grad(self, grad_output):
        hidens = self.inter_compute
        grad_output_hiden2_final_to_hiden1_final = grad_output + torch.matmul(grad_output *  (1-hidens[2].mul(hidens[2])),  self.weights[2].transpose(-2,-1))
        # grad_output_hiden2_final_to_hiden1 = grad_output_hiden2_final_to_hiden1_final
        grad_output_hiden2_final_to_hiden0 = grad_output_hiden2_final_to_hiden1_final + torch.matmul(grad_output_hiden2_final_to_hiden1_final *  (1-hidens[1].mul(hidens[1])),  self.weights[1].transpose(-2,-1))
        grad_output_hiden2_final_to_R1 = torch.matmul(grad_output_hiden2_final_to_hiden0 *  (1-hidens[0].mul(hidens[0])),  self.weights[0].transpose(-2,-1))
        # import ipdb; ipdb.set_trace()
        return grad_output_hiden2_final_to_R1
    
    def is_specialized(self):
        return len(self.cfg["network_size"]) == 3 and not self.cfg["resnet_dt"]


class FittingNet(nn.Module):
    def __init__(self, cfg, input_dim, ener_shift, magic=False):
        super(FittingNet, self).__init__()
        self.weights = nn.ParameterList()

        if cfg["bias"]:
            self.bias = nn.ParameterList()

        if cfg["resnet_dt"]:
            self.resnet_dt = nn.ParameterList()

        self.network_size = [input_dim] + cfg["network_size"]

        if cfg["activation"] == "tanh":
            cfg["activation"] = nn.Tanh()
        else:
            pass
        self.cfg = cfg

        for i in range(1, len(self.network_size) - 1):
            wij = torch.Tensor(self.network_size[i - 1], self.network_size[i])
            normal(
                wij,
                mean=0,
                std=(1.0 / np.sqrt(self.network_size[i - 1] + self.network_size[i])),
            )

            self.weights.append(nn.Parameter(wij, requires_grad=True))

            if self.cfg["bias"]:
                bias = torch.Tensor(1, self.network_size[i])
                normal(bias, mean=0, std=1)
                self.bias.append(nn.Parameter(bias, requires_grad=True))

            if i > 1 and self.cfg["resnet_dt"]:
                resnet_dt = torch.Tensor(1, self.network_size[i])
                normal(resnet_dt, mean=0.1, std=0.001)
                self.resnet_dt.append(nn.Parameter(resnet_dt, requires_grad=True))

        i = len(self.network_size) - 1
        wij = torch.randn(self.network_size[i - 1], self.network_size[i])
        normal(
            wij,
            mean=0,
            std=(1.0 / np.sqrt(self.network_size[i - 1] + self.network_size[i])),
        )

        self.weights.append(nn.Parameter(wij, requires_grad=True))

        if self.cfg["bias"]:
            bias_init = torch.randn(1, self.network_size[i])
            normal(bias_init, mean=ener_shift, std=1.0)
            self.bias.append(nn.Parameter(bias_init, requires_grad=True))  # 初始化指定均值
        
        self.inter_compute = []

    def forward(self, x):
        if self.is_specialized():
            return self.specialization_forward(x)

        for i in range(1, len(self.network_size) - 1):
            if self.cfg["bias"]:
                hiden = torch.matmul(x, self.weights[i - 1]) + self.bias[i - 1]
            else:
                hiden = torch.matmul(x, self.weights[i - 1])

            hiden = self.cfg["activation"](hiden)

            if i > 1:
                if (
                    self.network_size[i] == self.network_size[i - 1]
                    and self.cfg["resnet_dt"]
                ):
                    x = hiden * self.resnet_dt[i - 2] + x
                else:
                    x = hiden + x
            else:
                x = hiden

        i = len(self.network_size) - 1

        if self.cfg["bias"]:
            x = torch.matmul(x, self.weights[i - 1]) + self.bias[i - 1]
        else:
            x = torch.matmul(x, self.weights[i - 1])
        return x

    # Specialization for networksize (x, x, x, 1) and resnet_dt == True
    # @torch.compile
    def specialization_forward(self, x):
        self.inter_compute = []
        # hiden = MatmulBiasTanh.apply(x, self.weights[0], self.bias[0])
        hiden = torch.matmul(x, self.weights[0]) + self.bias[0]
        hiden = self.cfg["activation"](hiden)
        self.inter_compute.append(hiden)
        x = hiden

        # hiden = MatmulBiasTanh.apply(x, self.weights[1], self.bias[1])

        hiden = torch.matmul(x, self.weights[1]) + self.bias[1]
        hiden = self.cfg["activation"](hiden)
        self.inter_compute.append(hiden)
        # x = hiden * self.resnet_dt[0] + x
        x = hiden + x

        # hiden = MatmulBiasTanh.apply(x, self.weights[2], self.bias[2])

        hiden = torch.matmul(x, self.weights[2]) + self.bias[2]
        hiden = self.cfg["activation"](hiden)
        self.inter_compute.append(hiden)
        # x = hiden * self.resnet_dt[1] + x
        x = hiden + x

        x = torch.matmul(x, self.weights[3]) + self.bias[3]
        self.inter_compute.append(x)

        return x
    
    def compute_grad(self):
        hidens = self.inter_compute
        # grad_output = grad_output * (1 - hidens[-1].mul(hidens[-1]))
        # grad_x = torch.matmul(grad_output, self.weights[3].transpose(-2, -1))

        grad_output_Ei_to_hiden2_final = self.weights[3].transpose(-2, -1)
        grad_output_Ei_to_hiden2 = grad_output_Ei_to_hiden2_final
        grad_output_Ei_to_hiden1_final = grad_output_Ei_to_hiden2_final + torch.matmul(grad_output_Ei_to_hiden2 *  (1-hidens[2].mul(hidens[2])),  self.weights[2].transpose(-2,-1))
        # grad_output_Ei_to_hiden1 = grad_output_Ei_to_hiden1_final
        grad_output_Ei_to_hiden0 = grad_output_Ei_to_hiden1_final + torch.matmul(grad_output_Ei_to_hiden1_final * (1-hidens[1].mul(hidens[1])), self.weights[1].transpose(-2,-1))
        grad_output_Ei_to_D =  torch.matmul(grad_output_Ei_to_hiden0 * (1-hidens[0].mul(hidens[0])), self.weights[0].transpose(-2,-1))
        # self.inter_compute = []
        # import ipdb; ipdb.set_trace()
        return grad_output_Ei_to_D
    
    def is_specialized(self):
        return len(self.cfg["network_size"]) == 4 and self.cfg["resnet_dt"]
