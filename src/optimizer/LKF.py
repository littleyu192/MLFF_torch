import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math
import numpy as np
import ipdb


class LKFOptimizer(Optimizer):
    def __init__(self, params, kalman_lambda=0.1, kalman_nue=0.9, block_size=5120, device=torch.device('cuda')):
        super(LKFOptimizer, self).__init__(params, {"lr": 0.1})
        self.kalman_lambda = kalman_lambda
        self.kalman_nue = kalman_nue
        self.block_size = block_size
        self.device = device
        self.__init_P()

    def __init_P(self):

        param_nums = []
        param_sum = 0

        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                param_num = param.data.nelement()
                if param_sum + param_num > self.block_size:
                    param_nums.append(param_sum)
                    param_sum = param_num
                else:
                    param_sum += param_num

        param_nums.append(param_sum)

        self.P = []
        param_packed_index = []
        for param_num in param_nums:
            if param_num >= self.block_size:
                block_num = math.ceil(param_num / self.block_size)
                for i in range(block_num):
                    if i != block_num - 1:
                        self.P.append(torch.eye(self.block_size).to(self.device))
                        param_packed_index.append(self.block_size)
                    else:
                        self.P.append(
                            torch.eye(param_num - self.block_size * i).to(self.device)
                        )
                        param_packed_index.append(param_num - self.block_size * i)
            else:
                self.P.append(torch.eye(param_num).to(self.device))
                param_packed_index.append(param_num)

        self.weights_num = len(self.P)
        self.param_packed_index = param_packed_index

    def __split_weights(self, weight):
        param_num = weight.nelement()
        res = []
        if param_num < self.block_size:
            res.append(weight)
        else:
            block_num = math.ceil(param_num / self.block_size)
            for i in range(block_num):
                if i != block_num - 1:
                    res.append(weight[i * self.block_size : (i + 1) * self.block_size])
                else:
                    res.append(weight[i * self.block_size :])
        return res

    def __update(self, H, error, weights):
        tmp = 0

        for i in range(self.weights_num):
            tmp = tmp + (
                self.kalman_lambda + torch.matmul(torch.matmul(H[i].T, self.P[i]), H[i])
            )

        A = 1 / tmp

        for i in range(self.weights_num):
            K = torch.matmul(self.P[i], H[i])

            weights[i] = weights[i] + A * error * K

            self.P[i] = (1 / self.kalman_lambda) * (
                self.P[i] - A * torch.matmul(K, K.T)
            )

        self.kalman_lambda = self.kalman_nue * self.kalman_lambda + 1 - self.kalman_nue

        i = 0
        param_sum = 0
        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                param_num = param.nelement()
                weight_tmp = weights[i][param_sum : param_sum + param_num]
                if param_num < self.block_size:
                    if param.ndim > 1:
                        param.data = weight_tmp.reshape(param.data.T.shape).T
                    else:
                        param.data = weight_tmp.reshape(param.data.shape)

                    param_sum += param_num

                    if param_sum == self.param_packed_index[i]:
                        i += 1
                        param_sum = 0
                else:
                    block_num = math.ceil(param_num / self.block_size)
                    for j in range(block_num):
                        if j == 0:
                            tmp_weight = weights[i]
                        else:
                            tmp_weight = torch.concat([tmp_weight, weights[i]], dim=0)
                        i += 1
                    param.data = tmp_weight.reshape(param.data.T.shape).T

    def set_grad_prefactor(self, grad_prefactor):
        self.grad_prefactor = grad_prefactor

    def step(self, error):

        weights = []
        H = []
        param_index = 0
        param_sum = 0

        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                if param.ndim > 1:
                    tmp = param.data.T.reshape(param.data.nelement(), 1)
                    tmp_grad = (param.grad / self.grad_prefactor).T.reshape(
                        param.grad.nelement(), 1
                    )
                else:
                    tmp = param.data.reshape(param.data.nelement(), 1)
                    tmp_grad = (param.grad / self.grad_prefactor).reshape(
                        param.grad.nelement(), 1
                    )

                tmp = self.__split_weights(tmp)
                tmp_grad = self.__split_weights(tmp_grad)

                for split_grad, split_weight in zip(tmp_grad, tmp):
                    nelement = split_grad.nelement()

                    if param_sum == 0:
                        res_grad = split_grad
                        res = split_weight
                    else:
                        res_grad = torch.concat((res_grad, split_grad), dim=0)
                        res = torch.concat((res, split_weight), dim=0)

                    param_sum += nelement

                    if param_sum == self.param_packed_index[param_index]:
                        H.append(res_grad)
                        weights.append(res)
                        param_sum = 0
                        param_index += 1

        self.__update(H, error, weights)
