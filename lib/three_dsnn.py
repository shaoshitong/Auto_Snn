import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tensorboardX.summary as summary
import numpy as np
from torch.autograd.variable import Variable
import torch.nn.init as init
import copy
from Snn_Auto_master.lib.plt_analyze import vis_img
from matplotlib import pyplot as plt
from Snn_Auto_master.lib.parameters_check import pd_save, parametersgradCheck
import math
import pandas as pd
from torch.nn import Parameter

print_button = False
weight_button = True

class threshold(torch.autograd.Function):
    """
    heaviside step threshold function
    """

    @staticmethod
    def forward(ctx, input, sigma):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]
        ctx.save_for_backward(input)
        ctx.sigma = sigma
        output = input.clone()
        output = torch.max(torch.tensor(0.0,device=output.device),torch.sign(output-0.5))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigma = ctx.sigma
        exponent = -torch.pow((0.5-input), 2)/(2.0*sigma**2)
        exp = torch.exp(exponent)
        erfc_grad = exp / (2.506628 * sigma)
        grad = erfc_grad * grad_output
        return grad, None
class Trinomial_operation(object):
    def __init__(self, max_n):
        self.max_n = max_n
        self.Trinomial_list()

    def Trinomial_list(self):
        max_n = self.max_n
        self.diag = [1 for i in range(self.max_n * 3)]
        for i in range(1, max_n * 3):
            self.diag[i] = self.diag[i - 1] * (i)
        self.diag_T = torch.ones(self.max_n, self.max_n, self.max_n, dtype=torch.int32)
        for i in range(self.max_n):
            for j in range(self.max_n):
                for k in range(self.max_n):
                    self.diag_T[i][j][k] = self.diag[i + j + k] / (self.diag[i] * self.diag[j] * self.diag[k])

    def get_value(self, i, j, k):
        if i >= self.max_n or j >= self.max_n or k >= self.max_n:
            exit(-1)
        return self.diag_T[i][j][k]

class utilMask(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(utilMask, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


"""
高斯卷积
"""


class guassNet(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, sigma=1, group=1, requires_grad=True):
        super(guassNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.group = group
        self.requires_grad = requires_grad
        self.gauss_kernel = self._gauss2D(self.kernel_size, self.sigma, self.group, self.in_channel,
                                          self.out_channel, self.requires_grad)
        self.gauss_bias = Parameter(torch.zeros(self.in_channel), requires_grad=self.requires_grad)
        self.GaussianBlur = lambda x: F.conv1d(x, weight=self.gauss_kernel, bias=self.gauss_bias, stride=1,
                                               groups=self.group,
                                               padding=(self.kernel_size - 1) // 2)

    def forward(self, x):
        x = self.GaussianBlur(x)
        return x

    def _gauss1D(self, kernel_size, sigma, group, in_channel, out_channel, requires_grad=False):
        kernel = torch.zeros(kernel_size)
        center = kernel_size // 2
        if sigma <= 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        s = sigma ** 2
        sum_val = 0
        for i in range(kernel_size):
            x = i - center
            kernel[i] = math.exp(-(x ** 2) / (2 * s))
            sum_val += kernel[i]
        out_channel = out_channel // group
        kernel = ((kernel / sum_val).unsqueeze(0)).repeat(out_channel, 1)
        kernel = Parameter((kernel.unsqueeze(0)).repeat(in_channel, 1, 1), requires_grad == requires_grad)
        return kernel

    def _gauss2D(self, kernel_size, sigma, group, in_channel, out_channel, requires_grad=False):
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        if sigma <= 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        s = sigma ** 2
        sum_val = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = math.exp(-(x ** 2 + y ** 2) / (2 * s))
                sum_val += kernel[i, j]
        out_channel = out_channel // group
        kernel = ((kernel / sum_val).unsqueeze(0)).repeat(out_channel, 1, 1)
        kernel = Parameter((kernel.unsqueeze(0)).repeat(in_channel, 1, 1, 1), requires_grad == requires_grad)
        return kernel


"""
高斯卷积
"""


def DiffInitial(data, shape, in_feature, out_feature, group=1):
    """
    K_{GX} = [-1 0 1 ; -2 0 2 ; -1 0 1], K_{GY} = {-1 -2 -1 ; 0 0 0 ; 1 2 1}
    """
    tmp = data.clone().detach().cuda()
    tmp = tmp.view(shape).unsqueeze(1).repeat(1, in_feature, 1, 1)
    kernel_row = Parameter(
        torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).repeat(in_feature,
                                                                                             out_feature // group, 1,
                                                                                             1),
        requires_grad=False).cuda()
    kernel_col = Parameter(
        torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).repeat(in_feature,
                                                                                            out_feature // group, 1, 1),
        requires_grad=False).cuda()
    kernel_gauss = guassNet(1, 1, kernel_size=5, requires_grad=False).cuda()
    tmp = kernel_gauss(tmp).cuda()
    tmp_col = F.conv2d(tmp, kernel_col, bias=Parameter(torch.Tensor([0.] * in_feature).cuda(), requires_grad=False),
                       stride=1,
                       groups=group,
                       padding=(3 - 1) // 2)
    tmp_row = F.conv2d(tmp, kernel_row, bias=Parameter(torch.Tensor([0.] * in_feature).cuda(), requires_grad=False),
                       stride=1,
                       groups=group,
                       padding=(3 - 1) // 2)
    grad = -torch.sqrt(tmp_col ** 2 + tmp_row ** 2).float()
    min = torch.min(grad)
    max = torch.max(grad)
    return ((grad - min) / (max - min)).view_as(data)


class axonLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v1):
        ctx.save_for_backward(v1)
        return torch.min(torch.max(v1, torch.Tensor([-.3]).cuda()), torch.Tensor([1]).cuda())

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        exponent = torch.where((input > -1) & (input < 1), torch.ones_like(input).cuda(),
                               torch.zeros_like(input).cuda())
        exponent = torch.where((input > 1) | (input < -1),
                               (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
        return exponent * grad_output


class point_cul_Layer(nn.Module):
    def __init__(self, in_feature, out_feature, tau_m=4., tau_s=1., grad_small=False, weight_require_grad=False,
                 weight_rand=False, device=None, STuning=True,grad_lr=0.1):
        """
        该模型中任何层当前假定都将高维张量拉伸至一维
        因此输入的张量维度为（batch_size,in_feature）
        """
        super(point_cul_Layer, self).__init__()
        self.device = device
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.weight_rand = weight_rand
        # self.weight_rand=False
        self.grad_small = grad_small
        self.weight_require_grad = weight_require_grad
        self.in_feature, self.out_feature = in_feature, out_feature
        self.tensor_tau_m1 = Parameter(torch.full([in_feature], math.exp(-1. / tau_m)).to(self.device),
                                       requires_grad=self.weight_require_grad)
        self.tensor_tau_s1 = Parameter(torch.full([in_feature], math.exp(-1. / tau_s)).to(self.device),
                                       requires_grad=self.weight_require_grad)
        self.tensor_tau_sm1 = Parameter(torch.full([in_feature], math.exp(-1. / tau_s - 1. / tau_m)).to(self.device),
                                        requires_grad=self.weight_require_grad)

        self.gaussbur = guassNet(1, 1, kernel_size=3, requires_grad=True)
        self.fineTuning = utilMask(in_feature, out_feature)
        self.STuning = STuning
        self.grad_lr = grad_lr
        self.sigma=1
    def forward(self, x,weight):
        x1, x2, x3 ,tensor_reset= x.unbind(dim=-1)
        if self.STuning:
            x = x1 * self.tensor_tau_m1 + x2 * self.tensor_tau_s1 +x3 * self.tensor_tau_sm1-threshold.apply(tensor_reset,1)/float(2**weight)
        else:
            x = x1 * self.tensor_tau_m1 + x2 * self.tensor_tau_s1 + x3 * self.tensor_tau_sm1

        if self.weight_rand:
            m = self.gaussbur(x.unsqueeze(1).view(x.shape[0], 1, int(math.sqrt(x.shape[1])), -1)).squeeze(1).view(
                x.shape[0], -1)
            m = F.relu(m)
            x = m
        return x

    def subWeightGrad(self, epoch, epochs, sigma, diag_num, path_num_x, path_num_y, path_num_z):
        # weight = max(.8,float(sigma) * diag_num /( (math.exp(-1. / self.tau_s - 1. / self.tau_m)/3+math.exp(-1. / self.tau_s )/3+math.exp(-1. / self.tau_m )/3)**(path_num )))
        if weight_button == True:
            weight = float(sigma) * diag_num * (
                    (math.exp(-1. / self.tau_s - 1. / self.tau_m) ** path_num_z)
                    * (math.exp(-1. / self.tau_s) ** path_num_y)
                    * (math.exp(-1. / self.tau_m) ** path_num_x))
        else:
            weight = sigma
        if not weight_button:
            pass
        for name, param in self.named_parameters():
            if param.requires_grad == True and param.grad != None:
                param.grad /= weight
                if epoch <= epochs // 5:
                    param.grad.data = param.grad.data.clamp_(-1, 1)
                else:
                    param.grad.data = param.grad.data.clamp_(-0.1, 0.1)


class three_dim_Layer(nn.Module):
    def __init__(self, shape, device, weight_require_grad=False, weight_rand=False, grad_lr=0.0001):
        super(three_dim_Layer, self).__init__()
        """
        x是基于输入
        y是基于正态分布
        原点与其他为0特征
        """
        self.x, self.y, self.z = shape
        self.device = device
        self.weight_require_grad = weight_require_grad
        self.weight_rand = weight_rand
        self.data_x = []
        self.data_y = []
        self.diag_T = Trinomial_operation(max(max(self.x, self.y), self.z))
        self.grad_lr = grad_lr

    def forward(self, x):
        tensor_reset=x
        tensor_prev = [[x.clone() for i in range(self.y)] for j in range(self.x)]
        for i in range(self.z):
            for j in range(self.y):
                for k in range(self.x):
                    """
                    push xx,yy,zz
                    """
                    if i == 0:
                        zz = torch.zeros(x.shape, dtype=torch.float32).to(self.device)
                    else:
                        zz = tensor_prev[j][k]
                    if j == 0:
                        yy = self.data_y[k]
                    else:
                        yy = tensor_prev[j - 1][k]
                    if k == 0:
                        xx = self.data_x[j]
                    else:
                        xx = tensor_prev[j][k - 1]

                    tensor_prev[j][k] = self.point_layer_module[str(i) + '_' + str(j) + '_' + str(k)](
                        torch.stack([xx, yy, zz,tensor_reset], dim=-1),i+j+k)
                    tensor_reset=tensor_reset if torch.eq(torch.randn(1),torch.zeros(1)).item() else tensor_prev[j][k]
                    tensor_prev[j][k] = axonLimit.apply(tensor_prev[j][k])
        return tensor_prev[-1][-1]

    def initiate_data(self, data, epoch=0, epochs=100, require_grad=True, init_option=True):
        self.data_x = []
        self.data_y = []
        for i in range(self.y):
            self.data_y.append(
                Parameter(data=torch.rand(data.shape, dtype=torch.float32).to(self.device) / math.sqrt(1.),
                          requires_grad=require_grad) if init_option else Parameter(
                    DiffInitial(data, [64, 28, 28], 1, 1), requires_grad=require_grad)
            )

        self.data_x = [data.to(self.device) for i in range(self.x)]

    def initiate_layer(self, data):
        self.point_layer = {}
        for i in range(self.z):
            for j in range(self.y):
                for k in range(self.x):
                    self.point_layer[str(i) + '_' + str(j) + '_' + str(k)] = point_cul_Layer(data.shape[1],
                                                                                             data.shape[1],
                                                                                             grad_small=False,
                                                                                             weight_require_grad=self.weight_require_grad,
                                                                                             weight_rand=self.weight_rand,
                                                                                             device=self.device,
                                                                                             # bool((i+j+k)%2),False
                                                                                             STuning=bool((i+j+k)%2),
                                                                                             grad_lr=self.grad_lr)
        self.point_layer_module = nn.ModuleDict(self.point_layer)

    def subWeightGrad(self, epoch, epochs, sigma=1):
        for i in range(self.z):
            for j in range(self.y):
                for k in range(self.x):
                    self.point_layer_module[str(i) + '_' + str(j) + '_' + str(k)].subWeightGrad(epoch, epochs, sigma,
                                                                                                self.diag_T.get_value(
                                                                                                    self.x - k - 1,
                                                                                                    self.y - j - 1,
                                                                                                    self.z - i - 1),
                                                                                                self.x - k - 1,
                                                                                                self.y - j - 1,
                                                                                                self.z - i - 1)


class merge_layer(nn.Module):
    def __init__(self, device, shape=None, weight_require_grad=True, weight_rand=True, grad_lr=0.01):
        super(merge_layer, self).__init__()
        if shape == None:
            self.shape = [3, 3, 3]
        else:
            self.shape = shape
        self.device = device
        self.three_dim_layer = three_dim_Layer(self.shape, self.device, weight_require_grad, weight_rand, grad_lr)

    def forward(self, x):
        x = self.three_dim_layer(x)
        return self.linear(x), x

    def initiate_data(self, input, epoch=0, epochs=100, option=False):
        if len(input.shape) != 2:
            input = input.view(input.shape[0], -1)
            self.first = input.shape[0]
            self.second = input.numel() / self.first
        else:
            self.first = input.shape[0]
            self.second = input.numel() / self.first
        self.three_dim_layer.initiate_data(input.to(self.device), epoch, epochs, True, option)

    def initiate_layer(self, input,classes):
        if len(input.shape) != 2:
            input = input.view(input.shape[0], -1)
            self.first = input.shape[0]
            self.second = input.numel() / self.first
        else:
            self.first = input.shape[0]
            self.second = input.numel() / self.first
        self.three_dim_layer.initiate_layer(input.to(self.device))
        self.linear = nn.Linear(int(self.second), classes, bias=True)
        stdv = 1. / math.sqrt(self.linear.weight.data.size(1) + self.linear.weight.data.size(0))
        self.linear.weight.data.uniform_(-stdv, stdv)
        self.linear.bias.data.fill_(0.)
    def subWeightGrad(self, epoch, epochs, sigma=1):
        self.three_dim_layer.subWeightGrad(epoch, epochs, sigma)
