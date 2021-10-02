import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from Snn_Auto_master.lib.data_loaders import revertNoramlImgae
from Snn_Auto_master.lib.plt_analyze import vis_img
from Snn_Auto_master.lib.parameters_check import pd_save, parametersgradCheck
from Snn_Auto_master.lib.SNnorm import SNConv2d, SNLinear
from Snn_Auto_master.lib.fractallayer import LastJoiner
from Snn_Auto_master.lib.DenseNet import DenseNet
from Snn_Auto_master.lib.cocoscontextloss import ContextualLoss_forward
from Snn_Auto_master.lib.featurefocusing_v2 import Feature_forward
from Snn_Auto_master.lib.dimixloss import DimixLoss, DimixLoss_neg
from Snn_Auto_master.lib.PointConv import PointConv
import math
import pandas as pd
from torch.nn.utils import spectral_norm
from torch.nn.parameter import Parameter
from omegaconf import OmegaConf
import matplotlib.pyplot as plt


def yaml_config_get(yamlname):
    """
    该函数是为了获取模型的配置信息
    """

    conf = OmegaConf.load(yamlname)
    return conf


class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x, *args):
        return self.function(x, x.size()[-1], *args)


def batch_norm(input):
    input_linear = input.view(input.shape[0], -1)
    mean = input_linear.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    std = input_linear.std(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    return torch.div(torch.sub(input, mean), std)


filename = "./train.yaml"
yaml = yaml_config_get(filename)
# yaml = yaml_config_get("./train.yaml")
dataoption = yaml['data']


def size_change(f, s):
    def change(xx):
        xx: torch.Tensor
        xx = F.interpolate(xx, (s, s), mode='bilinear', align_corners=True)
        g = f // xx.shape[1]
        if (int(g) == 0):
            xx = xx[:, ::int(xx.shape[1] // f), :, :]
        else:
            xx = xx.repeat(1, f // xx.shape[1], 1, 1)
        return xx

    return change


class Shortcut(nn.Module):
    """
    该层是为了残差连接而设计，从输入数据剪切一块后进行填充
    目前在特征从初始变换到64大小时会用到
    """

    def __init__(self, in_feature, out_feature, use_same=False, proportion=2):
        in_feature: int
        out_feature: int
        super(Shortcut, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        if use_same == False:
            self.shortcut = lambda x: F.pad(x[:, :, ::proportion, ::proportion],
                                            (0, 0, 0, 0, (self.out_feature - x.shape[1]) // 2,
                                             (self.out_feature - x.shape[1]) // 2 + (
                                                     self.out_feature - x.shape[1]) % 2),
                                            "constant", 0)
        else:
            self.shortcut = lambda x: F.pad(x,
                                            (0, 0, 0, 0, (self.out_feature - x.shape[1]) // 2,
                                             (self.out_feature - x.shape[1]) // 2 + (
                                                     self.out_feature - x.shape[1]) % 2),
                                            "constant", 0)

    def forward(self, x):
        return self.shortcut(x)


class block_in(nn.Module):
    def __init__(self, in_feature, out_feature=64, p=0.2):
        super(block_in, self).__init__()
        num_input_feature_list_list = yaml['densenetparameter']['feature_list']
        bn_size = yaml['densenetparameter']['bn_size']
        drop_rate = yaml['densenetparameter']['drop_rate']
        use_size_change = yaml['densenetparameter']['use_size_change']
        num_layer = None
        if dataoption in ["mnist", "fashionmnist", "cifar100", "cifar10", "car", "svhn", "stl-10"]:
            self.block_in_layer = DenseNet(num_input_feature_list_list, bn_size, drop_rate, use_size_change, num_layer)
        elif dataoption == "eeg":
            self.block_in_layer = DenseNet(num_input_feature_list_list, bn_size, drop_rate, use_size_change, num_layer)
        else:
            raise KeyError("not have this dataset")
        self.conv_cat = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(out_feature, 3 * out_feature, (4, 4), stride=2, padding=0, bias=False),
                                      nn.BatchNorm2d(3 * out_feature), )
        self.out_feature = out_feature
        self.relu = nn.ReLU()
        self.f_conv = nn.ModuleList(
            [SNConv2d(3 * out_feature, out_feature, (1, 1), stride=1, padding=0, bias=False) for _ in range(3)])
        self.training = False
        self.dropout = nn.Dropout(p=p)
        self.in_sample = nn.Sequential(*[
            nn.Conv2d(in_feature, num_input_feature_list_list[0][0], (3, 3), stride=(1, 1), padding=1, bias=False),
        ])
        self.out_sample = nn.Sequential(*[
            nn.Conv2d(num_input_feature_list_list[-1][-1], out_feature, (3, 3), stride=(1, 1), padding=1, bias=False),
        ])

    def settest(self, training_status):
        self.training = training_status

    def forward(self, x):
        x = self.in_sample(x)
        x = self.block_in_layer(x)
        x = self.out_sample(x)
        m = size_change(3 * self.out_feature, x.size()[-1] // 2)
        x = self.relu(self.conv_cat(x) + m(x))
        a, b, c = self.f_conv[0](x), self.f_conv[1](x), self.f_conv[2](x)
        del x
        return a, b, c


class block_out(nn.Module):
    def __init__(self, feature, classes, size, use_pool='none'):
        super(block_out, self).__init__()

        if use_pool == 'none':
            self.classifiar = nn.Sequential(nn.Flatten(), nn.Linear((feature * 8 * 8), classes))
            self.biclassifier = nn.Sequential(nn.Flatten(), nn.Linear((feature * 8 * 8), 1))
        else:
            self.classifiar = nn.Sequential(nn.Flatten(), nn.Linear(feature, classes))
            self.biclassifier = nn.Sequential(nn.Flatten(), nn.Linear((feature), 1))
        self.transition_layer = nn.Sequential(*[
            nn.BatchNorm2d(feature),
            nn.ReLU(inplace=True),
            Lambda(F.avg_pool2d)])
        self.training = False
        self.use_pool = use_pool
        self.size = size
        self.classes = classes

    def settest(self, training_status):
        self.training = training_status

    def forward(self, tau, a=None, b=None, c=None):
        if a == None and b == None and c == None:
            tau = self.transition_layer(tau)
            if self.use_pool == 'none':
                tau = self.classifiar(tau)
            elif self.use_pool == 'max':
                tau = self.classifiar(F.avg_pool2d(tau, tau.shape[-1]))
            elif self.use_pool == 'avg':
                tau = self.classifiar(F.avg_pool2d(tau, tau.shape[-1]))
            return tau
        else:
            tau = self.transition_layer(tau)
            abc = F.max_pool2d(torch.abs(F.dropout(a, p=0.9) + F.dropout(b, p=0.9) + F.dropout(c, p=0.9)), a.shape[-1])
            abc = torch.sigmoid(self.biclassifier(abc)) * (self.classes - 1)
            if self.use_pool == 'none':
                tau = self.classifiar(tau)
            elif self.use_pool == 'max':
                tau = self.classifiar(F.avg_pool2d(tau, tau.shape[-1]))
            elif self.use_pool == 'avg':
                tau = self.classifiar(F.avg_pool2d(tau, tau.shape[-1]))
            round_abc = torch.round(abc)
            with torch.no_grad():
                t = torch.zeros_like(tau).to(tau.device).scatter(dim=1, index=round_abc.long(),
                                                                 value=math.exp(1 / self.classes)).float()
            tau = tau + t
            return tau, abc


class block_eq(nn.Module):
    def __init__(self, eq_feature,tmp_feature,dropout):
        super(block_eq, self).__init__()
        self.eq_feature = eq_feature
        self.tmp_feature=tmp_feature
        self.convz1 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (5, 1), padding=(2, 0))
        self.convo = nn.Sequential(*[
            nn.Conv2d(eq_feature, eq_feature, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(eq_feature),
        ])
        self.xgru=nn.Sequential(nn.Conv2d(eq_feature,tmp_feature,(1,1),(1,1),padding=0,bias=False))
        self._initialize()
    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="sigmoid")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
        for layer in self.convo.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)

        nn.init.kaiming_normal_(self.convq1.weight.data, mode="fan_in", nonlinearity="tanh")
        nn.init.kaiming_normal_(self.convq2.weight.data, mode="fan_in", nonlinearity="tanh")

    def forward(self, m):
        h, x = m
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = F.relu((1 + z) * h + (1 - z) * q)
        hx  = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = F.relu((1 + z) * h + (1 - z) * q)
        x =  self.xgru(h)
        h =  self.convo(h)
        del  m, z, r, q
        return (h, x)


class multi_block_eq(nn.Module):
    def __init__(self, in_feature, out_feature,hidden_size, multi_k=1, stride=1,dropout=0.1):
        super(multi_block_eq, self).__init__()
        if in_feature != out_feature or stride != 1:
            self.sample = nn.Sequential(
                nn.Conv2d(in_feature, out_feature, (stride, stride), (stride, stride), bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_feature))
            self.psample = nn.Sequential(
                nn.Conv2d(in_feature, hidden_size, (stride, stride), (stride, stride), bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(hidden_size))
            self.qsample= nn.Sequential(
                nn.Conv2d(hidden_size, out_feature, (1,1),(1,1), bias=False))
            for layer in self.sample.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
                if isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight.data)
                    nn.init.zeros_(layer.bias.data)
            for layer in self.psample.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
                if isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight.data)
                    nn.init.zeros_(layer.bias.data)
        self.xgru=nn.Sequential(nn.Conv2d(in_feature,in_feature,(1,1),(1,1),padding=0,bias=False))
        self.model = nn.Sequential(*[
            block_eq(out_feature,hidden_size,dropout) for _ in range(multi_k)
        ])

    def forward(self, x):
        x,h=x
        if h == None:
            h = self.xgru(x)
        if hasattr(self, "sample"):
            x = self.sample(x)
            h = self.psample(h)
        x,h=self.model((x, h))
        h=self.qsample(h)
        return (x,h)


class threshold(torch.autograd.Function):
    """
    该层是为了脉冲激活分形设计，在原版模型使用，当前模型撤销了
    """

    @staticmethod
    def forward(ctx, input, sigma):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]
        ctx.save_for_backward(input)
        ctx.sigma = sigma
        output = input.clone()
        output = torch.max(torch.tensor(0.0, device=output.device), torch.sign(output))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigma = ctx.sigma
        exponent = -torch.pow((input), 2) / (2.0 * sigma ** 2)
        exp = torch.exp(exponent)
        erfc_grad = exp / (2.506628 * sigma)
        grad = erfc_grad * grad_output
        return grad, None


class Trinomial_operation(object):
    def __init__(self, max_n, tau_m=1., tau_s=4.):
        self.max_n = max_n
        self.tau_s = tau_s
        self.tau_m = tau_m
        self.Trinomial_list()

    def Trinomial_list(self):
        max_n = self.max_n
        self.diag = [1 for i in range(self.max_n * 3)]
        for i in range(1, max_n * 3):
            self.diag[i] = self.diag[i - 1] * (i)
        self.diag_T = torch.ones(self.max_n, self.max_n, self.max_n, dtype=torch.float32)
        for i in range(self.max_n):
            for j in range(self.max_n):
                for k in range(self.max_n):
                    self.diag_T[i][j][k] = self.diag[i + j + k] / (self.diag[i] * self.diag[j] * self.diag[k])

    def get_value(self, i, j, k):
        if i >= self.max_n or j >= self.max_n or k >= self.max_n:
            exit(-1)
        return self.diag_T[i][j][k]


class guassNet(nn.Module):
    """
    配置了两种卷积操作，gauss和ungauss卷积，但保证第三维度和第四维度和输入相同
    """

    def __init__(self, in_channel, out_channel, kernel_size=5, sigma=1., group=1, requires_grad=True, use_gauss=True):
        super(guassNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.group = group
        self.use_gauss = use_gauss
        self.requires_grad = requires_grad
        if use_gauss == True:
            self.gauss_kernel = self._gauss2D(self.kernel_size, self.sigma, self.group, self.in_channel,
                                              self.out_channel, self.requires_grad)
        else:
            self.gauss_kernel = self._conv2D(self.kernel_size, self.sigma, self.group, self.in_channel,
                                             self.out_channel, self.requires_grad)
        self.gauss_bias = Parameter(torch.zeros(self.in_channel), requires_grad=self.requires_grad)
        self.GaussianBlur = lambda x: F.conv2d(x, weight=self.gauss_kernel, bias=self.gauss_bias, stride=1,
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

    def _conv2D(self, kernel_size, sigma, group, in_channel, out_channel, requires_grad=False, ):
        kernel = torch.normal(.0, sigma, (in_channel, out_channel // group, kernel_size, kernel_size)).float()
        kernel.clamp_(-2 * sigma, 2 * sigma)
        kernel = Parameter(kernel, requires_grad=requires_grad)
        return kernel


def DiffInitial(data, shape, in_feature, out_feature, group=1):
    """
    grad计算层，用于grad计算
    K_{GX} = [-1 0 1 ; -2 0 2 ; -1 0 1], K_{GY} = {-1 -2 -1 ; 0 0 0 ; 1 2 1}
    """
    tmp = data.clone().detach().cuda()
    if in_feature == 1:
        tmp = tmp.view(shape).repeat(1, in_feature, 1, 1)
    else:
        tmp = tmp.view(shape)
    kernel_row = Parameter(
        torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).repeat(in_feature,
                                                                                            out_feature // group, 1,
                                                                                            1),
        requires_grad=False).cuda()
    kernel_col = Parameter(
        torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).repeat(in_feature,
                                                                                            out_feature // group, 1, 1),
        requires_grad=False).cuda()
    if dataoption in ['mnist', 'fashionmnist']:
        kernel_gauss = guassNet(1, 1, kernel_size=5, requires_grad=False, group=group).cuda()
    elif dataoption in ['cifar10', 'cifar100', 'svhn', 'car', 'stl-10']:
        kernel_gauss = guassNet(3, 3, kernel_size=5, requires_grad=False, group=group).cuda()
    else:
        raise KeyError("error shape")
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
    if dataoption in ['cifar10', 'car', "mnist", "fashionmnist", "svhn", "cifar100", "stl-10"]:
        grad = grad.view(-1, 3, 32 * 32)
        mean = grad.mean(dim=-1, keepdim=True)
        std = grad.std(dim=-1, keepdim=True)
        return ((grad - mean) / (std + 1e-6)).view_as(data), mean, std
    else:
        raise KeyError("not have this dataset")


class axonLimit(torch.autograd.Function):
    """
    阈值控制层，该层对每一层输出的数据压缩在一定范围内，并对超出范围外的数据在求导时指定其导数值避免无法求导
    """

    @staticmethod
    def forward(ctx, v1):
        ctx.save_for_backward(v1)
        # v1 = 1.3 * torch.sigmoid(v1) - 0.2
        # return v1
        if dataoption in ['cifar10', 'car', "svhn", "mnist", "fashionmnist", "cifar100", "stl-10"]:
            return torch.min(torch.max(v1, torch.Tensor([-1.5]).cuda()), torch.Tensor([1.5]).cuda())
        elif dataoption == 'eeg':
            return torch.min(torch.max(v1, torch.Tensor([-1.5]).cuda()), torch.Tensor([1.5]).cuda())
        else:
            raise KeyError()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if dataoption in ['cifar10', 'car', "fashionmnist", "mnist", "svhn", "cifar100", "stl-10"]:
            exponent = torch.where((input > -1.6) & (input < 1.6), torch.ones_like(input).cuda(),
                                   torch.zeros_like(input).cuda())
            exponent = torch.where((input > 1.6) | (input < -1.6),
                                   (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
            return exponent * grad_output
        elif dataoption == 'eeg':
            exponent = torch.where((input > -1.6) & (input < 1.6), torch.ones_like(input).cuda(),
                                   torch.zeros_like(input).cuda())
            exponent = torch.where((input > 1.6) | (input < -1.6),
                                   (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
            return exponent * grad_output
        else:
            raise KeyError('not have this dataset')


class DoorMechanism(nn.Module):
    def __init__(self, in_pointnum, out_pointnum, in_feature, out_feature, lr=.9):
        """
        门机制层，对三条路径传来的数据进行选择
        """
        super(DoorMechanism, self).__init__()
        self.lr = lr
        self.in_pointnum = in_pointnum
        self.out_pointnum = out_pointnum
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.inx = nn.Sequential(*[

            nn.Conv2d(self.in_feature * 3, self.in_feature, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.in_feature),
        ])
        self.intau = nn.Sequential(*[

            nn.Conv2d(self.in_feature * 3, self.in_feature, (3, 3), (1, 1), (3, 3), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.in_feature),
        ])
        self.convrx1 = nn.Conv2d(self.in_feature + self.out_feature, self.out_feature, (1, 1), (1, 1), bias=False)
        self.convhx1 = nn.Conv2d(self.in_feature + self.out_feature, self.out_feature, (1, 1), (1, 1), bias=False)
        self.convqx1 = nn.Conv2d(self.in_feature + self.out_feature, self.out_feature, (1, 1), (1, 1), (1, 1),
                                 bias=False)

    def forward(self, x1, x2, x3, tau_m, tau_s, tau_sm) -> tuple:
        """
        input==>[batchsize,64,x_pointnum//2,y_pointnum//2]
        """
        x1: torch.Tensor
        x2: torch.Tensor
        x3: torch.Tensor
        x = torch.cat([x1, x2, x3], dim=1)
        tau = torch.cat([tau_m, tau_s, tau_sm], dim=1)
        inx = self.inx(x)
        intau = self.intau(tau)
        inxt = torch.cat([intau, inx], dim=1)
        rx = torch.sigmoid(self.convrx1(inxt))
        hx = torch.sigmoid(self.convhx1(inxt))
        q = torch.tanh(self.convqx1(torch.cat([rx * intau, inx], dim=1)))
        tau = intau * (1 - hx) + q * hx
        return inx, tau


class point_cul_Layer(nn.Module):
    def __init__(self, in_pointnum, out_pointnum, in_feature, out_feature, path_len, tau_m=4., tau_s=1.,
                 grad_small=False,
                 weight_require_grad=False,
                 weight_rand=False, device=None, STuning=True, grad_lr=0.1, p=0.3, use_gauss=True, mult_k=2):
        """
        输入的张量维度为（batch_size,64,x//2,y//2）
        该层通过门机制后进行卷积与归一化
        """
        super(point_cul_Layer, self).__init__()
        self.device = device
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.weight_rand = weight_rand
        self.grad_small = grad_small
        self.weight_require_grad = weight_require_grad
        self.in_pointnum, self.out_pointnum = in_pointnum, out_pointnum
        self.in_size = int(math.sqrt(self.in_pointnum // self.in_feature))
        self.out_size = int(math.sqrt(self.out_pointnum // self.out_feature))
        hidden_size=128
        g = False
        self.DoorMach = DoorMechanism(in_pointnum, in_pointnum, in_feature, in_feature)
        if dataoption == 'mnist':
            self.gaussbur = multi_block_eq(in_feature, out_feature, hidden_size,multi_k=mult_k)
        elif dataoption == 'cifar10':
            self.gaussbur = multi_block_eq(in_feature, out_feature, hidden_size,multi_k=mult_k)
        elif dataoption == 'cifar100':
            self.gaussbur = multi_block_eq(in_feature, out_feature, hidden_size,multi_k=mult_k)
        elif dataoption == 'svhn':
            self.gaussbur = multi_block_eq(in_feature, out_feature, hidden_size,multi_k=mult_k)
        elif dataoption == 'car':
            self.gaussbur = multi_block_eq(in_feature, out_feature, hidden_size,multi_k=mult_k)
        elif dataoption == 'stl-10':
            self.gaussbur = multi_block_eq(in_feature, out_feature, hidden_size,multi_k=mult_k)
        elif dataoption == 'fashionmnist':
            self.gaussbur = multi_block_eq(in_feature, out_feature, hidden_size,multi_k=mult_k)
        elif dataoption == 'eeg':
            self.gaussbur = multi_block_eq(in_feature, out_feature, hidden_size,multi_k=mult_k)
        else:
            raise KeyError("not import gaussbur!")
        self.STuning = STuning
        self.grad_lr = grad_lr
        self.sigma = 1
        self.norm = None
        self.index = random.randint(0, path_len)
        self.bnx = nn.BatchNorm2d(in_feature, momentum=0.1)
        self.bntau = nn.BatchNorm2d(in_feature, momentum=0.1)
        self._initialize()

    def forward(self, x, tau):
        x1, x2, x3 = x.unbind(dim=-1)
        tau_m, tau_s, tau_sm = tau
        """
        Try 1
        加入门机制，我们其实可以把tau_s,tau_m,tau_sm想象成门，通过门的筛选来决定他该选择什么样的输出，也就是对路径上所提取特征的筛选。
        该步骤为了减少参数量我先采用使用筛选特征的方式
        """
        x, tau = self.DoorMach(x1, x2, x3, tau_m, tau_s, tau_sm)
        x, tau = self.gaussbur(x, tau)
        x, tau = self.bnx(x), self.bntau(x)
        return x, tau

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.bias.data)
                nn.init.ones_(m.weight.data)


class two_dim_layer(nn.Module):
    def __init__(self, in_feature, out_feature, in_pointnum, out_pointnum, x, y, weight_require_grad, weight_rand,
                 device, grad_lr, use_gauss=False, tau_m=4., tau_s=1., mult_k=2, p=0.2):
        super(two_dim_layer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.in_pointnum = in_pointnum
        self.out_pointnum = out_pointnum
        self.x = x
        self.y = y
        self.point_cul_layer = {}
        self.test = False
        for i in range(self.x):
            for j in range(self.y):
                if not (i == self.x - 1 and j == self.y - 1):
                    self.point_cul_layer[str(i) + "_" + str(j)] = point_cul_Layer(
                        in_pointnum,
                        in_pointnum,
                        path_len=self.y + self.x,
                        in_feature=in_feature,
                        out_feature=in_feature,
                        tau_m=tau_m,
                        tau_s=tau_s,
                        grad_small=False,
                        weight_require_grad=weight_require_grad,
                        weight_rand=weight_rand,
                        device=device,
                        mult_k=mult_k,
                        # bool((i+j+k)%2),False
                        STuning=bool(
                            (i + j) % 2),
                        grad_lr=grad_lr,
                        use_gauss=use_gauss)
                else:
                    self.point_cul_layer[str(i) + "_" + str(j)] = point_cul_Layer(
                        in_pointnum,
                        out_pointnum,
                        path_len=self.y + self.x,
                        in_feature=in_feature,
                        out_feature=out_feature,
                        tau_m=tau_m,
                        tau_s=tau_s,
                        grad_small=False,
                        weight_require_grad=weight_require_grad,
                        weight_rand=weight_rand,
                        device=device,
                        mult_k=mult_k,
                        # bool((i+j+k)%2),False
                        STuning=bool(
                            (i + j) % 2),
                        grad_lr=grad_lr,
                        use_gauss=use_gauss)
        self.point_layer_module = nn.ModuleDict(self.point_cul_layer)
        self.dropout = [[nn.Dropout(p) for i in range(self.x)] for j in range(self.y)]

    def forward(self, x, y, z, tau):

        tensor_prev = [[z for i in range(self.x)] for j in range(self.y)]
        tau_prev = [[tau for i in range(self.x)] for j in range(self.y)]
        for i in range(self.x):
            for j in range(self.y):
                zz = z
                tau_sm = tau + z
                if i == 0:
                    yy = y
                    tau_s = tau + y
                else:
                    yy = tensor_prev[j][i - 1]
                    tau_s = tau_prev[j][i - 1]
                if j == 0:
                    xx = x
                    tau_m = tau + y
                else:
                    xx = tensor_prev[j - 1][i]
                    tau_m = tau_prev[j - 1][i]
                tensor_prev[j][i], tau_prev[j][i] = self.point_layer_module[str(i) + '_' + str(j)](
                    torch.stack([xx, yy, zz], dim=-1), (tau_m, tau_s, tau_sm))
                if j != self.x - 1 and i != self.y - 1 and self.test == False:
                    tensor_prev[j][i] = self.dropout[j][i](tensor_prev[j][i])
        result, tau = tensor_prev[-1][-1].clone(), tau_prev[-1][-1].clone()
        del tensor_prev, tau_prev
        self._initialize()
        return result, tau

    def settest(self, test=True):
        self.test = test

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.bias.data)
                nn.init.ones_(m.weight.data)


class turn_layer(nn.Module):
    def __init__(self, in_feature, out_feature, size_change=False):
        super(turn_layer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.size_change = size_change
        self.push_conv = nn.ModuleList([
            nn.Conv2d(in_feature, in_feature, (1, 5), (1, 1), padding=(0, 2), bias=False),
            nn.Conv2d(in_feature, in_feature, (5, 1), (1, 1), padding=(2, 0), bias=False),
        ])
        if size_change == True:
            self.out_conv = nn.ModuleList([
                nn.Conv2d(in_feature * 3, out_feature, (2, 2), (2, 2), padding=(0, 0), bias=False),
                nn.Conv2d(in_feature * 3, out_feature, (2, 2), (2, 2), padding=(0, 0), bias=False),
                nn.Conv2d(in_feature * 3, out_feature, (2, 2), (2, 2), padding=(0, 0), bias=False),
            ])
        else:
            self.out_conv = nn.ModuleList([
                nn.Conv2d(in_feature * 3, out_feature, (1, 1), (1, 1), padding=(0, 0), bias=False),
                nn.Conv2d(in_feature * 3, out_feature, (1, 1), (1, 1), padding=(0, 0), bias=False),
                nn.Conv2d(in_feature * 3, out_feature, (1, 1), (1, 1), padding=(0, 0), bias=False),
            ])

    def forward(self, x):
        m = []
        m.append(x)
        for push in self.push_conv:
            x = push(x)
            m.append(x)
        m = torch.cat(m, dim=1)
        l = []
        for out in self.out_conv:
            l.append(out(m))
        del m
        return l


class three_dim_Layer(nn.Module):
    def __init__(self, shape, device, weight_require_grad=False, weight_rand=False, grad_lr=0.0001, p=0.1, test=False,
                 use_gauss=True):
        super(three_dim_Layer, self).__init__()
        """
        该层便是three-dim层
        x维度代表原始数据通过卷积变换至[batchsize,64,x//2,y//2]
        y维度代表原始数据先获得grad后经过卷积变换到[batchsize,64,x//2,y//2]
        z维度目前用0向量填充，未来可以改进
        """
        self.x, self.y, self.z = len(shape), len(shape), len(shape)
        self.shape = shape
        self.device = device
        self.use_gauss = use_gauss
        self.weight_require_grad = weight_require_grad
        self.weight_rand = weight_rand
        self.p = p
        self.diag_T = Trinomial_operation(max(self.x, self.y, self.z))
        self.grad_lr = grad_lr
        self.test = test
        self.x_join, self.y_join, self.z_join = LastJoiner(2), LastJoiner(2), LastJoiner(2)
        self.losses = 0.

    def settest(self, test=True):
        for module in self.point_layer_module.values():
            module.settest(test)

    def forward(self, x, y, z, tau=None):
        """
        x,y=>[batchsize,64,x_pointnum//2,y_pointnum//2]
        """
        x = x
        y = y
        z = z
        old = [[x, y, z], ]
        if tau == None:
            tau = self.tau_layer(torch.cat([x, y, z], dim=1))
        self.losses = 0.
        for num in range(max(self.z, self.y, self.x)):
            xx = y
            yy = z
            zz = x
            if num < self.z:
                out_1, tau = self.point_layer_module[str(num) + '_' + str(0)](xx, yy, zz, tau)
            else:
                out_1 = zz.clone()
            xx, yy, zz = self.turn_layer_module[str(num) + '_' + str(0)](out_1)
            if num < self.x:
                out_2, tau = self.point_layer_module[str(num) + '_' + str(1)](xx, yy, zz, tau)
            else:
                out_2 = zz.clone()
            xx, yy, zz = self.turn_layer_module[str(num) + '_' + str(0)](out_2)
            if num < self.x:
                out_3, tau = self.point_layer_module[str(num) + '_' + str(2)](xx, yy, zz, tau)
            else:
                out_3 = zz.clone()
            xx, yy, zz = self.turn_layer_module[str(num) + '_' + str(0)](out_3)
            x = xx + self.change_conv[num + 0](x)
            y = yy + self.change_conv[num + 1](y)
            z = zz + self.change_conv[num + 2](z)
            old.append([x, y, z])
        del old
        return x, y, z, tau

    def initiate_layer(self, data, in_feature, out_feature, tau_m=4., tau_s=1., use_gauss=True, mult_k=2,
                       set_share_layer=True, use_feature_change=True):
        """
        three-dim层初始化节点
        """
        self.use_gauss = use_gauss
        self.point_layer = {}
        self.turn_layer = {}
        self.in_shape = data.shape
        self.tau_layer = nn.Conv2d(in_feature[0] * 3, in_feature[0], (1, 1), (1, 1), bias=False)
        if use_feature_change == True:
            self.feature_len = [in_feature[i] for i in range(max(self.x, self.y, self.z) + 1)]
            self.div_len = [2 ** i for i in range(max(self.x, self.y, self.z) + 1)]
        else:
            self.feature_len = [int(in_feature[0])] * (max(self.x, self.y, self.z) + 1)
            self.div_len = [1] * (max(self.x, self.y, self.z) + 1)
        old_size = int(math.sqrt(data.shape[1] / in_feature[0]))
        for i in range(max(self.x, self.y, self.z) + 1):
            if old_size % (2 ** i) == 0:
                pass
            else:
                self.feature_len[i] = self.feature_len[i - 1]
                self.div_len[i] = self.div_len[i - 1]

        self.change_conv = nn.ModuleList([])
        for num in range(max(self.z, self.y, self.x)):
            a, b, c = self.shape[num]
            tmp_list = [a, b, c]
            for i, tmp in enumerate(tmp_list):
                in_pointnum = int(data.shape[1] // self.div_len[num])
                in_feature = self.feature_len[num]
                out_pointnum = int(data.shape[1] // self.div_len[num + 1])
                out_feature = self.feature_len[num + 1]
                if i == 2:
                    self.point_layer[str(num) + "_" + str(i)] = two_dim_layer(in_feature=in_feature,
                                                                              out_feature=out_feature,
                                                                              in_pointnum=in_pointnum,
                                                                              out_pointnum=out_pointnum, mult_k=mult_k,
                                                                              use_gauss=use_gauss,
                                                                              tau_m=tau_m, tau_s=tau_s, x=(tmp),
                                                                              y=(tmp),
                                                                              weight_rand=self.weight_rand,
                                                                              weight_require_grad=self.weight_require_grad,
                                                                              p=self.p, device=self.device,
                                                                              grad_lr=self.grad_lr)
                    self.turn_layer[str(num) + "_" + str(i)] = turn_layer(out_feature, out_feature, size_change=False)
                else:
                    self.point_layer[str(num) + "_" + str(i)] = two_dim_layer(in_feature=in_feature,
                                                                              out_feature=in_feature,
                                                                              in_pointnum=in_pointnum,
                                                                              out_pointnum=in_pointnum, mult_k=mult_k,
                                                                              use_gauss=use_gauss,
                                                                              tau_m=tau_m, tau_s=tau_s, x=(tmp),
                                                                              y=(tmp),
                                                                              weight_rand=self.weight_rand,
                                                                              weight_require_grad=self.weight_require_grad,
                                                                              p=self.p, device=self.device,
                                                                              grad_lr=self.grad_lr)
                    self.turn_layer[str(num) + "_" + str(i)] = turn_layer(out_feature, in_feature, size_change=False)
                if use_feature_change == True:
                    size_m = 2
                else:
                    size_m = 1
                self.change_conv.append(nn.Conv2d(in_feature, out_feature, (size_m, size_m), (size_m, size_m)))
                self.change_conv.append(nn.Conv2d(in_feature, out_feature, (size_m, size_m), (size_m, size_m)))
                self.change_conv.append(nn.Conv2d(in_feature, out_feature, (size_m, size_m), (size_m, size_m)))
        self.turn_layer_module = nn.ModuleDict(self.turn_layer)
        self.point_layer_module = nn.ModuleDict(self.point_layer)
        del self.point_layer, self.turn_layer
        return self.feature_len[-1], max(self.z, self.y, self.x)

    def set_share_twodimlayer(self, layer1, layer2, layer3, tmp_list):
        layer1: two_dim_layer
        layer2: two_dim_layer
        layer3: two_dim_layer
        layer3.point_layer_module[str(0) + '_' + str(0)] = layer2.point_layer_module[str(0) + '_' + str(0)] = \
            layer1.point_layer_module[str(0) + '_' + str(0)]
        for i in range(1, tmp_list[0]):
            layer2.point_layer_module[str(0) + '_' + str(i)] = layer1.point_layer_module[str(i) + '_' + str(0)]
            layer3.point_layer_module[str(0) + '_' + str(i)] = layer2.point_layer_module[str(i) + '_' + str(0)]
            layer1.point_layer_module[str(0) + '_' + str(i)] = layer3.point_layer_module[str(i) + '_' + str(0)]


class multi_two_dim_layer(nn.Module):
    def __init__(self, shape, device, weight_require_grad=False, weight_rand=False, grad_lr=0.0001, p=0.1, test=False,
                 use_gauss=True):
        super(multi_two_dim_layer, self).__init__()
        """
        该层便是three-dim层
        x维度代表原始数据通过卷积变换至[batchsize,64,x//2,y//2]
        y维度代表原始数据先获得grad后经过卷积变换到[batchsize,64,x//2,y//2]
        z维度目前用0向量填充，未来可以改进
        """
        self.len = len(shape)
        self.shape = shape
        self.device = device
        self.use_gauss = use_gauss
        self.weight_require_grad = weight_require_grad
        self.weight_rand = weight_rand
        self.p = p
        self.grad_lr = grad_lr
        self.test = test

    def settest(self, test=True):
        for module in self.point_layer_module.values():
            module.settest(test)

    def forward(self, x, y, z, tau=None):
        """
        x,y=>[batchsize,64,x_pointnum//2,y_pointnum//2]
        """
        self.losses = 0.
        if tau == None:
            tau = self.tau_layer(torch.cat([x, y, z], dim=1))
        for num in range(self.len):
            xx = y
            yy = z
            zz = x
            out, tau = self.point_layer_module[str(num) + '_' + str(0)](xx, yy, zz, tau)
            xx, yy, zz = self.turn_layer_module[str(num) + '_' + str(0)](out)
            x = xx + self.change_conv[num + 0](x)
            y = yy + self.change_conv[num + 1](y)
            z = zz + self.change_conv[num + 2](z)
        return x, y, z, tau

    def initiate_layer(self, data, in_feature, out_feature, tau_m=4., tau_s=1., use_gauss=True, mult_k=2,
                       set_share_layer=True, use_feature_change=True):
        """
        three-dim层初始化节点
        """
        self.use_gauss = use_gauss
        self.point_layer = {}
        self.turn_layer = {}
        self.in_shape = data.shape
        self.tau_layer = nn.Conv2d(in_feature[0] * 3, in_feature[0], (1, 1), (1, 1), bias=False)
        if use_feature_change == True:
            self.feature_len = [in_feature[i] for i in range(self.len + 1)]
            self.div_len = [2 ** i for i in range(self.len + 1)]
        else:
            self.feature_len = [int(in_feature[0])] * (self.len + 1)
            self.div_len = [1] * (self.len + 1)
        old_size = int(math.sqrt(data.shape[1] / in_feature[0]))
        for i in range(self.len + 1):
            if old_size % (2 ** i) == 0:
                pass
            else:
                self.feature_len[i] = self.feature_len[i - 1]
                self.div_len[i] = self.div_len[i - 1]

        self.change_conv = nn.ModuleList([])
        for num in range(self.len):
            l = self.shape[num]
            in_pointnum = int(data.shape[1] // self.div_len[num])
            in_feature = self.feature_len[num]
            out_pointnum = int(data.shape[1] // self.div_len[num + 1])
            out_feature = self.feature_len[num + 1]
            self.point_layer[str(num) + "_" + str(0)] = two_dim_layer(in_feature=in_feature,
                                                                      out_feature=out_feature,
                                                                      in_pointnum=in_pointnum,
                                                                      out_pointnum=out_pointnum, mult_k=mult_k,
                                                                      use_gauss=use_gauss,
                                                                      tau_m=tau_m, tau_s=tau_s, x=(l),
                                                                      y=(l),
                                                                      weight_rand=self.weight_rand,
                                                                      weight_require_grad=self.weight_require_grad,
                                                                      p=self.p, device=self.device,
                                                                      grad_lr=self.grad_lr)
            self.turn_layer[str(num) + "_" + str(0)] = turn_layer(out_feature, out_feature, size_change=False)
            if use_feature_change == True:
                size_m = 2
            else:
                size_m = 1
            self.change_conv.append(nn.Conv2d(in_feature, out_feature, (size_m, size_m), (size_m, size_m)))
            self.change_conv.append(nn.Conv2d(in_feature, out_feature, (size_m, size_m), (size_m, size_m)))
            self.change_conv.append(nn.Conv2d(in_feature, out_feature, (size_m, size_m), (size_m, size_m)))
        self.turn_layer_module = nn.ModuleDict(self.turn_layer)
        self.point_layer_module = nn.ModuleDict(self.point_layer)
        del self.point_layer, self.turn_layer
        return self.feature_len[-1], self.len


class merge_layer(nn.Module):
    def __init__(self, device, shape=None, weight_require_grad=True, weight_rand=True, grad_lr=0.01, dropout=0.3,
                 test=False):
        """
        该层是basic层,包含了特征变换层和three-dim路径选择层
        """
        super(merge_layer, self).__init__()
        if shape == None:
            self.shape = [2, 2, 2]
        else:
            self.shape = shape
        self.device = device
        self.InputGenerateNet = multi_two_dim_layer(self.shape, self.device, weight_require_grad, weight_rand, grad_lr,
                                                    dropout,
                                                    test).to(device)
        self.feature_loss = DimixLoss_neg(0)
        self.time = 0
        self.eq = nn.Sequential(multi_block_eq(16, 128,128, 2, stride=1),
                                multi_block_eq(128, 256,128, 2, stride=2),
                                multi_block_eq(256, 512,128, 2, stride=2))
        self.cl = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(512, 10)
        ])
        self.inf = nn.Conv2d(3, 16, (3, 3), (1, 1), (1, 1), bias=False)

    def forward(self, x):
        # x, y = self.initdata(x)
        if hasattr(self, 'input_shape'):
            x = x.view(self.input_shape)
        else:
            if dataoption in ['cifar10', 'cifar100']:
                x = x.view(x.shape[0], 3, 32, 32)
                # y = y.view(y.shape[0], 3, 32, 32)
            elif dataoption == 'mnist':
                x: torch.Tensor
                x = x.view(x.shape[0], 1, 28, 28)
                x = F.interpolate(x, (32, 32), mode='bilinear', align_corners=True)
                # y = y.view(y.shape[0], 1, 28, 28)
            elif dataoption == 'fashionmnist':
                x = x.view(x.shape[0], 1, 28, 28)
                x = F.interpolate(x, (32, 32), mode='bilinear', align_corners=True)
                # y = y.view(y.shape[0], 1, 28, 28)
            elif dataoption == 'eeg':
                x = x.view(x.shape[0], 14, 32, 32)
                # 64,16,16
            elif dataoption == 'car':
                x = x.view(x.shape[0], 3, 64, 64)
                # 64,16,16
            elif dataoption == 'svhn':
                x = x.view(x.shape[0], 3, 32, 32)

            elif dataoption == "stl-10":
                x = x.view(x.shape[0], 3, 96, 96)
            else:
                raise KeyError()
        m = self.inf(x)
        # a,b,c = self.adaptiveconv[0](m),self.adaptiveconv[1](m),self.adaptiveconv[2](m)
        # x = self.reconv(x)
        # a,b,c,tau = self.InputGenerateNet(a, b, c)
        # self.kl_loss=self.feature_loss(a,b)+self.feature_loss(b,c)+self.feature_loss(c,a)
        m, p = self.eq((m,None))
        m = self.cl(F.avg_pool2d(m, m.shape[-1]))
        # m= self.out_classifier(m)
        return m

    def initiate_layer(self, input, in_feature, out_feature, classes, tmp_feature=64, tau_m=4., tau_s=1.,
                       use_gauss=True, batchsize=64, mult_k=2, p=0.2, use_share_layer=True, push_num=5, s=2,
                       use_feature_change=True):
        """
        配置相应的层
        """
        b, c, h, w = input.shape
        self.filter_list = [16, 16 * tmp_feature, 2 * 16 * tmp_feature, 4 * 16 * tmp_feature]
        self.k_f_list = [16 * tmp_feature, 2 * 16 * tmp_feature, 4 * 16 * tmp_feature]
        input = torch.rand(b, self.k_f_list[0], h, w).to(input.device)
        self.reconv = nn.Conv2d(self.filter_list[-1], self.k_f_list[0], (1, 1), (1, 1), bias=False)
        self.adaptiveconv = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(self.filter_list[0], 8 * tmp_feature, (1, 5), (1, 1), (0, 2), bias=False),
                nn.ELU(),
                nn.Conv2d(8 * tmp_feature, self.k_f_list[0], (5, 1), (1, 1), (2, 0), bias=False),
                nn.AvgPool2d((4, 4), (4, 4))
            ]),
            nn.Sequential(*[
                nn.Conv2d(self.filter_list[0], 8 * tmp_feature, (5, 1), (1, 1), (2, 0), bias=False),
                nn.ELU(),
                nn.Conv2d(8 * tmp_feature, self.k_f_list[0], (1, 5), (1, 1), (0, 2), bias=False),
                nn.AvgPool2d((4, 4), (4, 4))
            ]),
            nn.Sequential(*[
                nn.Conv2d(self.filter_list[0], 8 * tmp_feature, (1, 1), (1, 1), (0, 0), bias=False),
                nn.ELU(),
                nn.Conv2d(8 * tmp_feature, self.k_f_list[0], (3, 3), (1, 1), (1, 1), bias=False),
                nn.AvgPool2d((4, 4), (4, 4))
            ]),
        ])
        import copy
        feature_len, size_div = self.InputGenerateNet.initiate_layer(input,
                                                                     copy.deepcopy(self.k_f_list),
                                                                     copy.deepcopy(self.k_f_list),
                                                                     tau_m,
                                                                     tau_s,
                                                                     use_gauss,
                                                                     mult_k=mult_k,
                                                                     set_share_layer=use_share_layer,
                                                                     use_feature_change=use_feature_change)
        if use_feature_change == True:
            size_len = [h, h // 2, h // 4, int(h // (2 ** (size_div + 2)))]
        else:
            size_len = [h, h // 2, h // 4, h // 4]
        self.feature_forward = Feature_forward([in_feature] + self.filter_list, None, push_num=push_num, s=s, p=p)
        self.out_classifier = block_out(feature_len, classes=classes, size=size_len,
                                        use_pool='avg')

    def settest(self, test=True):
        """
        令模型知道当前处理test
        """
        for layer in self.modules():
            if isinstance(layer, multi_two_dim_layer) or isinstance(layer, block_in) or isinstance(layer, block_out):
                layer.settest(test)

    @staticmethod
    def _list_build():
        return [2., 0.75, 0.01, 0.1]

    @staticmethod
    def _list_print(list):
        for i in list:
            print(i.squeeze().item(), end=",")
        print("")

    def L2_biasoption(self, loss_list, sigma=None):
        if sigma == None:
            sigma = self._list_build()
        loss = [torch.tensor(0.).float().cuda()]
        normlist = []
        loss_feature = torch.tensor([0.]).float().cuda()
        loss_kl = torch.tensor([0.]).float().cuda()
        loss_tau = torch.tensor(0.).float().cuda()
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) and layer.bias is not None:
                layer: nn.Conv2d
                loss.append(torch.norm(torch.abs(layer.bias.data) - 1., p=2) / layer.bias.data.numel())
            elif isinstance(layer, point_cul_Layer):
                layer: point_cul_Layer
                if hasattr(layer, "norm"):
                    normlist.append(layer.norm)
            elif isinstance(layer, multi_two_dim_layer):
                # layer: InputGenerateNet
                # loss_feature += layer.three_dim_layer.losses
                # len += 1
                pass
            elif isinstance(layer, merge_layer):
                layer: merge_layer
                if hasattr(layer, "kl_loss"):
                    loss_kl += layer.kl_loss.squeeze()
            elif isinstance(layer, DoorMechanism):
                layer: DoorMechanism
                if hasattr(layer, "norm_mem_1"):
                    loss_tau += (layer.norm_mem_1 + layer.norm_mem_2 + layer.norm_mem_3)
        loss_feature = (loss_feature.squeeze(-1)) * sigma[0]
        loss_kl = (loss_kl * sigma[1]).squeeze(-1)
        loss_tau = loss_tau * sigma[2]
        loss_bias = torch.stack(loss, dim=-1).mean() * sigma[3]
        loss_list = loss_list + [loss_bias, loss_kl, loss_feature, loss_tau]
        # self._list_print(loss_list)
        loss = torch.stack(loss_list, dim=-1).sum()
        return loss


"""
高斯卷积
输出卷积核大小:output=[(input+padding*2-kernel_size)/stride]+1
group是需要根据通道进行调整的，三通道的卷积操作如果group为1则虽然输出三通道，但其实三通道信息是相同的，当然根据group能够做到通道之间的信息交互
如果是反卷积则output=[(input-1)*stride+kernel_size-padding*2]
如果需要自己实现卷积操作则需要通过torch.unfold将input进行分片，对于卷积操作中的卷积核而言，其尺寸为（输出通道数，输入通道数，卷积核长，卷积核宽）
对于输入数据而言通常为（数据批次数，特征数，点状图高，点状图宽），对于每个输出通道数维度，其（输入通道数，卷积核长，卷积核宽）和（特征数，点状图高，点状图宽）进行卷积操作，然后将输出通道数维度的结果进行拼接得到输出
其中特别要注意group,如果group不为一，那么其实卷积核尺寸为(输出通道数，输入通道数/groups,卷积核长，卷积核宽)，这时候其实卷积操作对通道的整合性减弱，而对输入信息所具备的特征信息增强
以下为实现卷积操作的函数：
def conv2d(x, weight, bias, stride, pad): 
    n, c, h_in, w_in = x.shape
    d, c, k, j = weight.shape
    x_pad = torch.zeros(n, c, h_in+2*pad, w_in+2*pad)   # 对输入进行补零操作
    if pad>0:
        x_pad[:, :, pad:-pad, pad:-pad] = x
    else:
        x_pad = x

    x_pad = x_pad.unfold(2, k, stride)
    x_pad = x_pad.unfold(3, j, stride)        # 按照滑动窗展开
    out = torch.einsum(                          # 按照滑动窗相乘，
        'nchwkj,dckj->ndhw',                    # 并将所有输入通道卷积结果累加
        x_pad, weight)
    out = out + bias.view(1, -1, 1, 1)          # 添加偏置值
    return out
"""
