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
from Snn_Auto_master.lib.dimixloss import DimIxLoss
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
        return self.function(x,x.size()[-1], *args)


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
        num_input_feature_list_list[0][0] = in_feature
        num_input_feature_list_list[-1][-1] = out_feature
        if dataoption in ["mnist", "fashionmnist", "cifar100", "cifar10", "car", "svhn", "stl-10"]:
            self.block_in_layer = DenseNet(num_input_feature_list_list,bn_size,drop_rate,use_size_change,num_layer)
        elif dataoption == "eeg":
            self.block_in_layer = DenseNet(num_input_feature_list_list,bn_size,drop_rate,use_size_change,num_layer)
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

    def settest(self, training_status):
        self.training = training_status

    def forward(self, x):
        x = self.block_in_layer(x)
        m = size_change(3 * self.out_feature, x.size()[-1] // 2)
        x = self.relu(self.conv_cat(x) + m(x))
        a, b, c = torch.split(x, dim=1, split_size_or_sections=[x.size()[1] // 3, x.size()[1] // 3, x.size()[1] // 3])
        a, b, c = a + self.f_conv[0](x), b + self.f_conv[1](x), c + self.f_conv[2](x)
        del x
        return a, b, c


class block_out(nn.Module):
    def __init__(self, feature, fl_feature, classes, size, use_pool='none'):
        super(block_out, self).__init__()

        if use_pool == 'none':
            self.classifiar = nn.Sequential(nn.Flatten(), nn.Linear((feature * 4 * 4), classes))
            self.classifiar_1 = nn.Sequential(nn.Flatten(), nn.Linear((fl_feature * 8 * 8), classes))
            self.classifiar_2 = nn.Sequential(nn.Flatten(), nn.Linear((fl_feature * 8 * 8), classes))
            self.classifiar_3 = nn.Sequential(nn.Flatten(), nn.Linear((fl_feature * 8 * 8), classes))
        else:
            self.classifiar = nn.Sequential(nn.Flatten(), nn.Linear(feature, classes))
            self.classifiar_1 = nn.Sequential(nn.Flatten(), nn.Linear(fl_feature, classes))
            self.classifiar_2 = nn.Sequential(nn.Flatten(), nn.Linear(fl_feature, classes))
            self.classifiar_3 = nn.Sequential(nn.Flatten(), nn.Linear(fl_feature, classes))
        self.transition_layer = nn.Sequential(*[
            nn.BatchNorm2d(feature),
            nn.ReLU(inplace=True),
            Lambda(F.avg_pool2d)])
        self.fl_transition_layer = nn.Sequential(*[
            nn.BatchNorm2d(fl_feature),
            nn.ReLU(inplace=True),
            Lambda(F.avg_pool2d)])
        self.training = False
        self.use_pool = use_pool
        self.size = size

    def settest(self, training_status):
        self.training = training_status

    def forward(self, x, a, b, c):
        x = self.transition_layer(x)
        a = self.fl_transition_layer(a)
        b = self.fl_transition_layer(b)
        c = self.fl_transition_layer(c)
        if self.use_pool == 'none':
            return self.classifiar(x), self.classifiar_1(a), self.classifiar_2(b), self.classifiar_3(c)
        elif self.use_pool == 'max':
            return self.classifiar(F.max_pool2d(x, x.shape[-1])), self.classifiar_1(F.max_pool2d(a, a.shape[-1])) \
                , self.classifiar_2(F.max_pool2d(b, b.shape[-1])), self.classifiar_3(F.max_pool2d(c, c.shape[-1]))
        elif self.use_pool == 'avg':
            return self.classifiar(F.avg_pool2d(x, x.shape[-1])), self.classifiar_1(F.avg_pool2d(a, a.shape[-1])) \
                , self.classifiar_2(F.avg_pool2d(b, b.shape[-1])), self.classifiar_3(F.avg_pool2d(c, c.shape[-1]))


class block_eq(nn.Module):
    def __init__(self, eq_feature, size, Use_Spectral=False, Use_fractal=False):
        super(block_eq, self).__init__()
        self.eq_feature = eq_feature
        self.longConv = nn.Sequential(*[
            nn.ReflectionPad2d(1),
            nn.Conv2d(eq_feature, eq_feature * 2, (3, 3), stride=_pair(1), padding=0,
                      bias=True) if Use_Spectral == False else SNConv2d(eq_feature, eq_feature * 2, (3, 3), stride=1,
                                                                        padding=0, bias=False),
            nn.BatchNorm2d(eq_feature * 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(eq_feature * 2, eq_feature, (3, 3), stride=_pair(1), padding=0,
                      bias=True) if Use_Spectral == False else SNConv2d(eq_feature * 2, eq_feature, (3, 3), stride=1,
                                                                        padding=0, bias=False),
            nn.BatchNorm2d(eq_feature),
        ])
        self.shortConv = nn.Sequential(*[
            Shortcut(eq_feature, eq_feature, use_same=True),
        ])
        self.shortConv_1 = nn.Sequential(*[
            nn.ReflectionPad2d(1),
            nn.Conv2d(eq_feature, eq_feature, (3, 3), stride=_pair(1), padding=0,
                      bias=True) if Use_Spectral == False else SNConv2d(eq_feature, eq_feature, (3, 3), stride=1,
                                                                        padding=0, bias=True),
            nn.BatchNorm2d(eq_feature),
            Shortcut(eq_feature, eq_feature, use_same=True),
        ])
        self.Use_fractal = Use_fractal
        if self.Use_fractal is True:
            self.merged = LastJoiner(2)

    def forward(self, x):
        if self.Use_fractal == False:
            x1 = self.longConv(x) + self.shortConv(x)
        else:
            x1 = self.merged([self.longConv(x), self.shortConv_1(x)])
        x2 = F.relu(x1, inplace=True)
        del x
        return x2


class multi_block_eq(nn.Module):
    def __init__(self, eq_feature, size, multi_k=1, Use_Spactral=False, Use_fractal=False):
        super(multi_block_eq, self).__init__()
        self.eq_feature = eq_feature
        self.model = nn.Sequential(*[
            block_eq(self.eq_feature, size, Use_Spectral=Use_Spactral, Use_fractal=Use_fractal) for _ in range(multi_k)
        ])

    def forward(self, x):
        return self.model(x)


class multi_block_neq(nn.Module):
    def __init__(self, in_feature, out_feature, size, multi_k=1, Use_Spactral=False, Use_fractal=False):
        super(multi_block_neq, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.model = nn.Sequential(*[
            block_eq(self.in_feature, size, Use_Spectral=Use_Spactral, Use_fractal=Use_fractal) for _ in range(multi_k)
        ])
        if Use_Spactral == True:
            self.out = nn.Sequential(SNConv2d(in_feature, out_feature, (4, 4), stride=2, padding=1), )
        else:
            self.out = nn.Sequential(nn.Conv2d(in_feature, out_feature, (4, 4), stride=2, padding=1), )

    def forward(self, x):
        return self.out(self.model(x))


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
        self.tau_m_weight1 = Parameter(torch.Tensor(in_feature, out_feature),
                                       requires_grad=True)
        self.tau_m_weight2 = Parameter(torch.Tensor(in_feature, out_feature),
                                       requires_grad=True)
        self.tau_s_weight1 = Parameter(torch.Tensor(in_feature, out_feature),
                                       requires_grad=True)
        self.tau_s_weight2 = Parameter(torch.Tensor(in_feature, out_feature),
                                       requires_grad=True)
        self.tau_sm_weight1 = Parameter(torch.Tensor(in_feature, out_feature),
                                        requires_grad=True)
        self.tau_sm_weight2 = Parameter(torch.Tensor(in_feature, out_feature),
                                        requires_grad=True)
        stdv = 6. / math.sqrt((in_pointnum // in_feature) * (out_pointnum // out_feature))
        self.tau_m_weight1.data.uniform_(-stdv, stdv)
        self.tau_m_weight2.data.uniform_(-stdv, stdv)
        self.tau_s_weight1.data.uniform_(-stdv, stdv)
        self.tau_s_weight2.data.uniform_(-stdv, stdv)
        self.tau_sm_weight1.data.uniform_(-stdv, stdv)
        self.tau_sm_weight2.data.uniform_(-stdv, stdv)
        self.tau_m_bias = Parameter(torch.zeros((1, out_feature)).float(), requires_grad=True)
        self.tau_s_bias = Parameter(torch.zeros((1, out_feature)).float(), requires_grad=True)
        self.tau_sm_bias = Parameter(torch.zeros((1, out_feature)).float(), requires_grad=True)

    def forward(self, x1, x2, x3, tau_m, tau_s, tau_sm) -> tuple:
        """
        input==>[batchsize,64,x_pointnum//2,y_pointnum//2]
        """
        x1: torch.Tensor
        x2: torch.Tensor
        x3: torch.Tensor
        y1 = x1.view(x1.shape[0], x1.shape[1], -1).mean(dim=-1).mean(dim=0, keepdim=True)
        y2 = x2.view(x2.shape[0], x2.shape[1], -1).mean(dim=-1).mean(dim=0, keepdim=True)
        y3 = x3.view(x3.shape[0], x3.shape[1], -1).mean(dim=-1).mean(dim=0, keepdim=True)  # [batchsize,feature]
        x1 = x1 + torch.eye(x1.shape[-1], dtype=torch.float32).to(x1.device).unsqueeze(0).unsqueeze(0)
        x2 = x2 + torch.eye(x2.shape[-1], dtype=torch.float32).to(x2.device).unsqueeze(0).unsqueeze(0)
        x3 = x3 + torch.eye(x3.shape[-1], dtype=torch.float32).to(x3.device).unsqueeze(0).unsqueeze(0)
        men_1 = torch.sigmoid(y1 @ self.tau_m_weight2 + tau_m @ self.tau_m_weight1 + self.tau_m_bias)
        men_2 = torch.sigmoid(y2 @ self.tau_s_weight2 + tau_s @ self.tau_s_weight1 + self.tau_s_bias)
        men_3 = torch.sigmoid(y3 @ self.tau_sm_weight2 + tau_sm @ self.tau_sm_weight1 + self.tau_sm_bias)
        self.norm_mem_1 = men_1.norm(p=2, dim=0).mean() / (men_1.numel() / men_1.size()[0])
        self.norm_mem_2 = men_2.norm(p=2, dim=0).mean() / (men_2.numel() / men_2.size()[0])
        self.norm_mem_3 = men_3.norm(p=2, dim=0).mean() / (men_3.numel() / men_3.size()[0])
        result = torch.tanh(
            men_1.unsqueeze(-1).unsqueeze(-1) * x1 + men_2.unsqueeze(-1).unsqueeze(-1) * x2 + men_3.unsqueeze(
                -1).unsqueeze(-1) * x3)
        with torch.no_grad():
            men_1 = (men_1 * self.lr + (1. - self.lr) * tau_m)
            men_2 = (men_2 * self.lr + (1. - self.lr) * tau_s)
            men_3 = (men_3 * self.lr + (1. - self.lr) * tau_sm)
        del tau_s, tau_m, tau_sm, x1, x2, x3, y1, y2, y3
        return (result, men_1, men_2, men_3)


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
        g = False
        self.tensor_tau_m1 = torch.rand((1, in_feature), dtype=torch.float32, requires_grad=g).to(
            self.device)
        self.tensor_tau_s1 = torch.rand((1, in_feature), dtype=torch.float32, requires_grad=g).to(
            self.device)
        self.tensor_tau_sm1 = torch.rand((1, in_feature), dtype=torch.float32, requires_grad=g).to(
            self.device)
        self.DoorMach = DoorMechanism(in_pointnum, in_pointnum, in_feature, in_feature)
        if dataoption == 'mnist':
            if use_gauss == True:
                if in_feature == out_feature:
                    self.gaussbur = guassNet(in_feature, in_feature, kernel_size=3, requires_grad=True)
                else:
                    self.gaussbur = guassNet(in_feature, out_feature, kernel_size=3, requires_grad=True)
            else:
                if in_feature == out_feature:
                    self.gaussbur = multi_block_eq(in_feature, self.in_size, multi_k=mult_k, Use_Spactral=True,
                                                   Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, self.in_size, multi_k=mult_k,
                                                    Use_Spactral=True,
                                                    Use_fractal=True)
            self.bn1 = nn.BatchNorm2d(out_feature)
        elif dataoption == 'cifar10':
            if use_gauss == True:
                if in_feature == out_feature:
                    self.gaussbur = guassNet(in_feature, in_feature, kernel_size=3, requires_grad=True)
                else:
                    self.gaussbur = guassNet(in_feature, out_feature, kernel_size=3, requires_grad=True)
            else:
                if in_feature == out_feature:
                    self.gaussbur = multi_block_eq(in_feature, self.in_size, multi_k=mult_k, Use_Spactral=True,
                                                   Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, self.in_size, multi_k=mult_k,
                                                    Use_Spactral=True,
                                                    Use_fractal=True)
            self.bn1 = nn.BatchNorm2d(out_feature)
        elif dataoption == 'cifar100':
            if use_gauss == True:
                if in_feature == out_feature:
                    self.gaussbur = guassNet(in_feature, in_feature, kernel_size=3, requires_grad=True)
                else:
                    self.gaussbur = guassNet(in_feature, out_feature, kernel_size=3, requires_grad=True)
            else:
                if in_feature == out_feature:
                    self.gaussbur = multi_block_eq(in_feature, self.in_size, multi_k=mult_k, Use_Spactral=True,
                                                   Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, self.in_size, multi_k=mult_k,
                                                    Use_Spactral=True,
                                                    Use_fractal=True)
            self.bn1 = nn.BatchNorm2d(out_feature)
        elif dataoption == 'svhn':
            if use_gauss == True:
                if in_feature == out_feature:
                    self.gaussbur = guassNet(in_feature, in_feature, kernel_size=3, requires_grad=True)
                else:
                    self.gaussbur = guassNet(in_feature, out_feature, kernel_size=3, requires_grad=True)
            else:
                if in_feature == out_feature:
                    self.gaussbur = multi_block_eq(in_feature, self.in_size, multi_k=mult_k, Use_Spactral=True,
                                                   Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, self.in_size, multi_k=mult_k,
                                                    Use_Spactral=True,
                                                    Use_fractal=True)
            self.bn1 = nn.BatchNorm2d(out_feature)
        elif dataoption == 'car':
            if use_gauss == True:
                if in_feature == out_feature:
                    self.gaussbur = guassNet(in_feature, in_feature, kernel_size=3, requires_grad=True)
                else:
                    self.gaussbur = guassNet(in_feature, out_feature, kernel_size=3, requires_grad=True)
            else:
                if in_feature == out_feature:
                    self.gaussbur = multi_block_eq(in_feature, self.in_size, multi_k=mult_k, Use_Spactral=True,
                                                   Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, self.in_size, multi_k=mult_k,
                                                    Use_Spactral=True,
                                                    Use_fractal=True)
        elif dataoption == 'stl-10':
            if use_gauss == True:
                if in_feature == out_feature:
                    self.gaussbur = guassNet(in_feature, in_feature, kernel_size=3, requires_grad=True)
                else:
                    self.gaussbur = guassNet(in_feature, out_feature, kernel_size=3, requires_grad=True)
            else:
                if in_feature == out_feature:
                    self.gaussbur = multi_block_eq(in_feature, self.in_size, multi_k=mult_k, Use_Spactral=True,
                                                   Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, self.in_size, multi_k=mult_k,
                                                    Use_Spactral=True,
                                                    Use_fractal=True)
            self.bn1 = nn.BatchNorm2d(out_feature)
        elif dataoption == 'fashionmnist':
            if use_gauss == True:
                if in_feature == out_feature:
                    self.gaussbur = guassNet(in_feature, in_feature, kernel_size=3, requires_grad=True)
                else:
                    self.gaussbur = guassNet(in_feature, out_feature, kernel_size=3, requires_grad=True)
            else:
                if in_feature == out_feature:
                    self.gaussbur = multi_block_eq(in_feature, self.in_size, multi_k=mult_k, Use_Spactral=True,
                                                   Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, self.in_size, multi_k=mult_k,
                                                    Use_Spactral=True,
                                                    Use_fractal=True)
            self.bn1 = nn.BatchNorm2d(out_feature)
        elif dataoption == 'eeg':
            if use_gauss == True:
                if in_feature == out_feature:
                    self.gaussbur = guassNet(in_feature, in_feature, kernel_size=3, requires_grad=True)
                else:
                    self.gaussbur = guassNet(in_feature, out_feature, kernel_size=3, requires_grad=True)
            else:
                if in_feature == out_feature:
                    self.gaussbur = multi_block_eq(in_feature, self.in_size, multi_k=mult_k, Use_Spactral=True,
                                                   Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, self.in_size, multi_k=mult_k,
                                                    Use_Spactral=True,
                                                    Use_fractal=True)
            self.bn1 = nn.BatchNorm2d(out_feature)
        else:
            raise KeyError("not import gaussbur!")
        self.STuning = STuning
        self.grad_lr = grad_lr
        self.sigma = 1
        self.norm = None
        self.index = random.randint(0, path_len)
        # self._initialize()

    def forward(self, x):
        x1, x2, x3 = x.unbind(dim=-1)
        """
        Try 1
        加入门机制，我们其实可以把tau_s,tau_m,tau_sm想象成门，通过门的筛选来决定他该选择什么样的输出，也就是对路径上所提取特征的筛选。
        该步骤为了减少参数量我先采用使用筛选特征的方式
        """
        x, self.tensor_tau_m1, self.tensor_tau_s1, self.tensor_tau_sm1 = self.DoorMach(x1, x2, x3, self.tensor_tau_m1,
                                                                                       self.tensor_tau_s1,
                                                                                       self.tensor_tau_sm1)
        m = self.gaussbur(x)
        m = F.relu(m)
        x = self.bn1(m)
        return x

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")


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

    def forward(self, x, y, z):
        tensor_prev = [[z for i in range(self.x)] for j in range(self.y)]
        for i in range(self.x):
            for j in range(self.y):
                zz = z
                if i == 0:
                    yy = y
                else:
                    yy = tensor_prev[j][i - 1]
                if j == 0:
                    xx = x
                else:
                    xx = tensor_prev[j - 1][i]
                tensor_prev[j][i] = self.point_layer_module[str(i) + '_' + str(j)](
                    torch.stack([xx, yy, zz], dim=-1))
                tensor_prev[j][i] = axonLimit.apply(tensor_prev[j][i])
                if j != self.x - 1 and i != self.y - 1 and self.test == False:
                    tensor_prev[j][i] = self.dropout[j][i](tensor_prev[j][i])
        result = tensor_prev[-1][-1].clone()
        del tensor_prev
        return result

    def settest(self, test=True):
        self.test = test


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
        self.feature_loss = DimIxLoss(max(self.x, self.y, self.z))
        self.test = test
        self.x_join, self.y_join, self.z_join = LastJoiner(2), LastJoiner(2), LastJoiner(2)
        self.losses = 0.

    def settest(self, test=True):
        for module in self.point_layer_module.values():
            module.settest(test)

    def forward(self, x, y, z):
        """
        x,y=>[batchsize,64,x_pointnum//2,y_pointnum//2]
        """
        x = x
        y = y
        z = z
        old = [[x, y, z], ]
        for num in range(max(self.z, self.y, self.x)):
            xx = y
            yy = z
            zz = x
            if num < self.z:
                out_1 = self.point_layer_module[str(num) + '_' + str(0)](xx, yy, zz)
            else:
                out_1 = zz.clone()

            xx = z
            yy = x
            zz = y

            if num < self.x:
                out_2 = self.point_layer_module[str(num) + '_' + str(1)](xx, yy, zz)
            else:
                out_2 = zz.clone()

            xx = x
            yy = y
            zz = z

            if num < self.x:
                out_3 = self.point_layer_module[str(num) + '_' + str(2)](xx, yy, zz)
            else:
                out_3 = zz.clone()
            m = size_change(out_1.shape[1], out_1.shape[2])
            x = out_1 + m(x)
            y = out_2 + m(y)
            z = out_3 + m(z)
            old.append([x, y, z])
        self.losses = self.feature_loss(old)
        for i in range(len(old[:-1])):
            for j in range(3):
                old[i][j] = self.change_conv[i * 3 + j](old[i][j])
                old[-1][j] = old[-1][j] + old[i][j]
        m = len(old)
        for i in range(m - 1):
            del old[0]
        x, y, z = old[-1]
        return x, y, z

    def initiate_layer(self, data, in_feature, out_feature, tau_m=4., tau_s=1., use_gauss=True, mult_k=2,
                       set_share_layer=True, use_feature_change=True):
        """
        three-dim层初始化节点
        """
        self.use_gauss = use_gauss
        self.point_layer = {}
        self.in_shape = data.shape
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
            in_pointnum = int(data.shape[1] // self.div_len[num])
            in_feature = self.feature_len[num]
            out_pointnum = int(data.shape[1] // self.div_len[num + 1])
            out_feature = self.feature_len[num + 1]
            for p in range(3):
                self.point_layer[str(num) + "_" + str(p)] = two_dim_layer(in_feature=in_feature,
                                                                          out_feature=out_feature,
                                                                          in_pointnum=in_pointnum,
                                                                          out_pointnum=out_pointnum, mult_k=mult_k,
                                                                          use_gauss=use_gauss,
                                                                          tau_m=tau_m, tau_s=tau_s, x=(tmp_list[p]),
                                                                          y=(tmp_list[p]),
                                                                          weight_rand=self.weight_rand,
                                                                          weight_require_grad=self.weight_require_grad,
                                                                          p=self.p, device=self.device,
                                                                          grad_lr=self.grad_lr)
                size_m = 2 ** (num + 1)
                self.change_conv.append(
                    nn.Conv2d(in_feature, self.feature_len[max(self.z, self.y, self.x)], (size_m, size_m),
                              (size_m, size_m), bias=False))

            if set_share_layer == True:
                self.set_share_twodimlayer(self.point_layer[str(num) + "_0"], self.point_layer[str(num) + "_1"],
                                           self.point_layer[str(num) + "_2"], tmp_list)
        self.point_layer_module = nn.ModuleDict(self.point_layer)
        del self.point_layer
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


class InputGenerateNet(nn.Module):
    def __init__(self, shape, device, weight_require_grad, weight_rand, grad_lr, dropout, test):
        super(InputGenerateNet, self).__init__()
        self.shape = shape
        self.device = device
        self.weight_require_grad = weight_require_grad
        self.weight_rand = weight_rand
        self.grad_lr = grad_lr
        self.dropout = dropout
        self.test = test
        self.three_dim_layer = three_dim_Layer(self.shape, self.device, weight_require_grad, weight_rand, grad_lr,
                                               p=dropout, test=test)

    def forward(self, x, y, z):
        return self.three_dim_layer(x, y, z)

    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data,mode="fan_in",nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.normal_(layer.bias.data,0,0.01)
                    layer.bias.data=layer.bias.data.abs()
            elif isinstance(layer,nn.BatchNorm2d):
                layer.weight.data.fill_(1.)
                layer.bias.data.zero_()

    def initiate_layer(self, input, in_feature, out_feature, tau_m=4., tau_s=1., use_gauss=True, batchsize=64,
                       old_in_feature=1, old_out_feature=1, mult_k=2, p=0.2, use_share_layer=True,
                       use_feature_change=True):
        b, c, h, w = input.shape
        r = self.three_dim_layer.initiate_layer(
            torch.rand(batchsize, in_feature[0] * h * w),
            in_feature,
            out_feature,
            tau_m=tau_m,
            tau_s=tau_s,
            use_gauss=use_gauss,
            mult_k=mult_k,
            set_share_layer=use_share_layer,
            use_feature_change=use_feature_change)
        self._initialize()
        return r

    def settest(self, test):
        self.three_dim_layer.settest(test)


class merge_layer(nn.Module):
    def __init__(self, device, shape=None, weight_require_grad=True, weight_rand=True, grad_lr=0.01, dropout=0.3,
                 test=False):
        """
        该层是basic层,包含了特征变换层和three-dim路径选择层
        """
        super(merge_layer, self).__init__()
        if shape == None:
            self.shape = [[2, 2, 2], [1, 1, 1]]
        else:
            self.shape = shape
        self.device = device
        self.InputGenerateNet = InputGenerateNet(self.shape, self.device, weight_require_grad, weight_rand, grad_lr,
                                                 dropout,
                                                 test).to(device)
        self.time = 0

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
        a, b, c = self.block_in_x_y_z(x)
        a_1, b_1, c_1 = self.InputGenerateNet(a, b, c)
        x = self.feature_forward(x, a_1, b_1, c_1)
        h = self.out_classifier(x, a_1, b_1, c_1)
        return h

    def initiate_layer(self, input, in_feature, out_feature, classes, tmp_feature=64, tau_m=4., tau_s=1.,
                       use_gauss=True, batchsize=64, mult_k=2, p=0.2, use_share_layer=True, push_num=5, s=2,
                       use_feature_change=True):
        """
        配置相应的层
        """
        b, c, h, w = input.shape
        self.filter_list = [16, 16 * tmp_feature, 2 * 16 * tmp_feature, 4 * 16 * tmp_feature]
        self.block_in_x_y_z = block_in(in_feature, self.filter_list[0], p=p)
        import copy
        feature_len, size_div = self.InputGenerateNet.initiate_layer(input,
                                                                     copy.deepcopy(self.filter_list),
                                                                     copy.deepcopy(self.filter_list),
                                                                     tau_m,
                                                                     tau_s,
                                                                     use_gauss,
                                                                     batchsize,
                                                                     old_in_feature=in_feature,
                                                                     old_out_feature=out_feature,
                                                                     mult_k=mult_k,
                                                                     use_share_layer=use_share_layer,
                                                                     use_feature_change=use_feature_change)
        for i, nums in enumerate(self.filter_list):
            if nums == feature_len:
                self.mk = i
        if not hasattr(self, "mk"):
            raise KeyError
        if use_feature_change == True:
            h = int(h // (2 ** (size_div + 1)))
        else:
            h = int(h // 2)
        size_len = [h, h, h // 2]
        feature_len = (copy.deepcopy([in_feature] + self.filter_list), self.mk)
        self.feature_forward = Feature_forward(feature_len, size_len, push_num=push_num, s=s, p=p)
        self.out_classifier = block_out(feature_len[0][-1], feature_len[0][self.mk + 1], classes=classes, size=size_len,
                                        use_pool='avg')

    def settest(self, test=True):
        """
        令模型知道当前处理test
        """
        for layer in self.modules():
            if isinstance(layer, InputGenerateNet) or isinstance(layer, block_in) or isinstance(layer, block_out):
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
        len = torch.tensor(0.).float().cuda()
        loss_tau = torch.tensor(0.).float().cuda()
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) and layer.bias is not None:
                layer: guassNet
                loss.append(torch.norm(torch.abs(layer.bias.data) - 1., p=2) / layer.bias.data.numel())
            elif isinstance(layer, guassNet):
                layer: guassNet
                loss.append(torch.norm(torch.abs(layer.gauss_bias.data) - 1., p=2) / layer.gauss_bias.data.numel())
            elif isinstance(layer, point_cul_Layer):
                layer: point_cul_Layer
                normlist.append(layer.norm)
            elif isinstance(layer, InputGenerateNet):
                layer: InputGenerateNet
                loss_feature += layer.three_dim_layer.losses
                len += 1
            elif isinstance(layer, Feature_forward):
                layer: Feature_forward
                loss_kl += layer.kl_loss.squeeze()
            elif isinstance(layer, DoorMechanism):
                layer: DoorMechanism
                if hasattr(layer, "norm_mem_1"):
                    loss_tau += (layer.norm_mem_1 + layer.norm_mem_2 + layer.norm_mem_3)
        loss_feature = (loss_feature.squeeze(-1) / len) * sigma[0]
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
