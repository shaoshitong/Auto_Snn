import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from Snn_Auto_master.lib.activation import Activation
from Snn_Auto_master.lib.data_loaders import revertNoramlImgae
from Snn_Auto_master.lib.plt_analyze import vis_img
from Snn_Auto_master.lib.parameters_check import pd_save, parametersgradCheck
from Snn_Auto_master.lib.SNnorm import SNConv2d, SNLinear
from Snn_Auto_master.lib.fractallayer import LastJoiner
from Snn_Auto_master.lib.DenseNet import Denselayer
from Snn_Auto_master.lib.cocoscontextloss import ContextualLoss_forward
from Snn_Auto_master.lib.featurefocusing import Feature_forward
import math
import pandas as pd
from torch.nn.parameter import Parameter
from omegaconf import OmegaConf
import matplotlib.pyplot as plt


def yaml_config_get():
    """
    该函数是为了获取模型的配置信息
    """
    conf = OmegaConf.load('./train.yaml')
    return conf


def batch_norm(input):
    input_linear = input.view(input.shape[0], -1)
    mean = input_linear.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    std = input_linear.std(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    return torch.div(torch.sub(input, mean), std)


yaml = yaml_config_get()
dataoption = yaml['data']


def size_change(f, s):
    def change(xx):
        xx: torch.Tensor
        xx = F.interpolate(xx, (s, s), mode='bilinear', align_corners=True)
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


# class block_in(nn.Module):
#     def __init__(self, in_feature, out_feature=64):
#         super(block_in, self).__init__()
#         self.conv0 = nn.Conv2d(in_feature, in_feature, (1, 1), stride=1, padding=0, bias=False)
#         self.conv1 = nn.Conv2d(in_feature, 32, (4, 4), stride=2, padding=1, bias=True, )
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, out_feature, (3, 3), stride=1, padding=1, bias=True)
#         self.bn2 = nn.BatchNorm2d(out_feature)
#         self.shortcut2 = Shortcut(32, out_feature, use_same=True)
#         self.shortcut1 = Shortcut(in_feature, 32)
#         self.shortcut0 = Shortcut(in_feature, out_feature)
#
#     def forward(self, x):
#         x = self.conv0(x)
#         x1 = self.bn1(self.conv1(x)) + self.shortcut1(x)
#         x1 = F.leaky_relu(x1, inplace=True)
#         x2 = self.bn2(self.conv2(x1)) + self.shortcut2(x1)
#         x2 = F.leaky_relu(x2, inplace=True)
#         x3 = x2 + self.shortcut0(x)
#         x3 = F.relu_(x3)
#         return x3

class block_in(nn.Module):
    def __init__(self, in_feature, out_feature=64):
        super(block_in, self).__init__()
        self.block_in_layer=Denselayer([in_feature,32,32,out_feature,out_feature])
        self.conv_cat=nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(out_feature,3*out_feature,(4,4),stride=2,padding=0),
                                    nn.BatchNorm2d(3*out_feature),)
        self.out_feature=out_feature
        self.relu=nn.LeakyReLU(1e-1)
        self.f_conv=nn.ModuleList([SNConv2d(3*out_feature,out_feature,(1,1),stride=1,padding=0) for _ in range(3)])
        self.training=False
    def settest(self,training_status):
        self.training=training_status
    def forward(self,x):
        x=self.block_in_layer(x,not self.training)
        m=size_change(3*self.out_feature,x.size()[-1]//2)
        x=self.relu(self.conv_cat(x)+m(x))
        a,b,c=torch.split(x,dim=1,split_size_or_sections=[x.size()[1]//3,x.size()[1]//3,x.size()[1]//3])
        a,b,c=a+self.f_conv[0](x),b+self.f_conv[0](x),c+self.f_conv[0](x)
        del x
        return a,b,c
class block_out(nn.Module):
    def __init__(self, in_feature, out_feature, classes, size, use_pool="max"):
        super(block_out, self).__init__()
        self.tmp_feature = int(math.sqrt(in_feature * out_feature))
        self.bn_out = nn.BatchNorm2d(in_feature)
        self.conv2 = nn.Conv2d(in_feature, self.tmp_feature, (2, 2), stride=2, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(self.tmp_feature)
        self.conv2_1 = nn.Conv2d(in_feature, out_feature, (2, 2), stride=2, padding=0, bias=True)
        self.bn2_1 = nn.BatchNorm2d(out_feature)
        self.shortcut2 = Shortcut(in_feature, self.tmp_feature)
        self.shortcut2_1 = Shortcut(in_feature, out_feature)
        self.shortcut1 = Shortcut(self.tmp_feature, out_feature)
        self.shortcut0 = Shortcut(in_feature, out_feature, proportion=4)
        self.shortcut0_1 = Shortcut(in_feature, out_feature, proportion=2)
        self.conv1 = nn.Conv2d(self.tmp_feature, out_feature, (2, 2), stride=2, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(out_feature)
        if use_pool == 'max':
            self.maxpool_1 = nn.MaxPool2d(kernel_size=_pair(1), padding=0, stride=1)
            self.maxpool = nn.MaxPool2d(kernel_size=_pair(1), padding=0, stride=1)
            self.linear = nn.Sequential(*[
                nn.Flatten(),
                nn.Linear(size * size * out_feature // 16, classes)
            ])
        elif use_pool == 'avg':
            self.maxpool_1 = nn.AvgPool2d(kernel_size=_pair(1), padding=0, stride=1)
            self.maxpool = nn.AvgPool2d(kernel_size=_pair(1), padding=0, stride=1)
            self.linear = nn.Sequential(*[
                nn.Flatten(),
                nn.Linear(size * size * out_feature // 16, classes)
            ])
        elif use_pool == 'none':
            self.linear_1 = nn.Sequential(*[
                nn.Flatten(),
                nn.Linear(size * size * out_feature // 16, classes)
            ])
            pass
        self.use_pool = use_pool

    def forward(self, x):
        if dataoption == 'cifar10' or dataoption == 'fashionmnist' or dataoption == 'mnist':
            x1 = self.bn_out(x) + x
            x2 = self.bn2(self.conv2(F.relu_(x1)))  # + self.shortcut2(x1)  # [32,8,8]
            x3 = self.bn1(self.conv1(F.relu_(x2))) + self.shortcut0(x)  # + self.shortcut1(x2)+self.shortcut0(x)
            if self.use_pool != 'none':
                x3 = self.maxpool(F.leaky_relu_(x3))
                x3 = self.linear(x3)
            else:
                x3 = self.linear_1(x3)
        else:
            raise KeyError("not import")
        # elif dataoption == 'mnist' or dataoption == 'fashionmnist':
        #     x1 = self.bn_out(x) + x
        #     x2 = self.bn2_1(self.conv2_1(F.relu_(x1))) + self.shortcut0_1(
        #         x)  # + self.shortcut2_1(x1)+self.shortcut0_1(x)  # [32,8,8]
        #     if self.use_pool != 'none':
        #         if self.use_pool == 'max':
        #             relu = F.relu_
        #         else:
        #             relu = F.leaky_relu_
        #         x2 = self.maxpool_1(relu(x2))
        #         x3 = self.linear2(x2)
        #     else:
        #         x3 = self.linear2_1(x2)
        #
        # else:
        #     raise KeyError('not is True')
        return x3


class block_eq(nn.Module):
    def __init__(self, eq_feature, Use_Spectral=False, Use_fractal=False):
        super(block_eq, self).__init__()
        self.eq_feature = eq_feature
        self.longConv = nn.Sequential(*[
            nn.ReflectionPad2d(1),
            nn.Conv2d(eq_feature, eq_feature * 2, (3, 3), stride=_pair(1), padding=0,
                      bias=True) if Use_Spectral == False else SNConv2d(eq_feature, eq_feature * 2, (3, 3), stride=1,
                                                                        padding=0, bias=True),
            nn.BatchNorm2d(eq_feature * 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(eq_feature * 2, eq_feature, (3, 3), stride=_pair(1), padding=0,
                      bias=True) if Use_Spectral == False else SNConv2d(eq_feature * 2, eq_feature, (3, 3), stride=1,
                                                                        padding=0, bias=True),
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
        self.bn_eq = nn.BatchNorm2d(eq_feature)
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
    def __init__(self, eq_feature, multi_k=1, Use_Spactral=False, Use_fractal=False):
        super(multi_block_eq, self).__init__()
        self.eq_feature = eq_feature
        self.model = nn.Sequential(*[
            block_eq(self.eq_feature, Use_Spectral=Use_Spactral, Use_fractal=Use_fractal) for _ in range(multi_k)
        ])

    def forward(self, x):
        return self.model(x)


class multi_block_neq(nn.Module):
    def __init__(self, in_feature, out_feature, multi_k=1, Use_Spactral=False, Use_fractal=False):
        super(multi_block_neq, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.model = nn.Sequential(*[
            block_eq(self.in_feature, Use_Spectral=Use_Spactral, Use_fractal=Use_fractal) for _ in range(multi_k)
        ])
        if Use_Spactral == True:
            self.out = SNConv2d(in_feature, out_feature, (4, 4), stride=2, padding=1)
        else:
            self.out = nn.Sequential(nn.LeakyReLU(1e-2, inplace=True),
                                     nn.Conv2d(in_feature, out_feature, (4, 4), stride=2, padding=1),
                                     nn.BatchNorm2d(out_feature, affine=False))

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
        if dataoption == 'mnist':
            output = torch.max(torch.tensor(0.0, device=output.device), torch.sign(output))
        elif dataoption == 'cifar10':
            output = torch.max(torch.tensor(0.0, device=output.device), torch.sign(output))
        elif dataoption == 'fashionmnist':
            output = torch.max(torch.tensor(0.0, device=output.device), torch.sign(output))
        else:
            raise KeyError()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigma = ctx.sigma
        if dataoption == 'mnist':
            exponent = -torch.pow((input), 2) / (2.0 * sigma ** 2)
        elif dataoption == 'cifar10':
            exponent = -torch.pow((input), 2) / (2.0 * sigma ** 2)
        elif dataoption == 'fashionmnist':
            exponent = -torch.pow((input), 2) / (2.0 * sigma ** 2)
        else:
            raise KeyError()
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
    if dataoption == 'mnist':
        kernel_gauss = guassNet(1, 1, kernel_size=5, requires_grad=False, group=group).cuda()
    elif dataoption == 'cifar10':
        kernel_gauss = guassNet(3, 3, kernel_size=5, requires_grad=False, group=group).cuda()
    elif dataoption == 'fashionmnist':
        kernel_gauss = guassNet(1, 1, kernel_size=5, requires_grad=False, group=group).cuda()
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
    if dataoption == 'mnist':
        grad = grad.view(-1, 1, 28 * 28)
        mean = grad.mean(dim=-1, keepdim=True)
        std = grad.std(dim=-1, keepdim=True)
        return ((grad - mean) / (std + 1e-6)).view_as(data), mean, std
    elif dataoption == 'cifar10':
        grad = grad.view(-1, 3, 32 * 32)
        mean = grad.mean(dim=-1, keepdim=True)
        std = grad.std(dim=-1, keepdim=True)
        return ((grad - mean) / (std + 1e-6)).view_as(data), mean, std
    elif dataoption == 'fashionmnist':
        grad = grad.view(-1, 1, 28 * 28)
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
        if dataoption == 'mnist':
            return torch.min(torch.max(v1, torch.Tensor([-1.5]).cuda()), torch.Tensor([1.5]).cuda())
        elif dataoption == 'cifar10':
            return torch.min(torch.max(v1, torch.Tensor([-1.5]).cuda()), torch.Tensor([1.5]).cuda())
        elif dataoption == 'fashionmnist':
            return torch.min(torch.max(v1, torch.Tensor([-1.5]).cuda()), torch.Tensor([1.5]).cuda())
        else:
            raise KeyError()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if dataoption == 'mnist':
            exponent = torch.where((input > -1.6) & (input < 1.6), torch.ones_like(input).cuda(),
                                   torch.zeros_like(input).cuda())
            exponent = torch.where((input > 1.6) | (input < -1.6),
                                   (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
            return exponent * grad_output
        elif dataoption == 'cifar10':
            exponent = torch.where((input > -1.6) & (input < 1.6), torch.ones_like(input).cuda(),
                                   torch.zeros_like(input).cuda())
            exponent = torch.where((input > 1.6) | (input < -1.6),
                                   (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
            return exponent * grad_output
        elif dataoption == 'fashionmnist':
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
        # x1 = (x1 * self.pointx_s).sum(dim=-2, keepdims=True) + (x1 * self.pointy_s).sum(dim=-1,
        #                                                                                 keepdims=True) + torch.eye(
        #     x1.shape[-1], dtype=torch.float32).to(x1.device).unsqueeze(0).unsqueeze(0)
        # x2 = (x2 * self.pointx_m).sum(dim=-2, keepdims=True) + (x2 * self.pointy_m).sum(dim=-1,
        #                                                                                 keepdims=True) + torch.eye(
        #     x2.shape[-1], dtype=torch.float32).to(x2.device).unsqueeze(0).unsqueeze(0)
        # x3 = (x3 * self.pointx_sm).sum(dim=-2, keepdims=True) + (x3 * self.pointy_sm).sum(dim=-1,
        #                                                                                   keepdims=True) + torch.eye(
        #     x3.shape[-1], dtype=torch.float32).to(x3.device).unsqueeze(0).unsqueeze(0)
        x1 = x1 + torch.eye(x1.shape[-1], dtype=torch.float32).to(x1.device).unsqueeze(0).unsqueeze(0)
        x2 = x2 + torch.eye(x2.shape[-1], dtype=torch.float32).to(x2.device).unsqueeze(0).unsqueeze(0)
        x3 = x3 + torch.eye(x3.shape[-1], dtype=torch.float32).to(x3.device).unsqueeze(0).unsqueeze(0)
        # 2.y1 = (torch.stack([x1.mean(dim=-1), x1.mean(dim=-2)], dim=-1).view(x1.shape[0], x1.shape[1], -1) @ F.softmax(
        #     self.feature_s_sift1, dim=0)).squeeze(
        #     -1).mean(dim=0, keepdim=True)
        # y2 = (torch.stack([x2.mean(dim=-1), x2.mean(dim=-2)], dim=-1).view(x2.shape[0], x2.shape[1], -1) @ F.softmax(
        #     self.feature_m_sift1, dim=0)).squeeze(
        #     -1).mean(dim=0, keepdim=True)
        # y3 = (torch.stack([x3.mean(dim=-1), x3.mean(dim=-2)], dim=-1).view(x3.shape[0], x3.shape[1], -1) @ F.softmax(
        #     self.feature_sm_sift1, dim=0)).squeeze(
        #     -1).mean(dim=0, keepdim=True)
        men_1 = torch.sigmoid(y1 @ self.tau_m_weight2 + tau_m @ self.tau_m_weight1 + self.tau_m_bias)
        men_2 = torch.sigmoid(y2 @ self.tau_s_weight2 + tau_s @ self.tau_s_weight1 + self.tau_s_bias)
        men_3 = torch.sigmoid(y3 @ self.tau_sm_weight2 + tau_sm @ self.tau_sm_weight1 + self.tau_sm_bias)
        self.norm_mem_1 = men_1.norm(p=2, dim=0).mean() / (men_1.numel() / men_1.size()[0])
        self.norm_mem_2 = men_2.norm(p=2, dim=0).mean() / (men_2.numel() / men_2.size()[0])
        self.norm_mem_3 = men_3.norm(p=2, dim=0).mean() / (men_3.numel() / men_3.size()[0])
        # x1=x1*torch.tanh(self.feature_s_sift1[:self.len_point,:].view(1,1,1,self.len_point)+self.feature_s_sift1[self.len_point:,:].view(1,1,self.len_point,1))
        # x2=x2*torch.tanh(self.feature_m_sift1[:self.len_point,:].view(1,1,1,self.len_point)+self.feature_m_sift1[self.len_point:,:].view(1,1,self.len_point,1))
        # x3=x3*torch.tanh(self.feature_sm_sift1[:self.len_point,:].view(1,1,1,self.len_point)+self.feature_sm_sift1[self.len_point:,:].view(1,1,self.len_point,1))
        result = torch.tanh(
            men_1.unsqueeze(-1).unsqueeze(-1) * x1 +men_2.unsqueeze(-1).unsqueeze(-1) * x2 + men_3.unsqueeze(-1).unsqueeze(-1) * x3)
        with torch.no_grad():
            men_1 = (men_1 * self.lr + (1. - self.lr) * tau_m)
            men_2 = (men_2 * self.lr + (1. - self.lr) * tau_s)
            men_3 = (men_3 * self.lr + (1. - self.lr) * tau_sm)
        del tau_s,tau_m,tau_sm,x1,x2,x3,y1,y2,y3
        return (result, men_1, men_2, men_3)


class point_cul_Layer(nn.Module):
    def __init__(self, in_pointnum, out_pointnum, in_feature, out_feature, path_len, tau_m=4., tau_s=1.,
                 grad_small=False,
                 weight_require_grad=False,
                 weight_rand=False, device=None, STuning=True, grad_lr=0.1, p=0.3, use_gauss=True):
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
                    self.gaussbur = multi_block_eq(in_feature, multi_k=2, Use_Spactral=True,
                                                   Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, multi_k=2, Use_Spactral=True,
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
                    self.gaussbur = multi_block_eq(in_feature, multi_k=2, Use_Spactral=True, Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, multi_k=2, Use_Spactral=True,
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
                    self.gaussbur = multi_block_eq(in_feature, multi_k=2, Use_Spactral=True,
                                                   Use_fractal=True)
                else:
                    self.gaussbur = multi_block_neq(in_feature, out_feature, multi_k=2, Use_Spactral=True,
                                                    Use_fractal=True)
            self.bn1 = nn.BatchNorm2d(out_feature)
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
        if self.weight_rand:
            if dataoption == 'mnist':
                m = self.gaussbur(x)
            elif dataoption == 'cifar10':
                # x:torch.Tensor
                # m,p = self.maxpool[0](torch.abs(x))
                # m=self.maxpool[1](self.maxpool[0](x)[0],p)
                m = self.gaussbur(x)

            elif dataoption == 'fashionmnist':
                m = self.gaussbur(x)
            else:
                raise KeyError('not have this dataset')
            # print(self.gaussbur.weight.data.abs().max())
            x = F.leaky_relu(m)
        # self.norm = (torch.norm(x, p=2) / (x.numel())).detach()
        # print(torch.svd(x1[0,0,:,:])[1],torch.svd(x2[0,0,:,:])[1],torch.svd(x3[0,0,:,:])[1],torch.svd(x[0,0,:,:])[1])
        return x

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")

    def subWeightGrad(self, epoch, epochs, sigma, diag_num, norm):
        """
        用三项式定理以及链式法则作为数学原理进行梯度修改
        """

        weight = float(sigma) * diag_num * self.norm / norm
        for name, param in self.named_parameters():
            if param.requires_grad == True and param.grad != None:
                param.grad /= weight


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
        self.x, self.y, self.z = shape
        self.device = device
        self.use_gauss = use_gauss
        self.weight_require_grad = weight_require_grad
        self.weight_rand = weight_rand
        self.diag_T = Trinomial_operation(max(max(self.x, self.y), self.z))
        self.grad_lr = grad_lr
        self.dropout = [[[nn.Dropout(p) for i in range(self.x)] for j in range(self.y)] for k in range(self.z)]
        self.context = ContextualLoss_forward(False)
        self.test = test
        self.x_join, self.y_join, self.z_join = LastJoiner(2), LastJoiner(2), LastJoiner(2)
        self.losses = 0.

    def settest(self, test=True):
        self.test = test

    def forward(self, x, y, z):
        """
        x,y=>[batchsize,64,x_pointnum//2,y_pointnum//2]
        """
        losses = self.context(x, y, z)
        x=torch.tanh(x)
        y=torch.tanh(y)
        z=torch.tanh(z)
        old = [[x, y, z], ]
        tensor_prev = [[[z for i in range(self.x)] for j in range(self.y)] for k in range(self.z)]
        for num in range(max(self.z, self.y, self.x)):
            fo_list = []
            if num < self.z:
                for i in range(num, self.y):
                    for j in range(num, self.x):
                        zz = z
                        if j == num:
                            yy = y
                        else:
                            yy = tensor_prev[num][i][j - 1]
                        if i == num:
                            xx = x
                        else:
                            xx = tensor_prev[num][i - 1][j]
                        tensor_prev[num][i][j] = self.point_layer_module[str(num) + '_' + str(i) + '_' + str(j)](
                            torch.stack([xx, yy, zz], dim=-1))
                        tensor_prev[num][i][j] = axonLimit.apply(tensor_prev[num][i][j])
                        if j != self.x - 1 and i != self.y - 1 and num != self.z - 1 and self.test == False:
                            tensor_prev[num][i][j] = self.dropout[num][i][j](tensor_prev[num][i][j])
            else:
                tensor_prev[-1][self.y - 1][self.x - 1] = z
            if num == max(self.z, self.y, self.x):
                fo_list.append(tensor_prev[-1][self.y - 1][self.x - 1].clone())
            if num < self.y:
                for i in range(num, self.z):
                    for j in range(num, self.x):
                        yy = y
                        if i == num:
                            xx = x
                        else:
                            xx = tensor_prev[i - 1][num][j]
                        if j == num:
                            zz = z
                        else:
                            zz = tensor_prev[i][num][j - 1]
                        tensor_prev[i][num][j] = self.point_layer_module[str(i) + '_' + str(num) + '_' + str(j)](
                            torch.stack([xx, yy, zz], dim=-1))
                        tensor_prev[i][num][j] = axonLimit.apply(tensor_prev[i][num][j])
                        if j != self.x - 1 and num != self.y - 1 and i != self.z - 1 and self.test == False:
                            tensor_prev[i][num][j] = self.dropout[i][num][j](tensor_prev[i][num][j])
            else:
                tensor_prev[self.z - 1][-1][self.x - 1] = y
            if num == max(self.z, self.y, self.x):
                fo_list.append(tensor_prev[-1][self.y - 1][self.x - 1].clone())
            if num < self.x:
                for i in range(num, self.z):
                    for j in range(num, self.y):
                        xx = x
                        if i == num:
                            yy = y
                        else:
                            yy = tensor_prev[i - 1][j][num]
                        if j == num:
                            zz = z
                        else:
                            zz = tensor_prev[i][j - 1][num]
                        tensor_prev[i][j][num] = self.point_layer_module[str(i) + '_' + str(j) + '_' + str(num)](
                            torch.stack([xx, yy, zz], dim=-1))
                        tensor_prev[i][j][num] = axonLimit.apply(tensor_prev[i][j][num])
                        if num != self.x - 1 and j != self.y - 1 and i != self.z - 1 and self.test == False:
                            tensor_prev[i][j][num] = self.dropout[i][j][num](tensor_prev[i][j][num])
            else:
                tensor_prev[self.z - 1][self.y - 1][-1] = x
            if num == max(self.z, self.y, self.x):
                fo_list.append(tensor_prev[-1][self.y - 1][self.x - 1].clone())
            m = size_change(tensor_prev[self.z - 1][self.y - 1][min(self.x - 1, num)].shape[1],
                            tensor_prev[self.z - 1][self.y - 1][min(self.x - 1, num)].shape[2])
            if num == max(self.z, self.y, self.x):
                x = batch_norm(torch.div((fo_list[2] + m(x)), 1.2))
                y = batch_norm(torch.div((fo_list[1] + m(y)), 1.2))
                z = batch_norm(torch.div((fo_list[0] + m(z)), 1.2))
            else:
                x = batch_norm(torch.div((tensor_prev[self.z - 1][self.y - 1][min(self.x - 1, num)] + m(x)), 1.2))
                y = batch_norm(torch.div((tensor_prev[self.z - 1][min(self.y - 1, num)][self.x - 1] + m(y)), 1.2))
                z = batch_norm(torch.div((tensor_prev[min(self.z - 1, num)][self.y - 1][self.x - 1] + m(z)), 1.2))
            old.append([x, y, z])
        self.losses = losses
        tensor_prev.clear()
        del x, y, z, m, fo_list
        return old
        # for i in range(self.z):
        #     for j in range(self.y):
        #         for k in range(self.x):
        #             """
        #             push xx,yy,zz
        #             """
        #             zz = tensor_prev[j][k]
        #             if j == 0:
        #                 xx = x
        #             else:
        #                 xx = tensor_prev[j - 1][k]
        #             if k == 0:
        #                 yy = y
        #             else:
        #                 yy = tensor_prev[j][k - 1]
        #             tensor_prev[j][k] = self.point_layer_module[str(i) + '_' + str(j) + '_' + str(k)](
        #                 torch.stack([xx, yy, zz], dim=-1))
        #             tensor_prev[j][k] = axonLimit.apply(tensor_prev[j][k])
        #             if k != self.x - 1 and j != self.y - 1 and i != self.z - 1 and self.test == False:
        #                 tensor_prev[j][k]= self.dropout[0][j][k](tensor_prev[j][k])
        # return tensor_prev[-1][-1]

    def initiate_layer(self, data, in_feature, out_feature, tau_m=4., tau_s=1., use_gauss=True):
        """
        three-dim层初始化节点
        """
        self.use_gauss = use_gauss
        self.point_layer = {}
        self.in_shape = data.shape
        self.feature_len = [int(in_feature * (2 ** _)) for _ in range(max(self.x, self.y, self.z) + 1)]
        self.div_len = [2 ** _ for _ in range(max(self.x, self.y, self.z) + 1)]
        old_size = int(math.sqrt(data.shape[1] / in_feature))
        for i in range(max(self.x, self.y, self.z) + 1):
            if old_size % (2 ** i) == 0:
                pass
            else:
                self.feature_len[i] = self.feature_len[i - 1]
                self.div_len[i] = self.div_len[i - 1]
        for i in range(self.z):
            for j in range(self.y):
                for k in range(self.x):
                    """
                    目前虽然有in_pointnum,out_pointnum,in_feature,out_feature,但实际上默认所有的输入输出相等，如果不相等将极大增加模型的设计难度。
                    该版本已经实现了特征维度不相等
                    """
                    num = min(i, j, k)
                    in_div = self.div_len[num]
                    in_feature = self.feature_len[num]
                    out_div = self.div_len[num + 1]
                    out_feature = self.feature_len[num + 1]
                    if (i == self.z - 1) + (j == self.y - 1) + (k == self.x - 1) >= 2:
                        self.point_layer[str(i) + '_' + str(j) + '_' + str(k)] = point_cul_Layer(
                            int(data.shape[1] // in_div),
                            int(data.shape[1] // out_div),
                            path_len=self.z + self.x + self.y,
                            in_feature=in_feature,
                            out_feature=out_feature,
                            tau_m=tau_m,
                            tau_s=tau_s,
                            grad_small=False,
                            weight_require_grad=self.weight_require_grad,
                            weight_rand=self.weight_rand,
                            device=self.device,
                            # bool((i+j+k)%2),False
                            STuning=bool(
                                (i + j + k) % 2),
                            grad_lr=self.grad_lr,
                            use_gauss=self.use_gauss)
                    else:
                        self.point_layer[str(i) + '_' + str(j) + '_' + str(k)] = point_cul_Layer(
                            int(data.shape[1] // in_div),
                            int(data.shape[1] // in_div),
                            path_len=self.z + self.x + self.y,
                            in_feature=in_feature,
                            out_feature=in_feature,
                            tau_m=tau_m,
                            tau_s=tau_s,
                            grad_small=False,
                            weight_require_grad=self.weight_require_grad,
                            weight_rand=self.weight_rand,
                            device=self.device,
                            # bool((i+j+k)%2),False
                            STuning=bool(
                                (i + j + k) % 2),
                            grad_lr=self.grad_lr,
                            use_gauss=self.use_gauss)

        self.point_layer_module = nn.ModuleDict(self.point_layer)
        del self.point_layer
        return self.feature_len[-1], data.shape[1] // self.div_len[-1]

    def subWeightGrad(self, epoch, epochs, sigma=1):
        """
        作为three-dim修改梯度使用
        """
        for i in range(self.z):
            for j in range(self.y):
                for k in range(self.x):
                    self.point_layer_module[str(i) + '_' + str(j) + '_' + str(k)].subWeightGrad(epoch, epochs, sigma,
                                                                                                self.diag_T.get_value(
                                                                                                    self.x - k - 1,
                                                                                                    self.y - j - 1,
                                                                                                    self.z - i - 1),
                                                                                                self.point_layer_module[
                                                                                                    str(
                                                                                                        self.z - 1) + '_' + str(
                                                                                                        self.y - 1) + '_' + str(
                                                                                                        self.x - 1)].norm)


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

    def initiate_layer(self, input, in_feature, out_feature, tau_m=4., tau_s=1., use_gauss=True, batchsize=64,
                       old_in_feature=1, old_out_feature=1):
        if dataoption == 'mnist' or dataoption == 'fashionmnist':
            input = torch.randn(input.shape[0], 1 * 32 * 32).to(input.device)
        elif dataoption == 'cifar10':
            input = torch.randn(input.shape[0], 3 * 32 * 32).to(input.device)
        self.three_dim_layer.initiate_layer(
            torch.rand(batchsize, in_feature * (input.shape[1] // (old_out_feature * 4))),
            in_feature, out_feature, tau_m=tau_m, tau_s=tau_s,
            use_gauss=use_gauss)
        return self.three_dim_layer.feature_len

    def settest(self, test):
        self.three_dim_layer.settest(test)

    def subWeightGrad(self, epoch, epochs, sigma):
        self.three_dim_layer.subWeightGrad(epochs, epoch, sigma)


class merge_layer(nn.Module):
    def __init__(self, device, shape=None, weight_require_grad=True, weight_rand=True, grad_lr=0.01, dropout=0.3,
                 test=False):
        """
        该层是basic层,包含了特征变换层和three-dim路径选择层
        """
        super(merge_layer, self).__init__()
        if shape == None:
            self.shape = [[3, 3, 3], ]
        else:
            self.shape = shape
        self.device = device
        self.InputGenerateNet = InputGenerateNet(self.shape[0], self.device, weight_require_grad, weight_rand, grad_lr,
                                                 dropout,
                                                 test).to(device)
        self.time = 0

    def forward(self, x):
        # x, y = self.initdata(x)
        if hasattr(self, 'input_shape'):
            x = x.view(self.input_shape)
        else:
            if dataoption == 'cifar10':
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
            else:
                raise KeyError()
        a,b,c = self.block_in_x_y_z(x)
        x_lists = self.InputGenerateNet(a,b,c)
        x = self.feature_forward(x_lists)
        h = self.out_classifier(x)
        return h

    def initiate_layer(self, input, in_feature, out_feature, classes, tmp_feature=64, tau_m=4., tau_s=1.,
                       use_gauss=True, batchsize=64):
        """
        配置相应的层
        """
        out_feature_lowbit = (int(out_feature) & int(-out_feature)) + out_feature
        if len(input.shape) != 2:
            input = input.view(input.shape[0], -1)
            self.first = input.shape[0]
            self.second = input.numel() / self.first
            self.input_shape = input.shape
        else:
            self.first = input.shape[0]
            self.second = input.numel() / self.first

        self.block_in_x_y_z = block_in(in_feature, tmp_feature)
        feature_len = self.InputGenerateNet.initiate_layer(input, tmp_feature, tmp_feature, tau_m, tau_s, use_gauss,
                                                           batchsize,
                                                           old_in_feature=in_feature, old_out_feature=out_feature)
        # self.block_out = block_out(tmp_feature, out_feature_lowbit, classes, use_pool='none')
        import copy
        feature_len.append(copy.deepcopy(feature_len[-1]))
        self.feature_forward = Feature_forward(feature_len)
        if dataoption == 'fashionmnist' or dataoption == 'mnist' or dataoption == 'cifar10':
            out_pointnum = max(16 // (feature_len[-1] // tmp_feature), 1)
        else:
            raise KeyError("not import")
        self.out_classifier = block_out(feature_len[-1], feature_len[-1] * 4, classes=classes, size=out_pointnum,
                                        use_pool='none')
        self._initialize()

    def subWeightGrad(self, epoch, epochs, sigma=1):
        """
        three-dim模型的梯度三项式
        """
        for layer in self.modules():
            if isinstance(layer, InputGenerateNet):
                layer.subWeightGrad(epoch, epochs, sigma)

    def settest(self, test=True):
        """
        令模型知道当前处理test
        """
        for layer in self.modules():
            if isinstance(layer, InputGenerateNet) or isinstance(layer,block_in):
                layer.settest(test)

    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) and layer.bias is not None:
                layer: nn.Conv2d
                layer.bias.data.fill_(1.)
                layer.bias.data -= torch.randn_like(layer.bias.data).abs()  # /math.sqrt(layer.bias.data.numel())
            elif isinstance(layer, guassNet):
                layer: guassNet
                layer.gauss_bias.data.fill_(1.)
                layer.gauss_bias.data -= torch.randn_like(
                    layer.gauss_bias.data).abs()  # /math.sqrt(layer.gauss_bias.data.numel())

    def L2_biasoption(self, sigma=1):
        loss = [torch.tensor(0.).float().cuda()]
        normlist = []
        loss_feature = torch.tensor(0.).float().cuda()
        loss_norm = torch.tensor(0.).float().cuda()
        len =torch.tensor(0.).float().cuda()
        len2 = torch.tensor(0.).float().cuda()
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
            elif isinstance(layer, DoorMechanism):
                layer: DoorMechanism
                loss_tau+=(layer.norm_mem_1+layer.norm_mem_2+layer.norm_mem_3)
                len2+=1
        loss_feature /= len
        loss_tau /= len2
        # loss_norm=torch.stack(normlist,dim=-1).std(dim=0)
        # loss_norm = ( torch.stack(loss_norm, dim=-1).min()-torch.stack(loss_norm, dim=-1))
        # loss_norm = (torch.exp(-loss_norm)/torch.exp(-loss_norm).sum(dim=-1)).std(dim=-1)
        loss_bias = torch.stack(loss, dim=-1).mean()
        return (loss_norm + loss_bias + loss_feature) * sigma


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
