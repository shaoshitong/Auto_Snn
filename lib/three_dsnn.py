import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tensorboardX.summary as summary
import random
import numpy as np
from Snn_Auto_master.lib.activation import Activation
from Snn_Auto_master.lib.data_loaders import revertNoramlImgae
from Snn_Auto_master.lib.plt_analyze import vis_img
from Snn_Auto_master.lib.parameters_check import pd_save, parametersgradCheck
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


yaml = yaml_config_get()
dataoption = yaml['data']



class Shortcut(nn.Module):
    """
    该层是为了残差连接而设计，从输入数据剪切一块后进行填充
    目前在特征从初始变换到64大小时会用到
    """

    def __init__(self, in_feature, out_feature, use_same=False):
        in_feature: int
        out_feature: int
        super(Shortcut, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        if use_same == False:
            self.shortcut = lambda x: F.pad(x[:, :, ::2, ::2],
                                            (0, 0, 0, 0, (self.out_feature - x.shape[1]) // 2,
                                             (self.out_feature - x.shape[1]) // 2),
                                            "constant", 0)
        else:
            self.shortcut = lambda x: F.pad(x,
                                            (0, 0, 0, 0, (self.out_feature - x.shape[1]) // 2,
                                             (self.out_feature - x.shape[1]) // 2),
                                            "constant", 0)

    def forward(self, x):
        return self.shortcut(x)

class block_in(nn.Module):
    def __init__(self, in_feature, out_feature=64):
        super(block_in, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, 32, (4, 4), stride=2, padding=3, bias=True, )
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, out_feature, (3, 3), stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_feature)
        self.shortcut2 = Shortcut(32, out_feature, use_same=True)

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)),inplace=True)
        x1 = F.leaky_relu(self.bn2(self.conv2(x1)) + self.shortcut2(x1),inplace=True)
        return x1


class block_out(nn.Module):
    def __init__(self, in_feature, out_feature, classes):
        super(block_out, self).__init__()
        self.conv2 = nn.Conv2d(in_feature, 32, (2, 2), stride=2, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.shortcut2 = Shortcut(in_feature, 32, )
        self.conv1 = nn.Conv2d(32, out_feature, (2, 2), stride=2, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(out_feature)
        self.linear = nn.Linear(out_feature * 4 * 4, classes)

    def forward(self, x):
        x_t = self.bn2(self.conv2(x))
        x = F.leaky_relu(self.shortcut2(x) + x_t,inplace=True)
        x = F.leaky_relu(self.bn1(self.conv1(x)),inplace=True)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
class block_eq(nn.Module):
    def __init__(self,eq_feature):
        super(block_eq,self).__init__()
        self.eq_feature=eq_feature
        self.longConv=nn.Sequential(*[
            nn.Conv2d(eq_feature,eq_feature,(3,3),stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(eq_feature)])
        self.shortConv=nn.Sequential(*[
            Shortcut(eq_feature,eq_feature,use_same=True),
            nn.BatchNorm2d(eq_feature)
        ])
        self.bn_eq=nn.BatchNorm2d(eq_feature)
    def forward(self,x):
        x=self.bn_eq(x)
        x=F.leaky_relu(self.shortConv(x)+self.longConv(x),inplace=True)
        return x
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
        # grad = grad.view(-1, 1,28*28)
        # min, _ = torch.min(grad, dim=-1, keepdim=True)
        # max, _ = torch.max(grad, dim=-1, keepdim=True)
        # return ((grad - min) / (max - min)).view_as(data),max,min
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
            return torch.min(torch.max(v1, torch.Tensor([-1.]).cuda()), torch.Tensor([1.]).cuda())
        elif dataoption == 'cifar10':
            return torch.min(torch.max(v1, torch.Tensor([-1.]).cuda()), torch.Tensor([1.]).cuda())
        elif dataoption == 'fashionmnist':
            return torch.min(torch.max(v1, torch.Tensor([-1.]).cuda()), torch.Tensor([1.]).cuda())
        else:
            raise KeyError()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if dataoption == 'mnist':
            exponent = torch.where((input > -1.1) & (input < 1.1), torch.ones_like(input).cuda(),
                                   torch.zeros_like(input).cuda())
            exponent = torch.where((input > 1.1) | (input < -1.1),
                                   (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
            return exponent * grad_output
        elif dataoption == 'cifar10':
            exponent = torch.where((input > -1.1) & (input < 1.1), torch.ones_like(input).cuda(),
                                   torch.zeros_like(input).cuda())
            exponent = torch.where((input > 1.1) | (input < -1.1),
                                   (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
            return exponent * grad_output
        elif dataoption == 'fashionmnist':
            exponent = torch.where((input > -1.1) & (input < 1.1), torch.ones_like(input).cuda(),
                                   torch.zeros_like(input).cuda())
            exponent = torch.where((input > 1.1) | (input < -1.1),
                                   (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
            return exponent * grad_output
        else:
            raise KeyError('not have this dataset')


class DoorMechanism(nn.Module):
    def __init__(self, in_pointnum, out_pointnum, in_feature, out_feature):
        """
        门机制层，对三条路径传来的数据进行选择
        """
        super(DoorMechanism, self).__init__()
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

    def forward(self, x1, x2, x3, tau_m, tau_s, tau_sm):
        """
        input==>[batchsize,64,x_pointnum//2,y_pointnum//2]
        """
        x1: torch.Tensor
        x2: torch.Tensor
        x3: torch.Tensor
        y1 = x1.view(x1.shape[0], x1.shape[1], -1).mean(dim=-1)
        y2 = x2.view(x2.shape[0], x2.shape[1], -1).mean(dim=-1)
        y3 = x3.view(x3.shape[0], x3.shape[1], -1).mean(dim=-1)
        men_1 = torch.sigmoid(y1 @ self.tau_m_weight2 + tau_m @ self.tau_m_weight1 + self.tau_m_bias).mean(dim=0,
                                                                                                           keepdim=True)
        men_2 = torch.sigmoid(y2 @ self.tau_s_weight2 + tau_s @ self.tau_s_weight1 + self.tau_s_bias).mean(dim=0,
                                                                                                           keepdim=True)
        men_3 = torch.sigmoid(y3 @ self.tau_sm_weight2 + tau_sm @ self.tau_sm_weight1 + self.tau_sm_bias).mean(dim=0,
                                                                                                               keepdim=True)
        result = torch.tanh(
            men_1.unsqueeze(-1).unsqueeze(-1) * x1 + men_2.unsqueeze(-1).unsqueeze(-1) * x2 + men_3.unsqueeze(
                -1).unsqueeze(-1) * x3)
        return result, men_1.clone().detach(), men_2.clone().detach(), men_3.clone().detach()


class point_cul_Layer(nn.Module):
    def __init__(self, in_pointnum, out_pointnum, in_feature, out_feature, tau_m=4., tau_s=1., grad_small=False,
                 weight_require_grad=False,
                 weight_rand=False, device=None, STuning=True, grad_lr=0.1, p=0.2, use_gauss=True):
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
        self.in_feature, self.out_feature = in_pointnum, out_pointnum
        g = False
        self.tensor_tau_m1 = torch.randn((1, 64), dtype=torch.float32, requires_grad=g).to(
            self.device)
        self.tensor_tau_s1 = torch.randn((1, 64), dtype=torch.float32, requires_grad=g).to(
            self.device)
        self.tensor_tau_sm1 = torch.randn((1, 64), dtype=torch.float32, requires_grad=g).to(
            self.device)
        self.activation = [Activation(["elu"], transform=False), Activation(["negative"], transform=False),
                           Activation(["leakyrelu"], transform=False)]
        self.DoorMach = DoorMechanism(in_pointnum, out_pointnum, 64, 64)
        if dataoption == 'mnist':
            if use_gauss == True:
                self.gaussbur = guassNet(64, 64, kernel_size=3, requires_grad=True)
            else:
                self.gaussbur = guassNet(64, 64, kernel_size=3, requires_grad=True, use_gauss=False)
            self.bn1 = nn.BatchNorm2d(64)
        elif dataoption == 'cifar10':
            if use_gauss == True:
                self.gaussbur = guassNet(64, 64, kernel_size=3, requires_grad=True)
            else:
                self.gaussbur = guassNet(64, 64, kernel_size=3, requires_grad=True, use_gauss=False)
            self.bn1 = nn.BatchNorm2d(64)
        elif dataoption == 'fashionmnist':
            if use_gauss == True:
                self.gaussbur = guassNet(64, 64, kernel_size=3, requires_grad=True)
            else:
                self.gaussbur = guassNet(64, 64, kernel_size=3, requires_grad=True, use_gauss=False)
            self.bn1 = nn.BatchNorm2d(64)
        self.STuning = STuning
        self.grad_lr = grad_lr
        self.sigma = 1
        self.x_index = random.randint(0, 2)
        self.y_index = random.randint(0, 2)
        self.z_index = random.randint(0, 2)

    def forward(self, x, weight):
        x1, x2, x3 = x.unbind(dim=-1)
        """
        Try 1
        加入门机制，我们其实可以把tau_s,tau_m,tau_sm想象成门，通过门的筛选来决定他该选择什么样的输出，也就是对路径上所提取特征的筛选。
        该步骤为了减少参数量我先采用使用筛选特征的方式
        """
        x, self.tensor_tau_m1, self.tensor_tau_s1, self.tensor_tau_sm1 = self.DoorMach(x1, x2, x3, self.tensor_tau_m1,
                                                                                       self.tensor_tau_s1,
                                                                                       self.tensor_tau_sm1)
        if self.STuning:
            pass
        if self.weight_rand:
            if dataoption == 'mnist':
                m = self.bn1(self.gaussbur(x))
            elif dataoption == 'cifar10':
                m = self.bn1(self.gaussbur(x))
            elif dataoption == 'fashionmnist':
                m = self.bn1(self.gaussbur(x))
            else:
                raise KeyError('not have this dataset')
            x = m
        return x

    def subWeightGrad(self, epoch, epochs, sigma, diag_num, path_num_x, path_num_y, path_num_z):
        """
        用三项式定理以及链式法则作为数学原理进行梯度修改
        """
        weight = float(sigma) * diag_num * math.pow((self.tensor_tau_m1.mean().clone().detach().cpu()), path_num_x) \
                 * math.pow((self.tensor_tau_s1.mean().clone().detach().cpu()), path_num_y) \
                 * math.pow((self.tensor_tau_sm1.mean().clone().detach().cpu()), path_num_z)
        for name, param in self.named_parameters():
            if param.requires_grad == True and param.grad != None:
                param.grad /= weight
                if epoch <= epochs // 4:
                    param.grad.data = param.grad.data.clamp_(-1, 1)
                else:
                    param.grad.data = param.grad.data.clamp_(-0.5, 0.5)


class three_dim_Layer(nn.Module):
    def __init__(self, shape, device, weight_require_grad=False, weight_rand=False, grad_lr=0.0001, p=0.3, test=False,
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
        self.data_x = []
        self.data_y = []
        self.data_z = []
        self.diag_T = Trinomial_operation(max(max(self.x, self.y), self.z))
        self.grad_lr = grad_lr
        self.dropout = nn.Dropout(p)
        self.test = test

    def settest(self, test=True):
        self.test = test

    def forward(self, x, y, z):
        """
        x,y=>[batchsize,64,x_pointnum//2,y_pointnum//2]
        """
        self.data_y = [y for i in range(self.y)]
        self.data_x = [x for i in range(self.x)]
        self.data_z = z
        tensor_prev = [[torch.zeros_like(x).to(self.device) for i in range(self.x)] for j in range(self.y)]
        for i in range(self.z):
            for j in range(self.y):
                for k in range(self.x):
                    """
                    push xx,yy,zz
                    """
                    if i == 0:
                        zz = self.data_z
                    else:
                        zz = tensor_prev[j][k]
                    if j == 0:
                        xx = self.data_x[k]
                    else:
                        xx = tensor_prev[j - 1][k]
                    if k == 0:
                        yy = self.data_y[j]
                    else:
                        yy = tensor_prev[j][k - 1]
                    tensor_prev[j][k] = self.point_layer_module[str(i) + '_' + str(j) + '_' + str(k)](
                        torch.stack([xx, yy, zz], dim=-1), i + j + k)
                    tensor_prev[j][k] = axonLimit.apply(tensor_prev[j][k])
                    if np.random.rand(1) > np.array([
                        .6666666]) and k != self.x - 1 and j != self.y - 1 and i != self.z - 1 and self.test == False:
                        tensor_prev[j][k] = self.dropout(tensor_prev[j][k])
        return tensor_prev[-1][-1]

    def initiate_layer(self, data, in_feature, out_feature, tau_m=4., tau_s=1., use_gauss=True):
        """
        three-dim层初始化节点
        """
        self.use_gauss = use_gauss
        self.point_layer = {}
        for i in range(self.z):
            for j in range(self.y):
                for k in range(self.x):
                    """
                    目前虽然有in_pointnum,out_pointnum,in_feature,out_feature,但实际上默认所有的输入输出相等，如果不相等将极大增加模型的设计难度。
                    """
                    self.point_layer[str(i) + '_' + str(j) + '_' + str(k)] = point_cul_Layer(data.shape[1] // 2,
                                                                                             data.shape[1] // 2,
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
        self.point_layer_module = nn.ModuleDict(self.point_layer)

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
                                                                                                self.x - k - 1,
                                                                                                self.y - j - 1,
                                                                                                self.z - i - 1)


class merge_layer(nn.Module):
    def __init__(self, device, shape=None, weight_require_grad=True, weight_rand=True, grad_lr=0.01, dropout=0.3,
                 test=False):
        """
        该层是basic层,包含了特征变换层和three-dim路径选择层
        """
        super(merge_layer, self).__init__()
        if shape == None:
            self.shape = [3, 3, 3]
        else:
            self.shape = shape
        self.device = device
        self.three_dim_layer = three_dim_Layer(self.shape, self.device, weight_require_grad, weight_rand, grad_lr,
                                               p=dropout, test=test)
        self.time = 0

    def initdata(self, x):
        """
        初始化输入数据，并求出梯度，返回原始数据，grad数据
        """
        if dataoption == 'cifar10':
            y = DiffInitial(x, [x.shape[0], 3, 32, 32], 3, 3, group=3)[0]
        elif dataoption == 'mnist':
            y = DiffInitial(x, [x.shape[0], 1, 28, 28], 1, 1, group=1)[0]
        elif dataoption == 'fashionmnist':
            y = DiffInitial(x, [x.shape[0], 1, 28, 28], 1, 1, group=1)[0]
        else:
            raise KeyError()
        return x, y

    def forward(self, x):
        # x, y = self.initdata(x)
        if dataoption == 'cifar10':
            x = x.view(x.shape[0], 3, 32, 32)
            # y = y.view(y.shape[0], 3, 32, 32)
        elif dataoption == 'mnist':
            x = x.view(x.shape[0], 1, 28, 28)
            # y = y.view(y.shape[0], 1, 28, 28)
        elif dataoption == 'fashionmnist':
            x = x.view(x.shape[0], 1, 28, 28)
            # y = y.view(y.shape[0], 1, 28, 28)
        else:
            raise KeyError()
        x1 = self.block_inx(x)
        x2 = self.block_iny(x1)
        x3 = self.block_inz(x2)
        x = self.three_dim_layer(x1, x2, x3)
        x = self.block_out(x)
        return x

    def initiate_layer(self, input, in_feature, out_feature, classes, tau_m=4., tau_s=1., use_gauss=True, batchsize=64):
        """
        配置相应的层
        """
        if len(input.shape) != 2:
            input = input.view(input.shape[0], -1)
            self.first = input.shape[0]
            self.second = input.numel() / self.first
        else:
            self.first = input.shape[0]
            self.second = input.numel() / self.first

        self.block_inx = block_in(in_feature,64)
        self.block_iny = block_eq(64)
        self.block_inz = block_eq(64)
        self.three_dim_layer.initiate_layer(torch.rand(batchsize, 64 * (input.shape[1] // (in_feature * 4))),
                                            in_feature, out_feature, tau_m=tau_m, tau_s=tau_s,
                                            use_gauss=use_gauss)
        self.block_out = block_out(64, out_feature, classes)

    def subWeightGrad(self, epoch, epochs, sigma=1):
        """
        three-dim模型的梯度三项式
        """
        self.three_dim_layer.subWeightGrad(epoch, epochs, sigma)

    def settest(self, test=True):
        """
        令模型知道当前处理test
        """
        self.three_dim_layer.settest(test)


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
