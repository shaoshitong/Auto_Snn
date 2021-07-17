import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tensorboardX.summary as summary
import random
import numpy as np
from Snn_Auto_master.lib.activation import Activation
from torch.autograd.variable import Variable
import torch.nn.init as init
import copy
from Snn_Auto_master.lib.plt_analyze import vis_img
from Snn_Auto_master.lib.parameters_check import pd_save, parametersgradCheck
import math
import pandas as pd
from torch.nn import Parameter
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
def yaml_config_get():
    conf = OmegaConf.load('./train.yaml')
    return conf
yaml = yaml_config_get()
print_button = False
weight_button = True
dataoption=yaml['data']

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
        if dataoption=='mnist':
            output = torch.max(torch.tensor(0.0, device=output.device), torch.sign(output))
        elif dataoption=='cifar10':
            output = torch.max(torch.tensor(0.0, device=output.device), torch.sign(output))
        elif dataoption=='fashionmnist':
            output = torch.max(torch.tensor(0.0, device=output.device), torch.sign(output))
        else:
            raise KeyError()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigma = ctx.sigma
        if dataoption=='mnist':
            exponent = -torch.pow((input), 2) / (2.0 * sigma ** 2)
        elif dataoption=='cifar10':
            exponent = -torch.pow((input), 2) / (2.0 * sigma ** 2)
        elif dataoption=='fashionmnist':
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


"""
高斯卷积
"""


class guassNet(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, sigma=1., group=1, requires_grad=True,use_gauss=True):
        super(guassNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.group = group
        self.use_gauss=use_gauss
        self.requires_grad = requires_grad
        if use_gauss==True:
            self.gauss_kernel = self._gauss2D(self.kernel_size, self.sigma, self.group, self.in_channel,
                                          self.out_channel, self.requires_grad)
        else:
            self.gauss_kernel = self._conv2D(self.kernel_size, self.sigma, self.group, self.in_channel,
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
    def _conv2D(self, kernel_size, sigma, group, in_channel, out_channel, requires_grad=False,):
        kernel=torch.normal(.0,sigma,(kernel_size,kernel_size)).float()
        kernel.clamp_(-2*sigma,2*sigma)
        out_channel = out_channel // group
        kernel = ((kernel).unsqueeze(0)).repeat(out_channel, 1, 1)
        kernel = Parameter((kernel.unsqueeze(0)).repeat(in_channel, 1, 1, 1), requires_grad == requires_grad)
        return kernel


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



def DiffInitial(data, shape, in_feature, out_feature, group=1):
    """
    K_{GX} = [-1 0 1 ; -2 0 2 ; -1 0 1], K_{GY} = {-1 -2 -1 ; 0 0 0 ; 1 2 1}
    """
    tmp = data.clone().detach().cuda()
    if in_feature==1:
        tmp = tmp.view(shape).unsqueeze(1).repeat(1, in_feature, 1, 1)
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
    if dataoption=='mnist':
        kernel_gauss = guassNet(1, 1, kernel_size=5, requires_grad=False).cuda()
    elif dataoption=='cifar10':
        kernel_gauss = guassNet(3, 3, kernel_size=5, requires_grad=False,group=3).cuda()
    elif dataoption=='fashionmnist':
        kernel_gauss = guassNet(1, 1, kernel_size=5, requires_grad=False).cuda()
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
    if dataoption=='mnist':
        # grad = grad.view(-1, 1,28*28)
        # min, _ = torch.min(grad, dim=-1, keepdim=True)
        # max, _ = torch.max(grad, dim=-1, keepdim=True)
        # return ((grad - min) / (max - min)).view_as(data),max,min
        grad = grad.view(-1, 1, 28 * 28)
        mean = grad.mean(dim=-1, keepdim=True)
        std = grad.std(dim=-1, keepdim=True)
        return ((grad - mean) / (std + 1e-6)).view_as(data), mean, std
    elif dataoption=='cifar10':
        grad=grad.view(-1,3,32*32)
        mean=grad.mean(dim=-1, keepdim=True)
        std=grad.std(dim=-1,keepdim=True)
        return ((grad-mean)/(std+1e-6)).view_as(data),mean,std
    elif dataoption=='fashionmnist':
        grad = grad.view(-1, 1, 28 * 28)
        mean = grad.mean(dim=-1, keepdim=True)
        std = grad.std(dim=-1, keepdim=True)
        return ((grad - mean) / (std + 1e-6)).view_as(data), mean, std
    else:
        raise KeyError("not have this dataset")


class axonLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v1):
        ctx.save_for_backward(v1)
        # v1 = 1.3 * torch.sigmoid(v1) - 0.2
        # return v1
        if dataoption=='mnist':
            return torch.min(torch.max(v1, torch.Tensor([-1.]).cuda()), torch.Tensor([1.]).cuda())
        elif dataoption=='cifar10':
            return torch.min(torch.max(v1, torch.Tensor([-.7]).cuda()), torch.Tensor([.7]).cuda())
        elif dataoption=='fashionmnist':
            return torch.min(torch.max(v1, torch.Tensor([-1.]).cuda()), torch.Tensor([1.]).cuda())
        else:
            raise KeyError()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if dataoption=='mnist':
            exponent = torch.where((input > -1.1) & (input < 1.1), torch.ones_like(input).cuda(),
                                   torch.zeros_like(input).cuda())
            exponent = torch.where((input > 1.1) | (input < -1.1),
                                   (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
            return exponent * grad_output
        elif dataoption=='cifar10':
            exponent = torch.where((input > -.7) & (input < .7), torch.ones_like(input).cuda(),
                                   torch.zeros_like(input).cuda())
            exponent = torch.where((input > .7) | (input < -.7),
                                   (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
            return exponent * grad_output
        elif dataoption=='fashionmnist':
            exponent = torch.where((input > -1.1) & (input < 1.1), torch.ones_like(input).cuda(),
                                   torch.zeros_like(input).cuda())
            exponent = torch.where((input > 1.1) | (input < -1.1),
                                   (torch.exp(-((input.float()) ** 2) / 2) / 2.506628).cuda(), exponent)
            return exponent * grad_output
        else:
            raise KeyError('not have this dataset')


class point_cul_Layer(nn.Module):
    def __init__(self, in_feature, out_feature, tau_m=4., tau_s=1., grad_small=False, weight_require_grad=False,
                 weight_rand=False, device=None, STuning=True, grad_lr=0.1, p=0.2,use_gauss=True):
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
        self.activation = [Activation(["elu"], transform=False), Activation(["negative"], transform=False),
                           Activation(["leakyrelu"], transform=False)]
        if dataoption=='mnist':
            if use_gauss==True:
                self.gaussbur = guassNet(1, 1, kernel_size=3, requires_grad=True)
            else:
                self.gaussbur= guassNet(1, 1, kernel_size=3, requires_grad=True,use_gauss=False)
            self.bn1 = nn.BatchNorm2d(1)
        elif dataoption=='cifar10':
            if use_gauss==True:
                self.gaussbur= guassNet(1, 1, kernel_size=3, requires_grad=True)
            else:
                self.gaussbur = guassNet(3, 3, kernel_size=3, requires_grad=True,use_gauss=False)
            self.bn1=nn.BatchNorm2d(3)
        elif dataoption == 'fashionmnist':
            if use_gauss == True:
                self.gaussbur = guassNet(1, 1, kernel_size=3, requires_grad=True)
            else:
                self.gaussbur = guassNet(1, 1, kernel_size=3, requires_grad=True, use_gauss=False)
            self.bn1 = nn.BatchNorm2d(1)
        self.STuning = STuning
        self.grad_lr = grad_lr
        self.sigma = 1
        self.x_index = random.randint(0, 2)
        self.y_index = random.randint(0, 2)
        self.z_index = random.randint(0, 2)

    def forward(self, x, weight):
        x1, x2, x3, tensor_reset = x.unbind(dim=-1)

        x1 = self.activation[self.x_index](x1) * self.tensor_tau_m1
        x2 = self.activation[self.y_index](x2) * self.tensor_tau_s1
        x3 = self.activation[self.z_index](x3) * self.tensor_tau_sm1
        x = x1 + x2 + x3
        if self.STuning:
            x = x - threshold.apply(tensor_reset, 1) / float(2 ** weight)
        if self.weight_rand:
            if dataoption=='mnist':
                m = self.bn1(self.gaussbur(x.unsqueeze(1).view(x.shape[0], 1, int(math.sqrt(x.shape[1])), -1))).view(
                    x.shape[0], -1)
            elif dataoption=='cifar10':
                m = self.bn1(self.gaussbur(x.view(x.shape[0], 3, int(math.sqrt(x.shape[1]//3)), -1))).view(
                    x.shape[0], -1)
            elif dataoption=='fashionmnist':
                m = self.bn1(self.gaussbur(x.unsqueeze(1).view(x.shape[0], 1, int(math.sqrt(x.shape[1])), -1))).view(
                    x.shape[0], -1)
            else:
                raise KeyError('not have this dataset')
            x = m
        return x

    def subWeightGrad(self, epoch, epochs, sigma, diag_num, path_num_x, path_num_y, path_num_z):
        # weight = max(.8,float(sigma) * diag_num /( (math.exp(-1. / self.tau_s - 1. / self.tau_m)/3+math.exp(-1. / self.tau_s )/3+math.exp(-1. / self.tau_m )/3)**(path_num )))
        if weight_button == True:
            weight = float(sigma) * diag_num * math.pow((self.tensor_tau_m1.mean().clone().detach().cpu()), path_num_x) \
                     * math.pow((self.tensor_tau_s1.mean().clone().detach().cpu()), path_num_y) \
                     * math.pow((self.tensor_tau_sm1.mean().clone().detach().cpu()), path_num_z)
        else:
            weight = sigma
        if not weight_button:
            pass
        for name, param in self.named_parameters():
            if param.requires_grad == True and param.grad != None:
                param.grad /= weight
                if epoch <= epochs // 4:
                    param.grad.data = param.grad.data.clamp_(-1, 1)
                else:
                    param.grad.data = param.grad.data.clamp_(-0.5, 0.5)


class three_dim_Layer(nn.Module):
    def __init__(self, shape, device, weight_require_grad=False, weight_rand=False, grad_lr=0.0001, p=0.3,test=False,use_gauss=True):
        super(three_dim_Layer, self).__init__()
        """
        x是基于输入
        y是基于正态分布
        原点与其他为0特征
        """
        self.x, self.y, self.z = shape
        self.device = device
        self.use_gauss=use_gauss
        self.weight_require_grad = weight_require_grad
        self.weight_rand = weight_rand
        self.data_x = []
        self.data_y = []
        self.diag_T = Trinomial_operation(max(max(self.x, self.y), self.z))
        self.grad_lr = grad_lr
        self.dropout = nn.Dropout(p)
        self.test=test
    def settest(self,test=True):
        self.test=test
    def forward(self, x):
        tensor_reset = x
        tensor_prev = [[x.clone() for i in range(self.x)] for j in range(self.y)]
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
                        xx = self.data_x[k]
                    else:
                        xx = tensor_prev[j - 1][k]
                    if k == 0:
                        yy = self.data_y[j]
                    else:
                        yy = tensor_prev[j][k - 1]
                    tensor_prev[j][k] = self.point_layer_module[str(i) + '_' + str(j) + '_' + str(k)](
                        torch.stack([xx, yy, zz, tensor_reset], dim=-1), i + j + k)
                    tensor_reset = tensor_reset if torch.ge(torch.randn(1), torch.zeros(1)).item() else tensor_prev[j][
                        k]
                    tensor_prev[j][k] = axonLimit.apply(tensor_prev[j][k])
                    if np.random.rand(1) > np.array([.6666666]) and k != self.x - 1 and j != self.y - 1 and i != self.z - 1 and self.test==False:
                        tensor_prev[j][k] = self.dropout(tensor_prev[j][k])
        return tensor_prev[-1][-1]

    def initiate_data(self, data, epoch=0, epochs=100, require_grad=True, init_option=True,real_img=False):
        self.data_x = []
        self.data_y = []
        if dataoption=='mnist':
            grad_data = lambda x: Parameter(DiffInitial(x, [data.shape[0], 28, 28], 1, 1)[0], requires_grad=require_grad)
        elif dataoption=='cifar10':
            grad_data = lambda x: Parameter(DiffInitial(x, [data.shape[0],3, 32, 32], 3, 3,group=3)[0], requires_grad=require_grad)
        elif dataoption=='fashionmnist':
            grad_data = lambda x: Parameter(DiffInitial(x, [data.shape[0], 28, 28], 1, 1)[0], requires_grad=require_grad)
        else:
            raise KeyError('NOT THIS DATA')
        """plt
        if real_img==True:
            plt.subplot(121)
            a = data.clone().detach().cpu()[0, :].view(3,32*32).permute(1,0)
            a = ((a - torch.min(a, dim=0, keepdim=True)[0]) / (
                        torch.max(a, dim=0, keepdim=True)[0] - torch.min(a, dim=0, keepdim=True)[0])).view(32, 32, 3)*255
            a=a.int()-+
                        plt.imshow(a.numpy(), 'viridis')
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.subplot(122)
            grad,mean,std=DiffInitial(data, [64, 3, 32, 32], 3, 3, group=1)
            a = (grad*std+mean).clone().detach().cpu()[0, :].view(3,32*32).permute(1,0)
            a = ((a - torch.min(a, dim=0, keepdim=True)[0]) / (
                        torch.max(a, dim=0, keepdim=True)[0] - torch.min(a, dim=0, keepdim=True)[0])).view(32, 32, 3)*255
            a=a.int()
            plt.imshow(a.numpy(), 'viridis')
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.savefig("grad.png")
            plt.show()
            exit()
        """

        for i in range(self.y):
            self.data_y.append(
                Parameter(data=torch.rand(data.shape, dtype=torch.float32).to(self.device) / math.sqrt(1.),
                          requires_grad=require_grad) if init_option else grad_data(data)
            )

        self.data_x = [data.to(self.device) for i in range(self.x)]

    def initiate_layer(self, data,tau_m=4.,tau_s=1.,use_gauss=True):
        self.use_gauss=use_gauss
        self.point_layer = {}
        for i in range(self.z):
            for j in range(self.y):
                for k in range(self.x):
                    self.point_layer[str(i) + '_' + str(j) + '_' + str(k)] = point_cul_Layer(data.shape[1],
                                                                                             data.shape[1],
                                                                                             tau_m=tau_m,
                                                                                             tau_s=tau_s,
                                                                                             grad_small=False,
                                                                                             weight_require_grad=self.weight_require_grad,
                                                                                             weight_rand=self.weight_rand,
                                                                                             device=self.device,
                                                                                             # bool((i+j+k)%2),False
                                                                                             STuning=bool(
                                                                                                 (i + j + k) % 2),
                                                                                             grad_lr=self.grad_lr,use_gauss=self.use_gauss)
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
    def __init__(self, device, shape=None, weight_require_grad=True, weight_rand=True, grad_lr=0.01,dropout=0.3,test=False):
        super(merge_layer, self).__init__()
        if shape == None:
            self.shape = [3, 3, 3]
        else:
            self.shape = shape
        self.device = device
        self.three_dim_layer = three_dim_Layer(self.shape, self.device, weight_require_grad, weight_rand, grad_lr,p=dropout,test=test)

    def forward(self, x):
        x = self.three_dim_layer(x)
        return self.linear2(F.leaky_relu(self.linear1(x)))

    def initiate_data(self, input, epoch=0, epochs=100, option=False,real_img=False):
        if len(input.shape) != 2:
            input = input.view(input.shape[0], -1)
            self.first = input.shape[0]
            self.second = input.numel() / self.first
        else:
            self.first = input.shape[0]
            self.second = input.numel() / self.first
        self.three_dim_layer.initiate_data(input.to(self.device), epoch, epochs, True, option,real_img=real_img)

    def initiate_layer(self, input, classes,tau_m=4.,tau_s=1.,use_gauss=True):
        if len(input.shape) != 2:
            input = input.view(input.shape[0], -1)
            self.first = input.shape[0]
            self.second = input.numel() / self.first
        else:
            self.first = input.shape[0]
            self.second = input.numel() / self.first
        self.three_dim_layer.initiate_layer(input.to(self.device),tau_m=tau_m,tau_s=tau_s,use_gauss=use_gauss)
        self.tmp_channel=int(math.sqrt(self.second*classes))
        self.linear1 = nn.Linear(int(self.second), self.tmp_channel, bias=True)
        self.linear2 = nn.Linear(self.tmp_channel, classes, bias=True)
        stdv = 6. / math.sqrt(self.linear2.weight.data.size(1) + self.linear2.weight.data.size(0))
        self.linear2.weight.data.uniform_(-stdv, stdv)
        self.linear2.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.linear1.weight.data.size(1) + self.linear1.weight.data.size(0))
        self.linear1.weight.data.uniform_(-stdv, stdv)
        self.linear1.bias.data.fill_(0.)

    def subWeightGrad(self, epoch, epochs, sigma=1):
        self.three_dim_layer.subWeightGrad(epoch, epochs, sigma)
    def settest(self,test=True):
        self.three_dim_layer.settest(test)