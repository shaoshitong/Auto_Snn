import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from lib.data_loaders import revertNoramlImgae
from lib.plt_analyze import vis_img
from lib.parameters_check import pd_save, parametersgradCheck
from lib.SNnorm import SNConv2d, SNLinear
from lib.fractallayer import LastJoiner
from lib.DenseNet import DenseNet
from lib.cocoscontextloss import ContextualLoss_forward
from lib.Wideresnet import Downsampleunit
from lib.featurefocusing_v2 import Feature_forward
from lib.dimixloss import DimixLoss, DimixLoss_neg
from lib.PointConv import PointConv
from lib.GRU import multi_GRU,multi_block_eq,Cat
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


filename = "./train_c10.yaml"
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

class block_out(nn.Module):
    def __init__(self, feature, classes, size, use_pool='none'):
        super(block_out, self).__init__()
        self.classifiar = nn.Sequential(nn.Flatten(), nn.Linear(feature, classes))
        self.transition_layer = nn.Sequential(*[
            nn.BatchNorm2d(feature),
            nn.ReLU(inplace=True),
            Lambda(F.avg_pool2d)])
        self.training = False
        self.use_pool = use_pool
        self.size = size
        self.classes = classes
        self._initialize()
    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data,mode="fan_in",nonlinearity="relu")
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer,nn.BatchNorm2d):
                layer.weight.data.fill_(1.)
                layer.bias.data.zero_()

    def forward(self, x):
        x = self.transition_layer(x)
        return self.classifiar(x)

# class block_eq(nn.Module):
#     def __init__(self, eq_feature,tmp_feature,dropout):
#         super(block_eq, self).__init__()
#         self.eq_feature = eq_feature
#         self.tmp_feature=tmp_feature
#         self.convz1 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (1, 5), padding=(0, 2))
#         self.convr1 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (1, 5), padding=(0, 2))
#         self.convq1 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (1, 5), padding=(0, 2))
#         self.convz2 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (5, 1), padding=(2, 0))
#         self.convr2 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (5, 1), padding=(2, 0))
#         self.convq2 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (5, 1), padding=(2, 0))
#         self.convo = nn.Sequential(*[
#             nn.Conv2d(eq_feature, eq_feature, (3, 3), (1, 1), (1, 1), bias=False),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout),
#             nn.BatchNorm2d(eq_feature),
#         ])
#         self.xgru=nn.Sequential(nn.Conv2d(eq_feature,tmp_feature,(1,1),(1,1),padding=0,bias=False))
#         self._initialize()
#     def _initialize(self):
#         for layer in self.modules():
#             if isinstance(layer, nn.Conv2d):
#                 nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="sigmoid")
#                 if layer.bias is not None:
#                     nn.init.zeros_(layer.bias.data)
#             if isinstance(layer, nn.BatchNorm2d):
#                 nn.init.ones_(layer.weight.data)
#                 nn.init.zeros_(layer.bias.data)
#         for layer in self.convo.modules():
#             if isinstance(layer, nn.Conv2d):
#                 nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
#                 if layer.bias is not None:
#                     nn.init.zeros_(layer.bias.data)
#
#         nn.init.kaiming_normal_(self.convq1.weight.data, mode="fan_in", nonlinearity="tanh")
#         nn.init.kaiming_normal_(self.convq2.weight.data, mode="fan_in", nonlinearity="tanh")
#
#     def forward(self, m):
#         h, x = m
#         hx = torch.cat([h, x], dim=1)
#         z = torch.sigmoid(self.convz1(hx))
#         r = torch.sigmoid(self.convr1(hx))
#         q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
#         h = F.relu((1 + z) * h + (1 - z) * q)
#         hx  = torch.cat([h, x], dim=1)
#         z = torch.sigmoid(self.convz2(hx))
#         r = torch.sigmoid(self.convr2(hx))
#         q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
#         h = F.relu((1 + z) * h + (1 - z) * q)
#         x =  self.xgru(h)
#         h =  self.convo(h)
#         del  m, z, r, q
#         return (h, x)
#
#
# class multi_block_eq(nn.Module):
#     def __init__(self, in_feature, out_feature,hidden_size, multi_k=1, stride=1,dropout=0.1):
#         super(multi_block_eq, self).__init__()
#         if in_feature != out_feature or stride != 1:
#             self.sample = nn.Sequential(
#                 nn.Conv2d(in_feature, out_feature, (stride, stride), (stride, stride), bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(out_feature))
#             self.psample = nn.Sequential(
#                 nn.Conv2d(in_feature, hidden_size, (stride, stride), (stride, stride), bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(hidden_size))
#             self.qsample= nn.Sequential(
#                 nn.Conv2d(hidden_size, out_feature, (1,1),(1,1), bias=False))
#             for layer in self.sample.modules():
#                 if isinstance(layer, nn.Conv2d):
#                     nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
#                 if isinstance(layer, nn.BatchNorm2d):
#                     nn.init.ones_(layer.weight.data)
#                     nn.init.zeros_(layer.bias.data)
#             for layer in self.psample.modules():
#                 if isinstance(layer, nn.Conv2d):
#                     nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
#                 if isinstance(layer, nn.BatchNorm2d):
#                     nn.init.ones_(layer.weight.data)
#                     nn.init.zeros_(layer.bias.data)
#         self.xgru=nn.Sequential(nn.Conv2d(in_feature,in_feature,(1,1),(1,1),padding=0,bias=False))
#         self.model = nn.Sequential(*[
#             block_eq(out_feature,hidden_size,dropout) for _ in range(multi_k)
#         ])
#
#     def forward(self, x):
#         x,h=x
#         if h == None:
#             h = self.xgru(x)
#         if hasattr(self, "sample"):
#             x = self.sample(x)
#             h = self.psample(h)
#         x,h=self.model((x, h))
#         h=self.qsample(h)
#         return (x,h)
#

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

class point_cul_Layer(nn.Module):
    def __init__(self, in_feature,out_feature,hidden_size,in_size,out_size, STuning=True, grad_lr=0.1, dropout=0.3, use_gauss=True, mult_k=2):
        """
        输入的张量维度为（batch_size,64,x//2,y//2）
        该层通过门机制后进行卷积与归一化
        """
        super(point_cul_Layer, self).__init__()
        self.DoorMach = multi_GRU(in_feature,hidden_size,dropout)
        self.cat=Cat(out_feature,in_feature)
        self.gaussbur=multi_block_eq(in_feature,out_feature,hidden_size,mult_k,stride=1,dropout=dropout)
        self.STuning = STuning
        self.grad_lr = grad_lr
        self.sigma = 1
        self.norm = None
    def forward(self, x):
        x1, x2, x3 = x
        x = self.DoorMach((x1+x3,x2+x3))
        x = self.gaussbur(x)
        return x

class two_dim_layer(nn.Module):
    def __init__(self, in_feature, out_feature,hidden_size,in_size,out_size, x, y, mult_k=2, p=0.2):
        super(two_dim_layer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_size=hidden_size
        self.in_pointnum = in_size
        self.out_pointnum = out_size
        self.x = x
        self.y = y
        self.point_cul_layer = {}
        self.test = False
        self.advance_layer=multi_block_eq(out_feature, out_feature, hidden_size, multi_k=mult_k)
        for i in range(self.x):
            for j in range(self.y):
                if not (i == self.x - 1 and j == self.y - 1):
                    self.point_cul_layer[str(i) + "_" + str(j)] = point_cul_Layer(
                        in_feature,
                        in_feature,
                        hidden_size,
                        in_size,
                        in_size,
                        dropout=p,
                        mult_k=mult_k)
                else:
                    self.point_cul_layer[str(i) + "_" + str(j)] = point_cul_Layer(
                        in_feature,
                        out_feature,
                        hidden_size,
                        in_size,
                        out_size,
                        dropout=p,
                        mult_k=mult_k)
        self.point_layer_module = nn.ModuleDict(self.point_cul_layer)
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
                    (xx,yy,zz))
        result=tensor_prev[-1][-1].clone()
        del tensor_prev
        return result

    def settest(self, test=True):
        self.test = test
class turn_layer(nn.Module):
    def __init__(self,in_feature,out_feature,stride=1,dropout=0.1):
        super(turn_layer, self).__init__()
        self.downsample=Downsampleunit(in_feature,out_feature,stride,dropout)
        self.turn=nn.ModuleList([
            nn.Sequential(
            nn.BatchNorm2d(out_feature),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feature,out_feature,(3,5),(1,1),(1,2),bias=False),
                                               ),
            nn.Sequential(
            nn.BatchNorm2d(out_feature),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feature, out_feature, (5, 3), (1, 1), (2, 1), bias=False),
                                                ),])
        self.feature_different=DimixLoss_neg()
        self._initialize()

    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
            elif isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)
    def forward(self,x):
        x=self.downsample(x)
        a,b=self.turn[0](x),self.turn[1](x)
        l=self.feature_different(a,b)
        return (a,b,x),l

class three_dim_Layer(nn.Module):
    def __init__(self, shape, device, p=0.1):
        super(three_dim_Layer, self).__init__()
        """
        该层便是three-dim层
        x维度代表原始数据通过卷积变换至[batchsize,64,x//2,y//2]
        y维度代表原始数据先获得grad后经过卷积变换到[batchsize,64,x//2,y//2]
        z维度目前用0向量填充，未来可以改进
        """
        self.a,self.b,self.c=shape[0],shape[1],shape[2]
        self.shape = shape
        self.device = device
        self.dropout = p
        self.diag_T = Trinomial_operation(max(self.a, self.b, self.c))
        self.a_join, self.b_join, self.c_join = LastJoiner(2), LastJoiner(2), LastJoiner(2)
    def forward(self, m):
        """
        x,y=>[batchsize,64,x_pointnum//2,y_pointnum//2]
        """
        (x,y,z),l1=self.turn_layer_module["0"](m)
        m = self.point_layer_module["0"](x,y,z)
        (x,y,z),l2=self.turn_layer_module["1"](m)
        m = self.point_layer_module["1"](x,y,z)
        (x,y,z),l3=self.turn_layer_module["2"](m)
        m = self.point_layer_module["2"](x,y,z)
        self.losses=(l1+l2+l3)
        return m
    def initiate_layer(self, data, feature_list,size_list,hidden_size_list,path_nums_list,mult_k=2):
        """
        three-dim层初始化节点
        """
        self.point_layer = {}
        self.turn_layer = {}
        self.in_shape = data.shape
        print(feature_list,size_list,hidden_size_list,path_nums_list)
        assert len(feature_list)==4 and len(size_list) == 4 and len(hidden_size_list) ==3 and len(path_nums_list)==3
        f1,f2,f3,f4=feature_list[0],feature_list[1],feature_list[2],feature_list[3]
        s1,s2,s3,s4=size_list[0],size_list[1],size_list[2],size_list[3]
        h1,h2,h3=hidden_size_list[0],hidden_size_list[1],hidden_size_list[2]
        p1,p2,p3=path_nums_list[0],path_nums_list[1],path_nums_list[2]
        self.point_layer["0"]=two_dim_layer(f2,f2,h1,s2,s2,p1,p1,mult_k,self.dropout)
        self.point_layer["1"]=two_dim_layer(f3,f3,h2,s3,s3,p2,p2,mult_k,self.dropout)
        self.point_layer["2"]=two_dim_layer(f4,f4,h3,s4,s4,p3,p3,mult_k,self.dropout)
        self.turn_layer["0"]=turn_layer(f1,f2,1,self.dropout)
        self.turn_layer["1"]=turn_layer(f2,f3,2,self.dropout)
        self.turn_layer["2"]=turn_layer(f3,f4,2,self.dropout)
        self.turn_layer_module = nn.ModuleDict(self.turn_layer)
        self.point_layer_module = nn.ModuleDict(self.point_layer)
        del self.point_layer, self.turn_layer



class merge_layer(nn.Module):
    def __init__(self, device, shape=None, dropout=0.3):
        """
        该层是basic层,包含了特征变换层和three-dim路径选择层
        """
        super(merge_layer, self).__init__()
        if shape == None:
            self.shape = [2, 2, 2]
        else:
            self.shape = shape
        self.device = device
        self.InputGenerateNet = three_dim_Layer(self.shape,self.device,dropout).to(device)
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
        x = self.inf(x)
        x = self.InputGenerateNet(x)
        x = self.out_classifier(x)
        return x

    def initiate_layer(self, data, num_classes,feature_list,size_list,hidden_size_list,path_nums_list,mult_k=2):
        """
        配置相应的层
        """
        b, c, h, w = data.shape
        input_shape = (b,c,h,w)
        self.inf = nn.Conv2d(c, feature_list[0], (3, 3), (1, 1), (1, 1), bias=False)
        self.InputGenerateNet.initiate_layer(data,feature_list,size_list,hidden_size_list,path_nums_list,mult_k)
        self.out_classifier = block_out(feature_list[-1],num_classes,size_list[-1])

    @staticmethod
    def _list_build():
        return [ 0.1, 0.1]

    @staticmethod
    def _list_print(list):
        for i in list:
            print(i.squeeze().item(), end=",")
        print("")

    def L2_biasoption(self, loss_list, sigma=None):
        if sigma == None:
            sigma = self._list_build()
        loss_bias = [torch.tensor(0.).float().cuda()]
        loss_feature = torch.tensor([0.]).float().cuda()
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) and layer.bias is not None:
                layer: nn.Conv2d
                loss_bias.append(torch.norm(torch.abs(layer.bias.data) - 1., p=2) / layer.bias.data.numel())
            elif isinstance(layer, three_dim_Layer):
                layer: three_dim_Layer
                loss_feature+=layer.losses
        loss_feature = (loss_feature.squeeze(-1)) * sigma[0]
        loss_bias = torch.stack(loss_bias, dim=-1).mean() * sigma[1]
        loss_list = loss_list + [loss_bias, loss_feature]
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
