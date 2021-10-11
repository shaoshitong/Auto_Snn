import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import torchvision.models
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
from lib.GRU import multi_GRU, multi_block_eq, Cat, DenseBlock, cat_result_get,return_tensor_add,numeric_get
from lib.DenseNet import DenseBlock as DenseDeepBlock
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
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1.)
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                # layer.weight.data.zero_()
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
    def __init__(self, in_feature, out_feature, hidden_size, in_size, out_size, path_len, cat_x, cat_y,b, STuning=True,
                 grad_lr=0.1, dropout=0.3,
                 use_gauss=True, mult_k=2):
        """
        输入的张量维度为（batch_size,64,x//2,y//2）
        该层通过门机制后进行卷积与归一化
        """
        super(point_cul_Layer, self).__init__()
        self.cat_feature = (path_len - 1) * (out_feature) + in_feature
        if cat_x==cat_y:
            fusion=1
        elif cat_x>cat_y:
            fusion=0
        else:
            fusion=2
        self.DoorMach = DenseBlock(self.cat_feature, out_feature, hidden_size, cat_x, cat_y,
                                   dropout,fusion,in_size)
        self.STuning = STuning
        self.b=b
        self.grad_lr = grad_lr
        self.sigma = 1
        self.norm = None

    def forward(self, x):
        tensor_prev, (i, j) = x
        x = self.DoorMach(cat_result_get(tensor_prev, i, j ,self.b))
        return x


class two_dim_layer(nn.Module):
    def __init__(self, in_feature, out_feature, hidden_size, in_size, out_size, x, y,b, mult_k=2, p=0.2):
        super(two_dim_layer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_size = hidden_size
        self.in_pointnum = in_size
        self.out_pointnum = out_size
        self.x = x
        self.y = y
        self.b=b
        assert self.x>=self.b
        assert self.x>=1 and self.y>=1
        """
        +3 +7 +11 +15
        +2 +5 +8 +11
        +1 +3 +5 +7
        +1 +1 +2 +3
        """
        self.point_cul_layer = {}
        self.test = False
        self.tensor_check=numeric_get(x,y,b)
        if self.x>0 and self.y>0:
            self.x_eq = nn.ModuleList(
                [DenseBlock(out_feature * (_) + in_feature, out_feature, hidden_size, 0, 0, p,1,in_size) for _ in
                 range(min(self.x - 1,self.b-1))])
            self.y_eq = nn.ModuleList(
                [DenseBlock(out_feature * (_) + in_feature, out_feature, hidden_size, 0, 0, p,1,in_size) for _ in
                 range(min(self.y - 1,self.b-1))])
            for i in range(1,self.x):
                for j in range(1,self.y):
                    if abs(i-j)<self.b:
                        if not (i==j):
                            self.point_cul_layer[str(i) + "_" + str(j)] = point_cul_Layer(
                                in_feature,
                                out_feature,
                                hidden_size,
                                in_size,
                                out_size,
                                (i+1)*(j+1)-1-self.tensor_check[i][j],
                                i ,
                                j ,
                                b,
                                dropout=p,
                                mult_k=mult_k)
                        else:
                            self.point_cul_layer[str(i) + "_" + str(j)] = point_cul_Layer(
                                in_feature,
                                out_feature,
                                hidden_size,
                                in_size,
                                out_size,
                                i*(2*self.x-i)-(self.tensor_check[self.x-1][j]+self.tensor_check[i][self.y-1]-self.tensor_check[i][j]),
                                i ,
                                j ,
                                b,
                                dropout=p,
                                mult_k=mult_k)
            self.point_layer_module = nn.ModuleDict(self.point_cul_layer)
            self.np_last = (self.x) * (self.y) - 1 - self.tensor_check[self.x-1][self.y-1]
        else:
            self.np_last = 1
        self.dimixloss=DimixLoss_neg()
    def forward(self, z):
        if self.x==0 and self.y==0:
            return z
        tensor_prev = [[z for i in range(self.x)] for j in range(self.y)]
        for i in range(min(self.y - 1,self.b-1)):
            tensor_prev[0][i + 1] = self.x_eq[i](cat_result_get(tensor_prev, 0, i + 1 ,self.b))
        for i in range(min(self.x - 1,self.b-1)):
            tensor_prev[i + 1][0] = self.y_eq[i](cat_result_get(tensor_prev, i + 1, 0, self.b))
        self.losses = self.dimixloss(tensor_prev[min(self.x - 1,self.b-1)][0], tensor_prev[0][min(self.y - 1,self.b-1)])
        for l in range(1,min(self.x,self.y)):
            tensor_prev[l][l]= self.point_layer_module[str(l) + '_' + str(l)]((
                        tensor_prev, (l,l)))
            for i in range(l+1,self.y):
                if abs(l-i)<self.b:
                    tensor_prev[l][i] =self.point_layer_module[str(l) + '_' + str(i)]((tensor_prev,(l,i)))
            for i in range(l+1,self.x):
                if abs(i-l)<self.b:
                    tensor_prev[i][l] =self.point_layer_module[str(i) + '_' + str(l)]((tensor_prev,(i,l)))
            self.losses =self.losses+self.dimixloss(tensor_prev[min(self.b+l-1,self.x - 1)][l], tensor_prev[l][min(self.b+l-1,self.y - 1)])
        result = []
        for i in range(self.x):
            for j in range(self.y):
                if abs(i-j)<self.b:
                    result.append(tensor_prev[i][j])
        result = torch.cat(result, dim=1)
        del tensor_prev
        return result

    def settest(self, test=True):
        self.test = test

class turn_layer(nn.Module):
    def __init__(self, in_feature, out_feature, bn_size, num_layer, decay_rate=2, stride=1, dropout=0.1):
        super(turn_layer, self).__init__()
        if num_layer != 0:
            self.dense_deep_block = DenseDeepBlock([in_feature] + [out_feature] * num_layer, bn_size, dropout,
                                                   num_layer)
            in_feature = in_feature + out_feature * num_layer
        if num_layer!=0:
            self.origin_out_feature = origin_out_feature = int(in_feature // decay_rate)
            self.num_layer = num_layer
            self.downsample = nn.Sequential(*[])
            self.downsample.add_module('norm', nn.BatchNorm2d(in_feature))
            self.downsample.add_module("relu", nn.ReLU(inplace=True))
            self.downsample.add_module("conv",nn.Conv2d(in_feature, origin_out_feature, (1, 1), (1, 1), (0, 0), bias=False))
            self.xsample = nn.Sequential(*[])
            self.xsample.add_module('pool', nn.AvgPool2d(kernel_size=(stride, stride), stride=(stride, stride)))
        else:
            self.origin_out_feature = origin_out_feature = int(in_feature)
            self.num_layer = num_layer
            self.downsample = nn.Sequential(*[])
            self.downsample.add_module('norm', nn.BatchNorm2d(in_feature))
            self.downsample.add_module("relu", nn.ReLU(inplace=True))
            self.downsample.add_module("pool", nn.MaxPool2d((stride,stride),(stride,stride)))
        self.feature_different = DimixLoss_neg()
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

    def forward(self, x):
        if self.num_layer != 0:
            x = self.dense_deep_block(x)
            x = self.downsample(x)
            x = self.xsample(x)
        else:
            x=self.downsample(x)
        return x


class three_dim_Layer(nn.Module):
    def __init__(self, shape, device, p=0.1):
        super(three_dim_Layer, self).__init__()
        """
        该层便是three-dim层
        x维度代表原始数据通过卷积变换至[batchsize,64,x//2,y//2]
        y维度代表原始数据先获得grad后经过卷积变换到[batchsize,64,x//2,y//2]
        z维度目前用0向量填充，未来可以改进
        """
        self.a, self.b, self.c = shape[0], shape[1], shape[2]
        self.shape = shape
        self.device = device
        self.dropout = p
        self.diag_T = Trinomial_operation(max(self.a, self.b, self.c))
        self.a_join, self.b_join, self.c_join = LastJoiner(2), LastJoiner(2), LastJoiner(2)

    def forward(self, m):
        """
        x,y=>[batchsize,64,x_pointnum//2,y_pointnum//2]
        """
        self.losses = 0.
        for i in range(self.len):
            z = self.turn_layer_module[str(i)](m)
            m = self.point_layer_module[str(i)](z)
        return m

    def initiate_layer(self, data, feature_list, size_list, hidden_size_list, path_nums_list, nums_layer, decay_rate,
                       mult_k,breadth_threshold):
        """
        three-dim层初始化节点
        """
        self.point_layer = {}
        self.turn_layer = {}
        self.in_shape = data.shape
        assert len(feature_list) == len(size_list) and len(hidden_size_list) == len(path_nums_list) and len(
            path_nums_list) == len(nums_layer) and len(breadth_threshold)==len(nums_layer)
        for i in range(len(hidden_size_list)):
            f1, f2 = feature_list[i], feature_list[i + 1]
            s1, s2 = size_list[i], size_list[i + 1]
            h1 = hidden_size_list[i]
            p1 = path_nums_list[i]
            n1 = nums_layer[i]
            b1 = breadth_threshold[i]
            if i == 0:
                self.turn_layer[str(i)] = turn_layer(f1, f2, h1, n1, decay_rate, int(s1 // s2), self.dropout)
            else:
                self.turn_layer[str(i)] = turn_layer(h, f2, h1, n1, decay_rate, int(s1 // s2), self.dropout)
            m = self.turn_layer[str(i)].origin_out_feature
            self.point_layer[str(i)] = two_dim_layer(m, f2, h1, s2, s2, p1, p1,b1, mult_k, self.dropout)
            h = self.point_layer[str(i)].np_last * f2 + m
        self.turn_layer_module = nn.ModuleDict(self.turn_layer)
        self.point_layer_module = nn.ModuleDict(self.point_layer)
        self.len = len(hidden_size_list)
        del self.point_layer, self.turn_layer
        return h


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
        self.InputGenerateNet = three_dim_Layer(self.shape, self.device, dropout).to(device)

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

    def initiate_layer(self, data, num_classes, feature_list, size_list, hidden_size_list, path_nums_list,
                       nums_layer_list,breadth_threshold, mult_k=2,drop_rate=2):
        """
        配置相应的层
        """
        b, c, h, w = data.shape
        input_shape = (b, c, h, w)
        self.inf = nn.Conv2d(c, feature_list[0], (3, 3), (1,1), (1, 1), bias=False)
        h = self.InputGenerateNet.initiate_layer(data, feature_list, size_list, hidden_size_list, path_nums_list,
                                                 nums_layer_list, drop_rate,mult_k,breadth_threshold)
        self.out_classifier = block_out(h, num_classes, size_list[-1])
        self._initialize()
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    @staticmethod
    def _list_build():
        return [0.1, 0.1]

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
                loss_bias.append(torch.norm(torch.abs(layer.weight.data), p=2) / layer.weight.data.numel())
            elif isinstance(layer, two_dim_layer):
                layer: two_dim_layer
                loss_feature += layer.losses
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
