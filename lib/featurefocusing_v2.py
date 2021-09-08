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
from Snn_Auto_master.lib.cocoscontextloss import ContextualLoss_forward
import math
import pandas as pd
from torch.nn.parameter import Parameter
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn.init as init
import math
from einops import rearrange


class linear(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(linear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_feature, out_feature))
        self.bias = Parameter(torch.Tensor(1, 1, out_feature))
        self.feature = 6. / math.sqrt(in_feature * out_feature)
        self._initialize()

    def forward(self, x):
        x = x @ self.weight + self.bias
        return x

    def _initialize(self):
        self.weight.data.uniform_(-self.feature, self.feature)
        self.bias.data.fill_(0.)


class MLP(nn.Sequential):
    def __init__(self, in_feature, out_feature, p=0.2000000000001):
        tmp_feature = int(math.sqrt(in_feature * out_feature) // 4)
        super(MLP, self).__init__()
        self.add_module("linear_1", linear(in_feature, tmp_feature))
        self.add_module("gelu", nn.GELU())
        self.add_module("linear_2", linear(tmp_feature, out_feature))

    def forward(self, x):
        x = super(MLP, self).forward(x)
        return x


class mixer_Layer(nn.Module):
    def __init__(self, feature, size, s):
        super(mixer_Layer, self).__init__()
        self.T_MLP = MLP(feature, feature)
        self.size = int(size // s)
        self.F_MLP = MLP(self.size ** 2, self.size ** 2)
        self.s = s
        self.Layernorm = nn.ModuleList([nn.LayerNorm([self.size ** 2, feature]),
                                        nn.LayerNorm([self.size ** 2, feature]), ])
        self.batchnorm = nn.BatchNorm1d(feature)

    def forward(self, x):  # b h*w c
        # x=rearrange(x,"b c h w -> b (h w) c")
        y = self.Layernorm[0](x)
        y = y.permute(0, 2, 1)
        y = self.F_MLP(y)
        x = y.permute(0, 2, 1) + x
        y = self.Layernorm[1](x)
        y = self.T_MLP(y) + x
        y = self.batchnorm(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y


class multi_mixer_layer(nn.Module):
    def __init__(self, feature, size, s, multi_num=1, push_num=10):
        super(multi_mixer_layer, self).__init__()
        self.pre_conv = nn.ModuleList([nn.Conv2d(feature, feature, (s, s), (s, s)) for _ in range(multi_num)])
        self.mixer_layer = nn.ModuleList(
            [(nn.Sequential(*[mixer_Layer(feature, size, s) for j in range(push_num)])) for i in range(multi_num)])
        self.multi_num = multi_num
        self.layernorm = nn.LayerNorm([feature])
        """
        self.weight=Parameter(torch.Tensor(multi_num,multi_num))
        self.weight.data.fill_(1./multi_num)
        self.weight.data+=torch.clamp(torch.randn(multi_num,multi_num),-.3/multi_num,.3/multi_num)
        """

    def forward(self, x):
        result = []
        for i in range(self.multi_num):
            x_1 = self.pre_conv[i](x)
            x_1 = rearrange(x_1, "b c h w -> b (h w) c")
            x_1 = self.mixer_layer[i](x_1)
            # x_1 = torch.mean(x_1, dim=1, keepdim=True)
            result.append(x_1)
        result = self.layernorm(torch.cat(result, dim=1)).permute(0, 2, 1)
        # result=result@self.weight  #b,c,multi_num
        return result


class multi_attention(nn.Module):
    def __init__(self, feature, multi_num, p=0.2,temperate=1.):
        super(multi_attention, self).__init__()
        self.linear = nn.ModuleList([linear(multi_num, multi_num) for _ in range(3)])
        self.dropout = nn.Dropout(p=p)
        self.temperate=temperate
        self.layernorm=nn.LayerNorm(feature)

    def forward(self, k, q, v):
        p=k+q+v
        k = self.linear[0](k-v)
        q = self.linear[1](q-v)
        v_1 = self.linear[2](p)
        c = torch.matmul(k/self.temperate, q.permute(0, 2, 1))  # b,c,c
        c = self.dropout(F.softmax(c, dim=-1))
        c = torch.matmul(c, v_1).permute(0,2,1) #b,c,l
        return self.layernorm(c + v).permute(0,2,1)


class Feature_forward(nn.Module):
    def __init__(self, feature_list, size_list, multi_num=4, push_num=5, s=4, p=0.2):
        """
        include in_feature,tmp_feature_1,.....,tmp_feature_n,out_feature
        """
        super(Feature_forward, self).__init__()
        self.feature_list = feature_list
        self.size_list = size_list
        assert (
                    multi_num == 1 or multi_num == 2 or multi_num == 4 or multi_num == 8 or multi_num == 16 or multi_num == 64)
        self.multi_mix_layer = nn.ModuleList(
            [multi_mixer_layer(feature_list[-1], size_list[-1], s, multi_num=multi_num, push_num=push_num)])
        self.multi_attention = multi_attention(feature_list[-1], multi_num * s * s, p)
        self.batchnorm = nn.ModuleList([nn.BatchNorm2d(feature_list[-1])])
        """
        self.out_conv=nn.Sequential(nn.Conv2d(3*self.feature_list[-1],self.feature_list[-1],(1,1),(1,1)),
                                    nn.BatchNorm2d(self.feature_list[-1]))
        """

    def forward(self, x_lists):
        k, q, v = x_lists
        b, c, h, w = q.shape
        k = k.view(b, c, -1)
        q = q.view(b, c, -1)
        v = v.view(b, c, -1)
        result = self.multi_attention(q, k, v).view(b, c, h, w)
        result = self.batchnorm[0](result)
        result = self.multi_mix_layer[0](result).view(b, c, h, w)
        return result
