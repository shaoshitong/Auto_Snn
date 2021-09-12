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
            result.append(x_1)
        result = self.layernorm(torch.cat(result, dim=1)).permute(0, 2, 1)
        return result
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        attn = torch.matmul(attn, v)

        return attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, feature,n_head=8, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(feature, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(feature, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(feature, n_head * d_v, bias=False)
        self.fc =   nn.Linear(n_head * d_v, feature, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = rearrange(q,"a b c d -> a c ( b d )")
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q




class Resnet_forward(nn.Module):
    def __init__(self, in_feature, tmp_feature, size, p=0.2):
        super(Resnet_forward, self).__init__()
        self.lnet = nn.Sequential()
        self.in_feature = in_feature
        self.tmp_feature = tmp_feature
        self.lnet.add_module("batchnorm_0", nn.BatchNorm2d(in_feature))
        self.lnet.add_module("gelu", nn.GELU())
        self.lnet.add_module("conv1", nn.Conv2d(in_feature, tmp_feature, (3, 3), (1, 1), (1, 1)))
        self.lnet.add_module("batchnorm_1", nn.BatchNorm2d(tmp_feature))
        self.lnet.add_module("relu", nn.ReLU(inplace=True))
        self.lnet.add_module("dropout", nn.Dropout(p=p))
        self.lnet.add_module("conv2", nn.Conv2d(tmp_feature, in_feature, (3, 3), (1, 1), (1, 1)))

        self.attention = nn.Sequential(*[
            nn.MaxPool2d(int(size // 2), (int(size // 2), int(size // 2))),
            nn.AvgPool2d(int(size // 2), (int(size // 2), int(size // 2))),
            nn.Flatten(),
            nn.Linear(in_feature, int(tmp_feature // 4)),
            nn.Linear(int(tmp_feature // 4), in_feature),
            nn.Sigmoid(),
            nn.Unflatten(1, (in_feature, 1, 1))
        ])

    def forward(self, x):
        """
        y = self.norm_act(x)
        y = y + self.lnet(y)
        y = y * self.attention(x)
        """
        y = x + self.lnet(x)
        return y


class Feature_forward(nn.Module):
    def __init__(self, feature_list, size_list, multi_num=4, push_num=5, s=4, p=0.2):
        """
        include in_feature,tmp_feature_1,.....,tmp_feature_n,out_feature
        """
        super(Feature_forward, self).__init__()
        assert push_num >= 1
        assert (
                multi_num == 1 or multi_num == 2 or multi_num == 4 or multi_num == 8 or multi_num == 16 or multi_num == 64)
        self.feature_list = feature_list
        self.size_list = size_list
        self.p = p
        self.multi_attention = nn.ModuleList(
            [MultiHeadAttention(int(multi_num * (size_list[-1] // s) ** 2),dropout=p) for _ in range(push_num)])
        self.resnet_forward = nn.ModuleList([nn.Sequential(
            *[Resnet_forward(feature_list[-1], feature_list[-1] * 2, size_list[-1], p=p) for _ in range(2)])
            for _ in range(push_num)])
        """
        self.out_conv=nn.Sequential(nn.Conv2d(3*self.feature_list[-1],self.feature_list[-1],(1,1),(1,1)),
                                    nn.BatchNorm2d(self.feature_list[-1]))
        self.multi_mix_layer = nn.ModuleList(
            [multi_mixer_layer(feature_list[-1], size_list[-1], s, multi_num=multi_num, push_num=push_num)])
        """

    def forward(self, x):
        x, pre_feature = x
        len_n = len(pre_feature)
        count = 0
        for attention, forward in zip(self.multi_attention, self.resnet_forward):
            b, c, h, w = x.shape
            """pre_feature[count%len_n]"""
            x = x.view(b, c, -1)
            x = attention(x, x, x)
            count += 1
            x_1 = x.view(b, c, h, w)
            x_1 = forward(x_1)
            if x_1.requires_grad == True:
                x_1 = F.dropout(x_1, p=self.p)
            else:
                x_1 = x_1
            x = x_1 + x.view(b,c,h,w)
        # result = self.batchnorm[0](result)
        # result = self.multi_mix_layer[0](result).view(b, c, h, w)
        return x
