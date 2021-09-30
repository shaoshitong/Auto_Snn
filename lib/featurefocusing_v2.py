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
from lib.Wideresnet import WideResNetBlock
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
def size_change(f, s):
    def change(xx):
        xx: torch.Tensor
        xx = F.interpolate(xx, (s, s), mode='bilinear', align_corners=True)
        return xx

    return change

class MLP(nn.Sequential):
    def __init__(self, in_feature, out_feature, p=0.2000000000001):
        tmp_feature = int(math.sqrt(in_feature * out_feature) // 4)
        super(MLP, self).__init__()
        self.add_module("linear_1", nn.Linear(in_feature, tmp_feature))
        self.add_module("gelu", nn.GELU())
        self.add_module("linear_2", nn.Linear(tmp_feature, out_feature))

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

    def __init__(self, feature, n_head=8, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(feature, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(feature, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(feature, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, feature, bias=False)

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
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = rearrange(q, "a b c d -> a c ( b d )")
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class semhash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v1, v2, training=True):
        index = torch.randint(low=0, high=v1.shape[0], size=[int(v1.shape[1] / 2)]).long()
        v1[index] = v2[index]
        return v1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class block_ad(nn.Module):

    def __init__(self, in_channel, out_channel, tag=True, T=4):
        super(block_ad, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc1 = nn.Linear(in_channel, max(int(in_channel / T), 1))  ##全连接
        self.fc2 = nn.Linear(max(1, int(in_channel / T)), out_channel)
        self.training = True
        self.count = 0
        self.v1 = torch.randn(1, out_channel).cuda()
        self.sig = lambda x: torch.sigmoid(x) * 1.2 - 0.1
        # init.kaiming_normal(self.fc1.weight)
        self.fc1.weight.data.fill_(0)
        init.kaiming_normal_(self.fc2.weight)
        self.fc1.bias.data.fill_(1)
        self.fc2.bias.data.fill_(1)  ##数据用1填充
        self.tag = tag
        self.eps = 1e-3
        if tag == 0:
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False
            self.fc2.weight.requires_grad = False
            self.fc2.bias.requires_grad = False

    def forward(self, input):
        x = F.avg_pool2d(input, input.shape[-1])
        x = x.view(x.size(0), -1)
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        if y.requires_grad == True:
            g = torch.randn(y.shape).to(y.device)
            g = g + y
            v1 = torch.clamp(torch.sigmoid(g) * 1.2 - 0.1, torch.Tensor([0]).to(g.device),
                             torch.Tensor([1]).to(g.device))
            v2 = torch.gt(g, torch.Tensor([0]).to(g.device)).float()
            predict_bin = semhash.apply(v1, v2, self.training)
        else:
            predict_bin = (y > 0).float()
        return predict_bin.unsqueeze(-1).unsqueeze(-1)


class Resnet_forward(nn.Module):
    def __init__(self, in_feature, tmp_feature, p=0.2):
        super(Resnet_forward, self).__init__()
        self.lnet = nn.Sequential()
        self.in_feature = in_feature
        self.tmp_feature = tmp_feature
        self.lnet.add_module("batchnorm_0", nn.BatchNorm2d(in_feature))
        self.lnet.add_module("gelu", nn.GELU())
        self.lnet.add_module("conv1", nn.Conv2d(in_feature, tmp_feature, (3, 3), (1, 1), (1, 1)))
        self.lnet.add_module("batchnorm_1", nn.BatchNorm2d(tmp_feature))
        self.lnet.add_module("relu", nn.ReLU(inplace=True))
        self.lnet.add_module("dropout", nn.Dropout(p=0.1))
        self.lnet.add_module("conv2", nn.Conv2d(tmp_feature, in_feature, (3, 3), (1, 1), (1, 1)))
        self.attention = block_ad(in_feature, in_feature)

    def forward(self, x):
        x = self.attention(x) * self.lnet(x) + x
        x = F.relu_(x)
        return x


class Resnet_forward_block(nn.Module):
    def __init__(self, in_feature, tmp_feature, out_feature, depth, p=0.2, use_change=False):
        super(Resnet_forward_block, self).__init__()
        self.block_layer = nn.Sequential(*[
            Resnet_forward(in_feature, tmp_feature, p=p) for _ in range(depth)

        ], nn.BatchNorm2d(in_feature))
        self.p = p
        self.use_change = use_change
        if use_change == True:
            self.conv = nn.Conv2d(in_feature, out_feature, (2, 2), (2, 2), bias=False)
        else:
            self.conv = nn.Conv2d(in_feature, out_feature, (1, 1), (1, 1), bias=False)
        nn.init.xavier_uniform_(self.conv.weight.data)

    def forward(self, x):
        x = self.block_layer(x)
        if self.use_change == True:
            x = self.conv(x)
        if x.requires_grad == True:
            x = F.dropout(x, p=self.p)
        return x


class Feature_forward(nn.Module):
    def __init__(self, feature, size, push_num=5, s=4, p=0.2):
        """
        include in_feature,tmp_feature_1,.....,tmp_feature_n,out_feature
        """
        super(Feature_forward, self).__init__()
        assert push_num >= 1
        self.feature = feature[0]
        self.tag = feature[1]+1
        self.size = size
        self.p = p
        self.three_dim_layer_out_feature = self.feature[self.tag]
        self.f = nn.Sequential(*[
            nn.Conv2d(self.feature[0], self.feature[1], (3, 3), stride=(1, 1), padding=1, bias=False),
        ])
        self.resnet_forward = nn.ModuleList([
            WideResNetBlock(self.feature[1], self.feature[2], 1, push_num, dropout=p, use_pool=False),
            WideResNetBlock(self.feature[2], self.feature[3], 2, push_num, dropout=p, use_pool=False),
            WideResNetBlock(self.feature[3], self.feature[4], 2, push_num, dropout=p, use_pool=False),
        ])
        self.transition_layer = nn.ModuleList([
            nn.Conv2d( self.feature[2]*2, self.feature[2], (1, 1), (1, 1), bias=False),
            nn.Conv2d( self.feature[3]*2, self.feature[3], (1, 1), (1, 1), bias=False),
            nn.Conv2d( self.feature[4]*2, self.feature[4], ( 1, 1), (1, 1), bias=False),
        ])
        self.balance_layer= nn.ModuleList([
            nn.Sequential(*[nn.Conv2d(self.three_dim_layer_out_feature*3, self.feature[2], (1, 1), (1, 1), bias=False),
                            nn.BatchNorm2d(self.feature[2]),
                            ]),
            nn.Sequential(*[nn.Conv2d(self.three_dim_layer_out_feature*3, self.feature[3], (1, 1), (1, 1), bias=False),
                            nn.BatchNorm2d(self.feature[3]),
                            ]),
            nn.Sequential(*[nn.Conv2d(self.three_dim_layer_out_feature*3, self.feature[4], (1, 1), (1, 1), bias=False),
                            nn.BatchNorm2d(self.feature[4]),
                            ]),
        ])

    def forward(self, x, A, B, C):
        self.kl_loss = 0.
        x = self.f(x)
        feature_list = torch.cat([A, B, C],dim=1)
        for transition, forward, balance in zip(self.transition_layer, self.resnet_forward,self.balance_layer):
            """pre_feature[count%len_n]"""
            x = forward(x)
            feature = size_change(x.size()[1], x.size()[2])(balance(feature_list))
            x = transition(torch.cat((x, feature), dim=1))
            log_soft_x = F.log_softmax(F.avg_pool2d(x, x.shape[-1]).squeeze(), dim=-1)
            soft_y = F.softmax(F.avg_pool2d(feature, feature.shape[-1]).squeeze(), dim=-1) + 1e-8
            self.kl_loss = self.kl_loss + F.kl_div(log_soft_x, soft_y, reduction='none').sum(dim=-1).mean()
        return x
