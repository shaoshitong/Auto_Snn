import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
import math
import numpy as np
import os, sys
from torch.nn.parameter import Parameter


class attnetion(nn.Module):
    def __init__(self, x_channel, y_channel):
        super(attnetion, self).__init__()


def cat_result_get(tensor_prev, i, j, b, tag, pre_i, pre_j):
    m = []
    for t_i in range(i + 1):
        for t_j in range(j + 1):
            if t_i != i or t_j != j:
                if abs(t_i - t_j) < b:
                    m.append(tensor_prev[t_i][t_j])
    if tag == True:
        m.append(tensor_prev[pre_i][pre_j])
    return torch.cat(m, dim=1)


def token_numeric_get(x, y, b, f, d):
    push_list = []
    tensor_check = [[0 for i in range(y)] for j in range(x)]
    for i in range(1, x):
        tensor_check[i][0] = tensor_check[i - 1][0] + max(1, int(f / (d ** min(1, abs(i - 0))))) * int(abs(i - 0) < b)
    for i in range(1, y):
        tensor_check[0][i] = tensor_check[0][i - 1] + max(1, int(f / (d ** min(1, abs(i - 0))))) * int(abs(i - 0) < b)
    for i in range(1, x):
        for j in range(1, y):
            tensor_check[i][j] = tensor_check[i][j - 1] + tensor_check[i - 1][j] - tensor_check[i - 1][j - 1] + max(1,
                                                                                                                    int(f / (
                                                                                                                            d ** min(
                                                                                                                        1,
                                                                                                                        abs(i - j))))) * int(
                abs(i - j) < b)
    for i in range(0, x):
        for j in range(0, y):
            if i != 0 or j != 0:
                tensor_check[i][j] -= max(1, int(f / (d ** min(1, abs(i - j))))) * int(abs(i - j) < b)
    L = x + y - 1
    tag = 0
    for i in range(L):
        if tag == 0:
            for a_t in range(0, i + 1, 1):
                b_t = i - a_t
                if abs(a_t - b_t) < b and x > a_t >= 0 and y > b_t >= 0:
                    push_list.append([a_t, b_t])
            tag = 1
        else:
            for a_t in range(i, -1, -1):
                b_t = i - a_t
                if abs(a_t - b_t) < b and x > a_t >= 0 and y > b_t >= 0:
                    push_list.append([a_t, b_t])
            tag = 0
    for i in range(len(push_list[1:])):
        a_2, b_2 = push_list[i + 1]
        a_1, b_1 = push_list[i]
        if a_1 <= a_2 and b_1 <= b_2:
            continue
        else:
            tensor_check[a_2][b_2] += max(1, int(f / (d ** min(1, abs(a_1 - b_1)))))
    return tensor_check, push_list


def numeric_get(x, y, b):
    tensor_check = [[0 for i in range(y)] for j in range(x)]
    for i in range(1, x):
        tensor_check[i][0] = tensor_check[i - 1][0] + int(not (abs(i - 0) < b))
    for i in range(1, y):
        tensor_check[0][i] = tensor_check[0][i - 1] + int(not (abs(i - 0) < b))
    for i in range(1, x):
        for j in range(1, y):
            tensor_check[i][j] = tensor_check[i][j - 1] + tensor_check[i - 1][j] - tensor_check[i - 1][j - 1] + int(
                not (abs(i - j) < b))
    return tensor_check


def return_tensor_add(tensor_prev, i, j):
    p = (i + 1) * (j + 1) - 1
    for a in range(i + 1):
        for b in range(j + 1):
            if tensor_prev[i][j].shape[1] == tensor_prev[a][b].shape[1] and (a != i or b != j):
                tensor_prev[i][j] += (tensor_prev[a][b] / p)


class semhash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v1, v2, training=True):
        index = torch.randint(low=0, high=v1.shape[0], size=[int(v1.shape[0] / 2)]).long()
        v1[index] = v2[index]
        return v1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class aplha_decay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=2.):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        alpha = ctx.alpha
        return grad_outputs / alpha, None


class BasicUnit(nn.Module):
    def __init__(self, channel: int, hidden_channel: int, dropout: float):
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ("0_batchnorm2d", nn.BatchNorm2d(channel)),
            ("1_activation", nn.ReLU(inplace=False)),
            ("2_convolution", nn.Conv2d(channel, hidden_channel, (3, 3), stride=(1, 1), padding=1, bias=False)),
            ("3_batchnorm2d", nn.BatchNorm2d(hidden_channel)),
            ("4_activation", nn.ReLU(inplace=False)),
            ("5_dropout", nn.Dropout(dropout, inplace=False)),
            ("6_convolution", nn.Conv2d(hidden_channel, channel, (3, 3), stride=(1, 1), padding=1, bias=False))
        ]))

    def forward(self, x):
        return x + self.block(x)


class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, class_fusion, cat_x, cat_y):
        super(DenseLayer, self).__init__()
        self.nums_input_features = num_input_features
        if class_fusion == 0:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features,eps=1e-6)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1',
                            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=(1, 1), stride=(1, 1),
                                      padding=(0, 0), bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate,eps=1e-6)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=(5, 2), stride=(1, 1), dilation=(1, 2), padding=(2, 1),
                                               bias=False))
        elif class_fusion == 1:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features,eps=1e-6)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1',
                            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=(1, 1), stride=(1, 1),
                                      padding=(0, 0), bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate,eps=1e-6)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features,eps=1e-6)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1',
                            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=(1, 1), stride=(1, 1),
                                      padding=(0, 0), bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate,eps=1e-6)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=(2, 5), stride=(1, 1), dilation=(2, 1), padding=(1, 2),
                                               bias=False))
        # self.embedding=nn.Embedding()
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            if x.requires_grad:
                new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.Module):
    def __init__(self, cat_feature, eq_feature, hidden_size, cat_x, cat_y, dropout, class_fusion, size):
        super(DenseBlock, self).__init__()
        self.denselayer = DenseLayer(cat_feature, eq_feature, hidden_size, dropout, class_fusion, cat_x, cat_y)
        self.eq_feature = eq_feature
        self.cat_x = cat_x
        self.cat_y = cat_y
        self._initialize()
        kernel_size = 4

        l = int((size / kernel_size) ** 2)
        # self.p = torch.randperm(int(l), dtype=torch.long,requires_grad=False).cuda()
        # self.transformer = nn.TransformerEncoderLayer(size ** 2, l, dim_feedforward=int(size), batch_first=True,
        #                                               layer_norm_eps=1e-6)
        # self.unfold = lambda image: F.unfold(image, (kernel_size, kernel_size), stride=(kernel_size, kernel_size), )
        # self.fold = lambda image: F.fold(image, (size, size), (kernel_size, kernel_size),
        #                                  stride=(kernel_size, kernel_size))
        self.kernel_size = kernel_size

    def _initialize(self):
        if self.cat_x == self.cat_y:
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
        else:
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
        x = self.denselayer(x)
        """
        x = self.unfold(x)  # b,c*mh*mw,l
        x = x[..., self.p]/math.e  + x
        x = rearrange(x, "b ( c mh mw ) l -> b  c ( l mh mw )", mh=self.kernel_size, mw=self.kernel_size)
        x = rearrange(self.transformer(x), "b  c ( l mh mw ) -> b ( c mh mw ) l", mh=self.kernel_size,
                      mw=self.kernel_size)
        x = self.fold(x)
        """
        return x


class block_eq(nn.Module):
    def __init__(self, eq_feature, tmp_feature, dropout):
        super(block_eq, self).__init__()
        self.eq_feature = eq_feature
        self.tmp_feature = tmp_feature
        self.dropout = dropout
        self.basicunit = BasicUnit(eq_feature, tmp_feature, dropout)
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
        x = self.basicunit(x)
        return x


class multi_block_eq(nn.Module):
    def __init__(self, in_feature, out_feature, hidden_size, multi_k=1, stride=1, dropout=0.1):
        super(multi_block_eq, self).__init__()
        self.act = nn.Sequential(
            nn.BatchNorm2d(in_feature),
            nn.ReLU(inplace=False))
        self.downsample = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, (3, 3), stride=(stride, stride), padding=1, bias=False),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout, inplace=False),
            nn.Conv2d(out_feature, out_feature, (3, 3), stride=(1, 1), padding=1, bias=False),
        )
        self.res = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, (1, 1), stride=(stride, stride), padding=0, bias=False),
        )
        self.model = nn.Sequential(*[
            block_eq(out_feature, hidden_size, dropout) for _ in range(multi_k)
        ])
        self._initialize()

    def _initialize(self):
        for layer in self.downsample.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
        for layer in self.res.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
        for layer in self.act.modules():
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)

    def forward(self, x):
        x = self.model(x)
        return x


class mixer_GRU(nn.Module):
    def __init__(self, feature, hidden_size):
        super(mixer_GRU, self).__init__()
        self.convq1 = nn.Conv2d(feature + hidden_size, feature, (3, 3), (1, 1), (1, 1))
        self.convz1 = nn.Conv2d(feature + hidden_size, feature, (3, 3), (1, 1), (1, 1))
        self.convr1 = nn.Conv2d(feature + hidden_size, feature, (3, 3), (1, 1), (1, 1))
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, "fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)

    def forward(self, x):
        h, x = x
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


# (1-z)*[conv(xy)+(x+y)/2]+z*conv(q)+conv(xy)+(x+y)/2
class multi_GRU(nn.Module):
    def __init__(self, feature, hidden_size, dropout, layer):
        super(multi_GRU, self).__init__()
        self.feature = feature
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.convz1 = nn.Conv2d(feature + hidden_size, feature, (1, 3), padding=(0, 1))
        self.convr1 = nn.Conv2d(feature + hidden_size, hidden_size, (1, 3), padding=(0, 1))
        self.convq1 = nn.Conv2d(feature + hidden_size, feature, (1, 3), padding=(0, 1))
        self.convd1 = nn.Conv2d(feature + feature, hidden_size, (1, 1), padding=(0, 0))
        self.convd2 = nn.Conv2d(feature + feature, feature, (1, 1), padding=(0, 0))
        self.convz2 = nn.Conv2d(feature + hidden_size, feature, (3, 1), padding=(1, 0))
        self.convr2 = nn.Conv2d(feature + hidden_size, hidden_size, (3, 1), padding=(1, 0))
        self.convq2 = nn.Conv2d(feature + hidden_size, feature, (3, 1), padding=(1, 0))
        self.convz3 = nn.Conv2d(feature + feature, feature, (1, 1), padding=(0, 0))
        self._initialize()
        self.advance_layer = layer

    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="sigmoid")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
        nn.init.kaiming_normal_(self.convd1.weight.data, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.convd2.weight.data, mode="fan_in", nonlinearity="sigmoid")
        nn.init.kaiming_normal_(self.convq1.weight.data, mode="fan_in", nonlinearity="tanh")
        nn.init.kaiming_normal_(self.convq2.weight.data, mode="fan_in", nonlinearity="tanh")

    def forward(self, m):
        x, y, _ = m
        m = torch.cat([x, y], dim=1)
        h = self.convd1(m)
        p = torch.sigmoid(F.avg_pool2d(self.convd2(m), m.shape[-1]))
        x = (p) * x + (1 - p) * y
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(F.avg_pool2d(self.convz1(hx), hx.shape[-1]))
        r = torch.sigmoid(F.avg_pool2d(self.convr1(hx), hx.shape[-1]))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        x = F.relu((1 + z) * x + (1 - z) * q)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(F.avg_pool2d(self.convz2(hx), hx.shape[-1]))
        r = torch.sigmoid(F.avg_pool2d(self.convr2(hx), hx.shape[-1]))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        x = F.relu((1 + z) * x + (1 - z) * q)
        x = self.advance_layer(x)
        del m, z, r, q
        return x


class Cat(nn.Module):
    def __init__(self, i_feature, r_feature):
        super(Cat, self).__init__()
        self.i_feature = i_feature
        self.r_feature = r_feature
        self.convsig = nn.Conv2d(i_feature + r_feature, i_feature, (1, 1), (1, 1), (0, 0), bias=False)
        self._initialize()

    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="sigmoid")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
            elif isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)

    def forward(self, m):
        x, y = m
        m = torch.cat([x, y], dim=1)
        p = torch.sigmoid(self.convsig(m))
        return x * (1 + p) + y * (1 - p)
