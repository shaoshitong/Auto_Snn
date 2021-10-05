import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import os, sys
from torch.nn.parameter import Parameter


class semhash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v1, v2, training=True):
        index = torch.randint(low=0, high=v1.shape[0], size=[int(v1.shape[0] / 2)]).long()
        v1[index] = v2[index]
        return v1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


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


class multi_GRU(nn.Module):
    def __init__(self, feature, hidden_size, dropout, in_size, out_size, layer):
        super(multi_GRU, self).__init__()
        self.feature = feature
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.inx = nn.Conv2d(feature, hidden_size, (1, 1), (1, 1), (0, 0), bias=False)
        self.iny = nn.Conv2d(feature, hidden_size, (1, 1), (1, 1), (0, 0), bias=False)
        self.convz1 = nn.Conv2d(feature + feature, feature, (1, 1), (1, 1), (0, 0))
        self.convr1 = nn.Conv2d(feature + feature, feature, (1, 1), (1, 1), (0, 0))
        self.convq1 = nn.Conv2d(feature + feature, feature, (1, 1), (1, 1), (0, 0), bias=False)
        self.convd1 = nn.Conv2d(feature * 2, feature, (1, 1), (1, 1), (0, 0), bias=False)
        self.convd2 = nn.Conv2d(feature * 2, feature, (1, 1), (1, 1), (0, 0), bias=False)
        self.b = nn.BatchNorm2d(hidden_size * 2)
        self.r = nn.ReLU(inplace=True)
        self._initialize()
        self.advance_layer = layer

    def _initialize(self):
        nn.init.kaiming_normal_(self.convr1.weight.data, mode="fan_in", nonlinearity="sigmoid")
        nn.init.xavier_normal_(self.convq1.weight.data,)
        nn.init.kaiming_normal_(self.convz1.weight.data, mode="fan_in", nonlinearity="sigmoid")
        nn.init.xavier_normal_(self.convd1.weight.data)
        nn.init.kaiming_normal_(self.convd2.weight.data, mode="fan_in", nonlinearity="sigmoid")
        nn.init.zeros_(self.convr1.bias.data)
        nn.init.zeros_(self.convz1.bias.data)
        nn.init.ones_(self.b.weight.data)
        nn.init.zeros_(self.b.bias.data)

    def forward(self, m):
        x, y, pre = m
        m = torch.cat([x, y], dim=1)
        z = torch.sigmoid(self.convz1(m))
        r = torch.sigmoid(self.convr1(m))
        q = self.convq1(torch.cat([r * y  , (1 - r) * x ], dim=1))
        x = (1 - z) * (x + y) + (z) * q + pre/2
        x = self.advance_layer(x)
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
        return x * (1 - p) + y * (p)
