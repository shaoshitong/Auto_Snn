import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import os, sys
from torch.nn.parameter import Parameter


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
    def __init__(self, feature, hidden_size, dropout):
        super(multi_GRU, self).__init__()
        self.feature = feature
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.convz1 = nn.Conv2d(feature + hidden_size, feature, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(feature + hidden_size, feature, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(feature + hidden_size, feature, (1, 5), padding=(0, 2))
        self.convd1 = nn.Conv2d(feature + feature, feature, (1, 1), padding=(0, 0))
        self.convd2 = nn.Conv2d(feature + feature, hidden_size, (1, 1), padding=(0, 0))
        self.convz2 = nn.Conv2d(feature + hidden_size, feature, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(feature + hidden_size, feature, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(feature + hidden_size, feature, (5, 1), padding=(2, 0))
        # self.convz3 = nn.Conv2d(feature + feature, feature, (1, 1), padding=(0, 0))
        self.convo = nn.Sequential(*[
            nn.Conv2d(feature, feature, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(feature),
        ])
        self._initialize()

    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="sigmoid")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
        for layer in self.convo.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)
        nn.init.kaiming_normal_(self.convd1.weight.data, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.convd2.weight.data, mode="fan_in", nonlinearity="relu")

        nn.init.kaiming_normal_(self.convq1.weight.data, mode="fan_in", nonlinearity="tanh")
        nn.init.kaiming_normal_(self.convq2.weight.data, mode="fan_in", nonlinearity="tanh")

    def forward(self, m):
        x, y = m
        m = torch.cat([x, y], dim=1)
        h = (x+y)/2
        x = self.convd2(m)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 + z) * h + (1 - z) * q
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 + z) * h + (1 - z) * q
        del m, z, r, q
        return h


class Cat(nn.Module):
    def __init__(self, i_feature, r_feature):
        super(Cat, self).__init__()
        self.i_feature = i_feature
        self.r_feature = r_feature
        self.conv1 = nn.Sequential(*[nn.Conv2d(i_feature, i_feature, (1, 1), (1, 1), bias=False),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(i_feature),
                                     nn.Conv2d(i_feature, i_feature, (3, 3), (1, 1), (1, 1), bias=False)])
        self.conv2 = nn.Sequential(*[nn.Conv2d(r_feature, i_feature, (1, 1), (1, 1), bias=False),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm2d(i_feature),
                                     nn.Conv2d(i_feature, i_feature, (3, 3), (1, 1), (1, 1), bias=False)])
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

    def forward(self, m):
        x, y = m
        xy = self.conv1(x) + self.conv2(y)
        return xy
