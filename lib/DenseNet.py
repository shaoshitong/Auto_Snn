import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.nn.utils as utils
import numpy as np
import os, sys
import copy
class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=(1, 1), stride=(1, 1), bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
class DenseBlock(nn.Sequential):
    def __init__(self, num_input_features_list, bn_size, drop_rate, num_layers=None, ):
        super(DenseBlock, self).__init__()
        if num_layers == None:
            num_layers = len(num_input_features_list) - 1
        assert num_layers <= len(num_input_features_list) - 1
        for i in range(num_layers):
            layer = DenseLayer(sum(num_input_features_list[:i + 1]), num_input_features_list[i + 1], bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=(1, 1), stride=(1, 1), bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class ForwardTurning(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(ForwardTurning, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=(1, 1), stride=(1, 1), bias=False))


class DenseNet(nn.Module):
    def __init__(self, num_input_feature_list_list, bn_size, drop_rate, use_size_change: bool, num_layer=None, ):
        super(DenseNet, self).__init__()
        print(num_input_feature_list_list)
        self.densenet = nn.Sequential()
        for i, num_input_feature_list in enumerate(num_input_feature_list_list):
            self.densenet.add_module(       "denseblock%d" % i,
                                            DenseBlock(num_input_feature_list,
                                            bn_size, drop_rate, num_layer))
            if i != len(num_input_feature_list_list) - 1 and use_size_change:
                self.densenet.add_module(   "transition%d" % i,
                                            Transition(sum(num_input_feature_list),
                                            num_input_feature_list_list[i + 1][0]))
            elif i != len(num_input_feature_list_list) - 1 and not use_size_change:
                self.densenet.add_module(   "transition%d" % i,
                                            ForwardTurning(sum(num_input_feature_list),
                                            num_input_feature_list_list[i + 1][0]))
            else:
                self.densenet.add_module(   "transition%d" % i,
                                            ForwardTurning(sum(num_input_feature_list),
                                            num_input_feature_list[-1]))
        self._initialize()

    def forward(self, x):
        return self.densenet(x)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
