import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import os, sys
from torch.nn.parameter import Parameter
class attnetion(nn.Module):
    def __init__(self,x_channel,y_channel):
        super(attnetion, self).__init__()

def cat_result_get(i,j,dil_rate):
    all = (i+1) * (j+1)-1
    choose = int(all * dil_rate)
    row, col = np.meshgrid( np.arange(0, i + 1, 1), np.arange(0, j + 1, 1))
    choose_list = [[0,0]]+sorted(list(filter(lambda x: (x[0] != 0 or x[1] != 0) and (x[0] != i or x[1] != j),
                              np.concatenate([row.flatten()[..., None], col.flatten()[..., None]], axis=1)[
                                  np.random.choice(all, choose, replace=False)].tolist())),key=lambda x:x[0]*(j+1)+x[1],reverse=False)
    return choose_list
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
class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate , class_fusion):
        super(DenseLayer, self).__init__()
        if class_fusion==0:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1',
                            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=(1, 1), stride=(1, 1),
                                      padding=(0, 0), bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=(2, 5), stride=(1, 1),dilation=(2,1), padding=(1, 2), bias=False))
        elif class_fusion==1:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1',
                            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=(1, 1), stride=(1, 1),
                                      padding=(0, 0), bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
        else:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1',
                            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=(1, 1), stride=(1, 1),
                                      padding=(0, 0), bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=(5, 2), stride=(1, 1), dilation=(1, 2), padding=(2, 1),bias=False))
        self.drop_rate = drop_rate


    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            if x.requires_grad:
                new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseLayer_first(nn.Sequential):
    def __init__(self,in_planes,bn_size,growth_rate):
        super(DenseLayer_first,self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_planes)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1',
                        nn.Conv2d(in_planes, bn_size * growth_rate, kernel_size=(1, 1), stride=(1, 1),
                                  padding=(0, 0), bias=False))
class DenseLayer_second(nn.Sequential):
    def __init__(self,bn_size,growth_rate,drop_rate,class_fusion):
        super(DenseLayer_second, self).__init__()
        if class_fusion==0:
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=(2, 5), stride=(1, 1),dilation=(2,1), padding=(1, 2), bias=False))
        elif class_fusion==1:
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
        else:
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=(5, 2), stride=(1, 1), dilation=(1, 2), padding=(2, 1),bias=False))
        self.drop_rate = drop_rate
    def forward(self, input):
        new_features=super(DenseLayer_second,self).forward(input)
        if self.drop_rate > 0:
            if input.requires_grad:
                new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features
class DenseLayer_last(nn.Module):
    def __init__(self,cat_feature,growth_rate,bn_size,cat_x,cat_y,dropout,class_fusion):
        super(DenseLayer_last, self).__init__()
        self.denselayer=nn.ModuleList([])
        for feature in cat_feature:
            self.denselayer.append(DenseLayer_first(feature,bn_size,growth_rate))
        self.transition=DenseLayer_second(bn_size,growth_rate,dropout,class_fusion)
    def forward(self,x,choose_indices,i,j):
        r=0.
        print(i,j,choose_indices)
        for indices in choose_indices:
            z=self.denselayer[indices[0]*(j+1)+indices[1]](x[indices[0]][indices[1]])
            r+=z
        r=self.transition(r)
        return r


class DenseBlock(nn.Module):
    def __init__(self,cat_feature,eq_feature,hidden_size,cat_x,cat_y,dropout ,class_fusion):
        super(DenseBlock, self).__init__()
        self.denselayer=DenseLayer_last(cat_feature,eq_feature,hidden_size,cat_x,cat_y,dropout,class_fusion)
        # self.denselayer=DenseLayer(cat_feature,eq_feature,hidden_size,dropout ,class_fusion)
        self.eq_feature=eq_feature
        self.cat_x=cat_x
        self.cat_y=cat_y
        self._initialize()
        # self.transformer=nn.TransformerEncoderLayer(eq_feature,1,dim_feedforward=int(eq_feature*1.5),batch_first=True,layer_norm_eps=1e-6)
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
    def forward(self,x,choose_indices,i,j):
        x=self.denselayer(x,choose_indices,i,j)
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

    def forward(self, x,choose_indices,i,j):

        x = self.model(x,choose_indices,i,j)
        return x


class mixer_GRU(nn.Module):
    def __init__(self, feature, hidden_size):
        super(mixer_GRU, self).__init__()
        self.convq1 = nn.Conv2d(feature + hidden_size, feature, (3, 3), (1, 1), (1, 1))
        self.convz1 = nn.Conv2d(feature + hidden_size, feature, (3, 3), (1, 1), (1, 1))
        self.convr1 = nn.Conv2d(feature + hidden_size, feature, (3, 3), (1, 1), (1, 1))
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data,"fan_in",nonlinearity="relu")
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
    def __init__(self, feature, hidden_size, dropout,layer):
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
        self.advance_layer=layer
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
        x, y ,_= m
        m=torch.cat([x, y], dim=1)
        h = self.convd1(m)
        p = torch.sigmoid(F.avg_pool2d(self.convd2(m),m.shape[-1]))
        x = (p) * x + (1 - p) * y
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(F.avg_pool2d(self.convz1(hx),hx.shape[-1]))
        r = torch.sigmoid(F.avg_pool2d(self.convr1(hx),hx.shape[-1]))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        x = F.relu((1 + z) * x + (1 - z) * q)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(F.avg_pool2d(self.convz2(hx),hx.shape[-1]))
        r = torch.sigmoid(F.avg_pool2d(self.convr2(hx),hx.shape[-1]))
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
