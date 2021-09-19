import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
class BasicUnit(nn.Module):
    def __init__(self,channel:int,dropout:float):
        super(BasicUnit, self).__init__()
        self.block=nn.Sequential(OrderedDict([
            ("0_batchnorm2d",nn.BatchNorm2d(channel)),
            ("1_activation",nn.ReLU(inplace=True)),
            ("2_convolution",nn.Conv2d(channel,channel,(3,3),stride=(1,1),padding=1,bias=False)),
            ("3_batchnorm2d",nn.BatchNorm2d(channel)),
            ("4_activation",nn.ReLU(inplace=False)),
            ("5_dropout",nn.Dropout(dropout,inplace=False)),
            ("6_convolution",nn.Conv2d(channel,channel,(3,3),stride=(1,1),padding=1,bias=False))
        ]))
    def forward(self,x):
        return x+self.block(x)
class Downsampleunit(nn.Module):
    def __init__(self,in_channel:int ,out_channel:int,stride:int,dropout:float,use_pool=False):
        super(Downsampleunit, self).__init__()
        self.norm=nn.Sequential(*[
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=False),
        ])
        self.block=nn.Sequential(*[
            nn.Conv2d(in_channel,out_channel,(3,3),stride=(stride,stride),padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout,inplace=False),
            nn.Conv2d(out_channel,out_channel,(3,3),stride=(1,1),padding=1,bias=False),
        ])
        if use_pool==False:
            self.downsample=nn.Conv2d(in_channel,out_channel,(1,1),stride=(stride,stride),padding=0,bias=False)
        else:
            self.downsample=nn.MaxPool2d((stride,stride),stride=(stride,stride),padding=0)
    def forward(self,x):
        x=self.norm(x)
        return self.block(x)+self.downsample(x)
class WideResNetBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,stride:int,depth:int,dropout:float,use_pool:bool):
        super(WideResNetBlock, self).__init__()
        self.block_1=nn.Sequential(*[
            Downsampleunit(in_channels,out_channels,stride,dropout,use_pool)
        ])
        self.block_2=nn.Sequential(*[
            BasicUnit(out_channels,dropout) for _ in range(depth)
        ])
    def forward(self,x):
        x=self.block_1(x)
        x=self.block_2(x)
        return x
    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data,mode="fan_in",nonlinearity="relu")
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer,nn.BatchNorm2d):
                layer.weight.data.fill_(1.)
                layer.bias.data.zero_()
