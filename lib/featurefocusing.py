import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from Snn_Auto_master.lib.activation import Activation
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
def block_conv(eq_feature,groups=1):
    return nn.Sequential(nn.LeakyReLU(1e-2,inplace=True),
                         nn.BatchNorm2d(eq_feature*3),
                         nn.Conv2d(eq_feature*3,eq_feature*3,(3,3),padding=1,stride=1,groups=groups))
def norm_layer(input):
    input:torch.Tensor
    input_norm=input.norm(dim=1,p=2,keepdim=True)
    return torch.div(input,input_norm)
class feature_norm_layer(nn.Module):
    def __init__(self,eq_feature,groups=1):
        super(feature_norm_layer,self).__init__()
        self.eq_feature=eq_feature
        self.block_conv=block_conv(self.eq_feature,groups=groups)
        self.norm_conv=norm_layer
    def forward(self,x):
        y=self.norm_conv(self.block_conv(x))
        return y+x
class Feature_parallel(nn.Module):
    def __init__(self,feature_list,latest=False):
        super(Feature_parallel,self).__init__()
        self.feature_list=feature_list
        self.block_conv=\
            nn.ModuleList([feature_norm_layer(eq_feature=self.feature_list[0],groups=i) for i
             in [self.feature_list[0]]*2])
        if latest==False:
            self.out_conv = nn.Conv2d(self.feature_list[0] * 3, self.feature_list[1], stride=2, kernel_size=(2, 2), )
        else:
            self.out_conv = nn.Conv2d(self.feature_list[0] * 3, self.feature_list[0], stride=1, kernel_size=(1, 1), )
    def forward(self,x_list):
        tuple_list=[]
        f_list=[]
        for x in x_list:
            x:torch.Tensor
            x_tuple=x.split(dim=1,split_size=1)
            tuple_list.append(x_tuple)
        for i in zip(*tuple_list):
            f_list.append(torch.cat(i,dim=1))
        group_input=torch.cat(tuple(f_list),dim=1)
        for Net in self.block_conv:
            group_input=Net(group_input)
        return self.out_conv(group_input)
class Feature_forward(nn.Module):
    def __init__(self,feature_list):
        super(Feature_forward,self).__init__()
        self.feature_list=feature_list
        self.Feature_parallel=[]
        self.feature_list2=feature_list[2:]
        len_1=len(self.feature_list)
        for i in range(len_1-1):
            self.Feature_parallel.append(Feature_parallel(copy.deepcopy(self.feature_list),(i==len_1-2)))
            self.feature_list.pop(0)
        self.Feature_parallel=nn.ModuleList(self.Feature_parallel)
        self.conv_list=nn.ModuleList([nn.Conv2d(i*2,i,(1,1),stride=1) for i in self.feature_list2])
    def forward(self,x_lists):
        begin=[]
        for i,x_list in enumerate(x_lists):
            if len(begin)!=0:
                if i!=len(x_lists)-1:
                    begin.append(torch.cat((self.Feature_parallel[i](x_list),F.interpolate(begin[-1],scale_factor=0.5).repeat(1,2,1,1)),dim=1))
                else:
                    begin.append(torch.cat((self.Feature_parallel[i](x_list),begin[-1]),dim=1))
                begin[-1]=self.conv_list[i-1](begin[-1])
            else:
                begin.append(self.Feature_parallel[i](x_list))
        for i,b in enumerate(begin):
            if i!=len(begin)-1:
                if i!=len(begin)-2:
                    begin[i+1]+=F.interpolate(begin[i], scale_factor=0.5).repeat(1, 2, 1, 1)
                else:
                    begin[i+1]+=begin[i]
        return begin[-1]









