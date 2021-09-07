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
class linear(nn.Module):
    def __init__(self,in_feature,out_feature):
        super(linear,self).__init__()
        self.weight=Parameter(torch.Tensor(in_feature,out_feature))
        self.bias=Parameter(torch.Tensor(1,out_feature))
        self._initialize()
    def forward(self,x):
        x=x@self.weight+self.bias
        return x
    def _initialize(self):
        self.weight.data.fill_(1.)
        self.bias.data.fill_(0.)


class MLP(nn.ModuleList):
    def __init__(self,in_feature,out_feature,p=0.2000000000001):
        tmp_feature=int(math.sqrt(in_feature*out_feature)//4)
        super(MLP,self).__init__([])
        self.append(linear(in_feature,tmp_feature))
        self.append(nn.GELU())
        self.append(linear(tmp_feature,out_feature))
        self.p=p
        self.dropout=nn.Dropout(p=p)
    def forward(self,x):
        x=super(MLP,self).forward(x)
        if abs(self.p-0.2000000000001)<0.000001:
            x=self.dropout(x)
        return x
class multi_MLP(nn.Module):
    def __init__(self,in_feature,out_feature,size,p,split_num=4):
        super(multi_MLP,self).__init__()
        self.split_num=split_num
        self.size=size
        assert int(math.sqrt(split_num))**2==split_num
        assert size%int(math.sqrt(split_num))==0
        self.sqrt_size=int(size//math.sqrt(split_num))
        self.linear_list=nn.ModuleList([MLP(in_feature*self.sqrt_size*self.sqrt_size,
                                            out_feature*self.sqrt_size*self.sqrt_size,p) for _ in range(split_num)])
    def unfold(self,x):
        return F.unfold(x,_pair(self.sqrt_size),stride=self.sqrt_size ,)  # [B, C* sqrt_size*sqrt_size, L]
    def fold(self,x):
        return F.fold(x,(self.size,self.size),(self.sqrt_size,self.sqrt_size),stride=self.sqrt_size)
    def forward(self,x):
        x_L=self.unfold(x)
        B,C_S,L=x_L.shape
        for i in range(L):
            x_L[...:i]=self.linear_list[i](x_L[...:i])
        x_L=self.fold(x_L)
        return x_L
class parallel_multi_MLP(nn.Module):
    def __init__(self,in_feature,out_feature,size,p,split_num=4):
        super(parallel_multi_MLP, self).__init__()
        self.mylti_MLP_list=nn.ModuleList([multi_MLP(in_feature,out_feature,size,split_num,p) for _ in range(3)])
    def forward(self,x):
        a,b,c=x
        a=self.mylti_MLP_list[0](a)
        b=self.mylti_MLP_list[1](b)
        c=self.mylti_MLP_list[2](c)
        return (a,b,c)
class multi_parallel_multi_MLP(nn.Module):
    def __init__(self,in_feature,out_feature,size,out_size,p,split_num=4,parallel_num=1):
        super(multi_parallel_multi_MLP, self).__init__()
        self.parallel_multi_MLP_list=nn.ModuleList([parallel_multi_MLP(in_feature,in_feature,size,split_num,p) for _ in range(parallel_num-1)])
        self.parallel_multi_MLP_list.append(parallel_multi_MLP(in_feature,out_feature,size,split_num,p))
        self.multi_conv=nn.ModuleList([nn.Conv2d(in_feature,in_feature,(1,1),(1,1)),nn.Conv2d(in_feature,out_feature,(1,1),(1,1)),
                                       nn.Conv2d(in_feature,in_feature,(1,1),(1,1)),nn.Conv2d(in_feature,out_feature,(1,1),(1,1)),
                                       nn.Conv2d(in_feature,in_feature,(1,1),(1,1)),nn.Conv2d(in_feature,out_feature,(1,1),(1,1))])
        self.out_size=out_size
    def forward(self,x):
        a,b,c=x
        a_1,b_1,c_1=a.clone(),b.clone(),c.clone()
        for i in range(len(self.parallel_multi_MLP_list)-1):
            a,b,c=self.parallel_multi_MLP_list[i]((a,b,c))
        a=self.multi_conv[0](a_1)+a
        b=self.multi_conv[2](b_1)+b
        c=self.multi_conv[4](c_1)+c
        a_1, b_1, c_1 = a.clone(), b.clone(), c.clone()
        a,b,c=self.parallel_multi_MLP_list[-1]((a,b,c))
        a=F.interpolate(self.multi_conv[1](a_1)+a,size=self.out_size)
        b=F.interpolate(self.multi_conv[3](b_1)+b,size=self.out_size)
        c=F.interpolate(self.multi_conv[5](c_1)+c,size=self.out_size)
        return (a,b,c)
class Feature_forward(nn.Module):
    def __init__(self, feature_list,size_list,split_num=4, p=0.2):
        """
        include in_feature,tmp_feature_1,.....,tmp_feature_n,out_feature
        """
        super(Feature_forward, self).__init__()
        self.feature_list = feature_list
        self.size_list=size_list
        self.multi_multi_parallel_multi_MLP_list=nn.ModuleList([multi_parallel_multi_MLP(self.feature_list[i],self.feature_list[-1],
                                                                                         self.size_list[i],self.size_list[-1],p,
                                                                                         split_num=split_num,parallel_num=len(self.feature_list)-1-i) for i in range(len(self.feature_list)-1)])
        self.out_conv=nn.Sequential(nn.Conv2d((3*len(self.feature_list)-3)*self.feature_list[-1],self.feature_list[-1],(1,1),(1,1)),
                                    nn.BatchNorm2d(self.feature_list[-1]))
    def forward(self, x_lists):
        for i in range(len(x_lists)):
            x_lists[i]=torch.cat(self.multi_multi_parallel_multi_MLP_list[i](x_lists[i]),dim=1)
        x_lists=torch.cat(x_lists,dim=1)
        return self.out_conv(x_lists)
