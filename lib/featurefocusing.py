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
import torch
import torch.nn.init as init
import math
class Sigmoid_w_b(nn.Module):
    '''因为Variable自动求导，所以不需要实现backward()'''
    def __init__(self):
        super().__init__()
    def forward(self, x):  # 参数 x  是一个Variable对象
        x = torch.sigmoid(x)*1.2-0.1
        return x # 让b的形状符合  输出的x的形状
class semhash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v1, v2, training=True):
        index = torch.randint(low=0, high=v1.shape[0], size=[int(v1.shape[1]/2)]).long()
        v1[index] = v2[index]
        return v1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
class block_ad(nn.Module):

    def __init__(self, in_channel, out_channel,tag=True, T=4):
        super(block_ad, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc1 = nn.Linear(in_channel, max(int(in_channel/T),1))##全连接
        self.fc2 = nn.Linear(max(1,int(in_channel/T)), out_channel)
        self.training = True
        self.count=0
        self.v1=torch.randn(1,out_channel).cuda()
        self.sig=Sigmoid_w_b()
        # init.kaiming_normal(self.fc1.weight)
        self.fc1.weight.data.fill_(0)
        init.kaiming_normal_(self.fc2.weight)
        self.fc1.bias.data.fill_(1)
        self.fc2.bias.data.fill_(1)##数据用1填充
        self.tag=tag
        if tag==0:
            self.fc1.weight.requires_grad=False
            self.fc1.bias.requires_grad = False
            self.fc2.weight.requires_grad=False
            self.fc2.bias.requires_grad = False
    def forward(self, input):
        x = F.avg_pool2d(input, input.shape[3])
        x = x.view(x.size(0), -1)
        y = F.relu(self.fc1(x))
        self.y = self.fc2(y)
        mid = self.sig(self.y)
        if self.training:
            if self.v1.shape[0] is not mid.shape[0]:
                with torch.no_grad():
                    self.v1 = mid.clone()
                    self.count = 0
            with torch.no_grad():
                a=torch.abs(self.v1-mid).clone()
            if str(next(self.fc1.parameters()).device) == "cpu":
                a=torch.min(a,torch.Tensor([1]).cpu())
            else:
                a=torch.min(a,torch.Tensor([1]).cuda())
            if self.count is not 0:
                a=torch.normal(mean=0,std=a).cuda()/(2+math.pow(self.count,1/4))
            if str(next(self.fc1.parameters()).device)=="cpu":
                v1 = torch.min(torch.max(mid, torch.Tensor([0]).cpu()), torch.Tensor([1]).cpu())
            else:
                v1 = torch.min(torch.max(mid, torch.Tensor([0]).cuda()), torch.Tensor([1]).cuda())
            mid=mid+a
            self.v1=mid
            self.count=self.count+1
            if str(next(self.fc1.parameters()).device)=="cpu":
                v2 = torch.min(torch.max(mid, torch.Tensor([0]).cpu()), torch.Tensor([1]).cpu())
            else:
                v2 = torch.min(torch.max(mid, torch.Tensor([0]).cuda()), torch.Tensor([1]).cuda())
            predict_bin = semhash.apply(v1, v2, self.training)
        else:
            if str(next(self.fc1.parameters()).device)=="cpu":
                predict_bin = torch.min(torch.max(mid, torch.Tensor([0]).cpu()), torch.Tensor([1]).cpu())
            else:
                predict_bin = torch.min(torch.max(mid, torch.Tensor([0]).cuda()), torch.Tensor([1]).cuda())
        return predict_bin.unsqueeze(2).unsqueeze(3)

"""=======================================================above is block========================================================="""

def block_conv(eq_feature,groups=1):
    model1=nn.Sequential(nn.LeakyReLU(1e-2),
                         nn.BatchNorm2d(eq_feature*3),
                         nn.Conv2d(eq_feature*3,eq_feature*2,(3,3),padding=1,stride=1,groups=groups),
                         nn.Conv2d(eq_feature*2,eq_feature*2,(3,3),padding=1,stride=1,groups=1),
                         nn.Conv2d(eq_feature*2,eq_feature*3,(3,3),stride=1,padding=1,groups=groups),
                         )
    model2=nn.Sequential(block_ad(eq_feature*3,eq_feature*3))
    return model1,model2
def norm_layer(input):
    input:torch.Tensor
    input_norm=input.norm(dim=1,p=2,keepdim=True)
    return torch.div(input,input_norm)
class feature_norm_layer(nn.Module):
    def __init__(self,eq_feature,groups=1):
        super(feature_norm_layer,self).__init__()
        self.eq_feature=eq_feature
        self.block_conv1,self.block_conv2=block_conv(self.eq_feature,groups=groups)
        self.norm_conv=norm_layer
    def forward(self,x):
        x_f=self.block_conv2(x)
        x=(self.block_conv1(x)+x)*x_f
        y=self.norm_conv(x)
        return y
class Feature_parallel(nn.Module):
    def __init__(self,feature_list,latest=False):
        super(Feature_parallel,self).__init__()
        self.feature_list=feature_list
        self.block_conv=\
            nn.ModuleList([feature_norm_layer(eq_feature=self.feature_list[0],groups=i) for i
             in [self.feature_list[0]]*3])
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
        group_tmp=group_input
        for Net in self.block_conv:
            group_tmp=Net(group_tmp)
        group_tmp+=group_input
        return self.out_conv(group_tmp)
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









