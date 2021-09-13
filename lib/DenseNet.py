import torch
import torch.nn as nn
import torch.nn.functional as F
from  Snn_Auto_master.lib.SNnorm import SNConv2d
import torch.nn.utils as utils
import numpy as np
import os,sys
import copy
class Denseblock(nn.Sequential):
    def __init__(self,in_feature,out_feature,training_rate=True):
        super(Denseblock,self).__init__()
        self.add_module("norm1",nn.BatchNorm2d(in_feature))
        self.add_module("relu1",nn.ReLU(inplace=True))
        self.add_module("conv1",nn.Conv2d(in_feature,out_feature,(1,1),stride=1,padding=0,bias=False))
        self.add_module("pad1", nn.ReflectionPad2d(1))
        self.add_module("conv2",nn.Conv2d(out_feature,out_feature,(3,3),stride=1,padding=0))
        self.training=training_rate
    def forward(self,x):
        new_features = super(Denseblock, self).forward(x)
        return new_features
class Denselayer(nn.Module):
    def __init__(self,feature_list,training=True):
        super(Denselayer,self).__init__()
        self.modellist=[]
        for i,feature_in in enumerate(feature_list):
            if feature_in!=feature_list[-1]:
                in_feature=feature_list[i]
                out_feature=feature_list[i+1]
                self.modellist.append(Denseblock(in_feature,out_feature,training))
        self.modellist=nn.ModuleList(self.modellist)
        self.old_feature_list=copy.deepcopy(feature_list)
        self.training=training
        for i,feature in enumerate(feature_list):
            if i!=len(feature_list)-1:
                feature_list[i+1]+=feature_list[i]
        self.joinlist=nn.ModuleList([
            nn.Sequential(nn.BatchNorm2d(feature_list[i]),
                          nn.ReLU(),
                          nn.Conv2d(feature_list[i],self.old_feature_list[i],(1,1),stride=1,padding=0)) for i in range(1,len(feature_list))
        ])
    def forward(self,x,training=True):
        pre=x
        for (model,join) in zip(self.modellist,self.joinlist):
            x=model(x)
            pre=torch.cat((pre,x),dim=1)
            x=join(pre)
            if training==True:
                x=F.dropout(x,p=0.2)
        return x


