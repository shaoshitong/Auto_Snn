import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
class Attention(nn.Module):
    def __init__(self,in_feature,out_feature,in_pointnum,out_pointnum,head_num=8,use_bias=False,require_grad=True):
        super(Attention,self).__init__()
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.use_bias=use_bias
        self.requires_grad=require_grad
        self.in_pointnum=in_pointnum
        self.out_pointnum=out_pointnum
        self.head_num=head_num
        self.weight=Parameter(torch.Tensor(in_pointnum,head_num*head_num),requires_grad=True)
        stdv = 6. / math.sqrt(self.weight.data.size(1) + self.weight.data.size(0))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if use_bias==True:
            self.bias=Parameter(torch.Tensor([head_num*head_num]),requires_grad=True)
            self.bias.data.fill_(.1)
            pass
    def forward(self,input):
        x:torch.Tensor
        y:torch.Tensor
        z:torch.Tensor
        x,y,z=input # batch_size,in_feature*width*height
        x,y,z=x.unsqueeze(-1).view(-1,self.in_pointnum,self.in_feature),\
              y.unsqueeze(-1).view(-1,self.in_pointnum,self.in_feature),\
              z.unsqueeze(-1).view(-1,self.in_pointnum,self.in_feature)
        Feature_concat=torch.cat([x,y,z],dim=-1)
        Feature_concat=Feature_concat.permute(0,2,1).view(-1,self.in_pointnum) # batchsize , in_feature*3, weight*height
        Feature_concat=torch.matmul(Feature_concat,self.weight)
        if self.use_bias==True:
            Feature_concat+=self.bias





