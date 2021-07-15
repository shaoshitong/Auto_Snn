import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.data
import numpy as np
def rand_one_in_array(count,arr):
    assert count > 0
    index = torch.randperm(count, dtype=torch.long).cuda()
    return torch.index_select(arr, 1, index)
class Activation(nn.Module):
    def __init__(self,nDict,in_pointer=None,out_pointer=None,transform=True,use_bias=True,require_grad=False,LReLU=0.6,ELU=0.6):
        super(Activation,self).__init__()
        self.nDict=nDict
        self.in_pointer=in_pointer
        self.out_pointer=out_pointer
        self.use_bias=use_bias
        self.requires_grad=require_grad
        self.transform=transform
        self.relu1=nn.LeakyReLU(LReLU)
        self.relu2=nn.ReLU()
        self.relu3=nn.ELU()
        self.relu4=nn.LeakyReLU(0.8)
        self.relu5=nn.Hardtanh()
        self.relu6=nn.Sigmoid()
    def forward(self,x):
        nDict=self.nDict
        if 'sigmoid' in nDict:
            x=self.relu6(x)
        if 'hardtanh' in nDict:
            x=self.relu5(x)
        if 'elu' in nDict:
            x=self.relu3(x)
        if 'negative' in nDict:
            x=-self.relu4(-x)
        if 'relu' in nDict:
            x=self.relu2(x)
        if 'leakyrelu' in nDict:
            x=self.relu1(x)
        if self.transform==True:
            x=self.translation2(x)
        return x
    def translation1(self,x,btas=10):
        r,c=x.shape
        arr = torch.tensor(np.random.randint(0,c//btas,size=(r,c)),dtype=torch.int32)
        for i in range(x.shape[0]):
            x[i,:]=x[i,arr[i]]
        return x
    def translation2(self,x,nums=3,split=7):
        r,c=x.shape
        index = torch.randint(low=0, high=x.shape[0], size=[int(x.shape[0] / nums)]).long()
        y=x.clone()
        for i in range(c//split):
            y[index,i*split:(i+1)*split]=rand_one_in_array(split,x[index,i*split:(i+1)*split])
        return y
