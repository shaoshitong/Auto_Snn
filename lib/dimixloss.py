import torch
import torch.nn as nn
import torch.nn.functional as F
def feature_normalize(feature_in, eps=1e-10):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + eps
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm


def weighted_l1_loss(input, target, weights):
    out = torch.abs(input - target)
    out = out * weights.expand_as(out)
    loss = out.mean()
    return loss


def mse_loss(input, target=0):
    return torch.mean((input - target) ** 2)
def MatmulTopkLoss(X,Y):
    X:torch.Tensor
    Y:torch.Tensor
    Y_permute=Y.permute(0,2,1)
    Y_p_mat_X=torch.matmul(X,Y_permute)
    YX=Y_p_mat_X+Y_p_mat_X.permute(0,2,1)
    M=F.softmax(YX,dim=-1)
    value,key=torch.topk(M,M.size()[-1]//2,dim=-1)# batchsize size size//2
    value_sum=value.sum(dim=-1,keepdim=True)
    index=torch.arange(0,value.size()[1]).unsqueeze(0).unsqueeze(-1).to(key.device)# 1,size,1
    C_xy=value*torch.abs(key-index)/(value_sum)
    C_xy=C_xy.mean(-1)
    return C_xy



class DimixLoss_neg(nn.Module):
    def __init__(self,list_len=1):
        super(DimixLoss_neg,self).__init__()
        self.list_len=list_len
    def forward(self,X,M,feature_centering=True):
        all_loss=torch.Tensor([0]).to(X.device)
        batchsize,feature,size=X.shape[0],X.shape[1],X.shape[2]
        X,M=X.view(batchsize,feature,-1),M.view(batchsize,feature,-1)
        if feature_centering==True:
            X=X-X.mean(dim=-2,keepdim=True)
            M=M-M.mean(dim=-2,keepdim=True)
            X,M=feature_normalize(X),feature_normalize(M)
            xy=MatmulTopkLoss(X,M)
            xy=torch.exp(-xy+xy.min()-1e-6)
            all_loss=all_loss+(xy).mean()
        return all_loss
class DimixLoss(nn.Module):
    def __init__(self,list_len=1):
        super(DimixLoss,self).__init__()
        self.list_len=list_len
    def forward(self,X,M,feature_centering=True):
        all_loss=torch.Tensor([0]).to(X.device)
        batchsize,feature,size=X.shape[0],X.shape[1],X.shape[2]
        X,M=X.view(batchsize,feature,-1),M.view(batchsize,feature,-1)
        if feature_centering==True:
            X=X-X.mean(dim=-2,keepdim=True)
            M=M-M.mean(dim=-2,keepdim=True)
            X,M=feature_normalize(X),feature_normalize(M)
            xy=MatmulTopkLoss(X,M)
            xy=1.-torch.exp(-xy+xy.min()-1e-6)
            all_loss=all_loss+(xy).mean()
        return all_loss


