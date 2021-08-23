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
    Y_p_mat_X=torch.matmul(Y_permute,X)
    YX=Y_p_mat_X.permute(0,2,1)+Y_p_mat_X
    YX=F.softmax(YX,dim=-1)
    value,key=torch.topk(YX,YX.size()[-1]//2,dim=-1)# batchsize size size//2
    index=torch.arange(0,value.size()[1]).unsqueeze(0).unsqueeze(-1).to(key.device)# 1,size,1
    C_xy=value*(key-index)/YX.size()[-1]
    C_xy=C_xy.mean(-1) #batchsize,size
    return C_xy



class DimIxLoss(nn.Module):
    def __init__(self,list_len=1):
        super(DimIxLoss,self).__init__()
        self.list_len=list_len
    def forward(self,mult_list,feature_centering=True):
        all_loss=torch.Tensor([0]).to(mult_list[0][0].device)
        for list in mult_list:
            x,y,z=list[0],list[1],list[2]
            batchsize,feature,size=x.shape[0],x.shape[1],x.shape[2]
            x,y,z=x.view(batchsize,feature,-1),y.view(batchsize,feature,-1),z.view(batchsize,feature,-1)
            if feature_centering==True:
                x=x-x.mean(dim=-1).unsqueeze(dim=-1)
                y=y-y.mean(dim=-1).unsqueeze(dim=-1)
                z=z-z.mean(dim=-1).unsqueeze(dim=-1)
                x,y,z=feature_normalize(x),feature_normalize(y),feature_normalize(z)
                xy=MatmulTopkLoss(x,y).mean()
                yz=MatmulTopkLoss(y,z).mean()
                zx=MatmulTopkLoss(z,x).mean()
                xy=torch.exp(-xy+xy.min()-1e-1)
                yz=torch.exp(-yz+yz.min()-1e-1)
                zx=torch.exp(-zx+zx.min()-1e-1)
                all_loss+=(xy+yz+zx)
        return all_loss



