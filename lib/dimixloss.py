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
class Linear_adaptive_loss(nn.Module):
    def __init__(self,channels,size,classes=None):
        super(Linear_adaptive_loss,self).__init__()
        if classes==None:
            classes=10
        self.mlp                    =nn.Sequential(nn.Flatten(),
                                    nn.Linear(channels,channels),
                                    nn.ELU(),
                                    nn.Linear(channels,classes))
        self._initialize()
    def _initialize(self,):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.zeros_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
    def info_nec_loss(self,features,tempature=0.07):
        labels=torch.cat([torch.arange(0,features.shape[0]//2) for i in range(2)],dim=0)
        labels=(labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels=labels.to(features.device)
        features=F.normalize(features,dim=1)
        similarity_matrix=torch.matmul(features,features.T)
        mask=torch.eye(labels.shape[0],dtype=torch.bool).to(features.device)
        labels=labels[~mask].view(labels.shape[0],-1)
        similarity_matrix=similarity_matrix[~mask].view(similarity_matrix.shape[0],-1)
        positives=similarity_matrix[labels.bool()].view(labels.shape[0],-1)
        negatives=similarity_matrix[~labels.bool()].view(labels.shape[0],-1)
        negatives_left,negatives_right=torch.split(negatives,dim=1,split_size_or_sections=[negatives.shape[1]//2,negatives.shape[1]//2])
        negatives_left_top,negatives_left_bottom=torch.split(negatives_left,dim=0,split_size_or_sections=[negatives.shape[0]//2,negatives.shape[0]//2])
        negatives_right_top,negatives_right_bottom=torch.split(negatives_right,dim=0,split_size_or_sections=[negatives.shape[0]//2,negatives.shape[0]//2])

        choose=torch.softmax(torch.cat([torch.sum(negatives_left_top-negatives_right_top,dim=1,keepdim=True),torch.sum(negatives_right_bottom-negatives_left_bottom,dim=1,keepdim=True)],dim=1),dim=1)
        logits=torch.cat([positives,negatives],dim=1)/tempature
        return logits,choose
    def forward(self,x,y):
        x_l=F.avg_pool2d(x,x.shape[-1])
        y_l=F.avg_pool2d(y,y.shape[-1])
        c=torch.cat([x_l,y_l],dim=0)
        c=self.mlp(c)
        logits,choose=self.info_nec_loss(c)
        result=choose[:,0].view(x.shape[0],1,1,1)*x+choose[:,1].view(x.shape[0],1,1,1)*y
        return logits,result

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


