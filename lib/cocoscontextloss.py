"""
this .py file use to calcuate the feature difference
"""
import torch
import torch.nn as nn
class reverse(torch.autograd.Function):
    """
    该层是为了脉冲激活分形设计，在原版模型使用，当前模型撤销了
    """

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

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


class ContextualLoss_binary(nn.Module):
    def __init__(self,opt):
        super(ContextualLoss_binary,self).__init__()
        self.opt=opt
    def forward(self,X_features,Y_features,h=5.,feature_centering=True):
        assert X_features.ndim==4 and Y_features.ndim==4
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        if feature_centering==True:
            X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(
                dim=-1).unsqueeze(dim=-1)
            Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(
                dim=-1).unsqueeze(dim=-1)
        X_features = feature_normalize(X_features).view(batch_size, feature_depth,
                                                        -1)  # batch_size * feature_depth * feature_size * feature_size
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth,
                                                        -1)  # batch_size * feature_depth * feature_
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        # d_norm = d
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-3)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)
        loss = -torch.log(CX)
        """
        首先将两个特征映射到同一个域s,然后求在域s中一个特征相较于另一个特征在某一维最具相关性的点，该点占比越大越好
        """

        # contextual loss per batch
        # loss = torch.mean(loss)
        return loss

class ContextualLoss_forward(nn.Module):
    '''
        input is Al, Bl, channel=64,size is [batchsize,channel,16,16]
    '''

    def __init__(self, opt):
        super(ContextualLoss_forward, self).__init__()
        self.opt = opt
        self.contextualloss_binary1=ContextualLoss_binary(self.opt)
        self.contextualloss_binary2=ContextualLoss_binary(self.opt)
        self.contextualloss_binary3=ContextualLoss_binary(self.opt)


    def forward(self, X_features, Y_features, Z_features, h=5., feature_centering=True):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        if X_features.ndim != 3 or Y_features.ndim != 3 or Z_features.ndim != 3:
            self.opt = False
        loss1=self.contextualloss_binary1(X_features,Y_features)
        loss2=self.contextualloss_binary2(Y_features,Z_features)
        loss3=self.contextualloss_binary3(Z_features,X_features)
        return torch.sigmoid(-(loss1+loss2+loss3).mean())*10.
