import torch
import torch.nn as nn
import torch.nn.functional as F
def Regularization(model,stringName=None):
    l1_regularization=torch.tensor([0],dtype=torch.float32).cuda()
    l2_regularization=torch.tensor([0],dtype=torch.float32).cuda()
    for name,param in model.named_parameters():
        if stringName is None:
            if 'linear.weight' in name:
                l1_regularization += torch.norm(param, 1)
                l2_regularization += torch.norm(param, 2)
        else:
            if stringName in name or 'linear.weight' in name:
                l1_regularization += torch.norm(param, 1)
                l2_regularization += torch.norm(param, 2)
    return l1_regularization,l2_regularization