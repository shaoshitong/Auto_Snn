import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets
import random
from PIL import Image
from torchvision.transforms import *
from torch.utils.data import DataLoader,Dataset
# def criterion(batch_x, batch_y, alpha=1.0, use_cuda=True):
#     '''
#     batch_x：批样本数，shape=[batch_size,channels,width,height]
#     batch_y：批样本标签，shape=[batch_size]
#     alpha：生成lam的beta分布参数，一般取0.5效果较好
#     use_cuda：是否使用cuda
#
#     returns：
#     	mixed inputs, pairs of targets, and lam
#     '''
#     if alpha > 0:
#         lam = np.random.beta(2, alpha)
#     else:
#         lam = 1
#     batch_size = batch_x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#     else:
#         index = torch.randperm(batch_size) #生成打乱的batch_size索引
# 	mixed_batchx = lam * batch_x + (1 - lam) * batch_x[index, :]

class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self,root, train,download, transform):
        super(CIFAR10Dataset, self).__init__(root=root,train=train,download=download,transform=transform)
        self.nums=10
        self.beta=0.2
        self.trans=self.nums
        self.size_rate=1
    def reset_beta(self,beta,size_rate):
        self.nums=int((1-beta)*10)
        self.trans=beta
        self.size_rate=size_rate
    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        label=torch.zeros(10)
        label[self.targets[idx]]=1
        if self.transform:
            transform=Change_Compose(self.transform,self.trans,self.size_rate)
            image=transform(image)
        if self.train and idx>0 and idx%self.nums==0:
            mixup_idx=random.randint(0,len(self.data)-1)
            mixup_image, mixup_target = self.data[mixup_idx], self.targets[mixup_idx]
            mixup_image = Image.fromarray(mixup_image)

            mixup_label=torch.zeros(10)
            mixup_label[self.targets[mixup_idx]]=1
            if self.transform:
                transform = Change_Compose(self.transform, self.trans,self.size_rate)
                mixup_image=transform(mixup_image)
            beta=self.beta
            lam=np.random.beta(beta,beta)
            image=lam*image+(1-lam)*mixup_image
            label=lam*label+(1-lam)*mixup_label
        return image,label

def Change_Compose(compose:torchvision.transforms.Compose,p,size_rate):
    z=torchvision.transforms.Compose([])
    for i in range(len(compose.transforms)):
        if isinstance(compose.transforms[i], torchvision.transforms.Resize):
            z.transforms.append(torchvision.transforms.Resize(int(32 * size_rate)))
        elif isinstance(compose.transforms[i],AutoAugment):
            z.transforms.append(torchvision.transforms.RandomApply([compose.transforms[i]],p))
        elif isinstance(compose.transforms[i],ToTensor) or isinstance(compose.transforms[i],Normalize):
            z.transforms.append(compose.transforms[i])
        else:
            z.transforms.append(compose.transforms[i])
    return z








