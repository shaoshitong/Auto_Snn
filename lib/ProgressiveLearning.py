import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets
import random
from PIL import Image
import math
from torchvision.transforms import *


class ImageNetDataset(torchvision.datasets.ImageFolder):
    def __init__(self,root,transform):
        super(ImageNetDataset, self).__init__(root=root,transform=transform)
        self.nums=10
        self.beta=0.2
        self.trans=self.nums
        self.size_rate=1
    def reset_beta(self,beta,size_rate):
        self.nums=int((1-beta)*10)
        self.trans=beta
        self.size_rate=size_rate
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        label=torch.zeros(1000)
        label[target]=1
        if self.transform is not None:
            transform = Change_Compose_224(self.transform, self.trans, self.size_rate)
            sample = transform(sample)
        if  index > 0 and index % self.nums == 0:
            mixup_idx=random.randint(0,len(self.samples)-1)
            mixup_path, mixup_target = self.samples[mixup_idx]
            mixup_sample = self.loader(mixup_path)
            mixup_label = torch.zeros(1000)
            mixup_label[mixup_target] = 1
            if self.transform is not None:
                transform = Change_Compose_224(self.transform, self.trans, self.size_rate)
                mixup_sample = transform(mixup_sample)
            beta = self.beta
            lam = np.random.beta(beta, beta)
            image = lam * sample + (1 - lam) * mixup_sample
            label = lam * label + (1 - lam) * mixup_label
        return sample, target
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
            transform=Change_Compose_32(self.transform,self.trans,self.size_rate)
            image=transform(image)
        if self.train and idx>0 and idx%self.nums==0:
            mixup_idx=random.randint(0,len(self.data)-1)
            mixup_image, mixup_target = self.data[mixup_idx], self.targets[mixup_idx]
            mixup_image = Image.fromarray(mixup_image)

            mixup_label=torch.zeros(10)
            mixup_label[self.targets[mixup_idx]]=1
            if self.transform:
                transform = Change_Compose_32(self.transform, self.trans,self.size_rate)
                mixup_image=transform(mixup_image)
            beta=self.beta
            lam=np.random.beta(beta,beta)
            image=lam*image+(1-lam)*mixup_image
            label=lam*label+(1-lam)*mixup_label
        return image,label


class CIFAR100Dataset(torchvision.datasets.CIFAR100):
    def __init__(self,root, train,download, transform):
        super(CIFAR100Dataset, self).__init__(root=root,train=train,download=download,transform=transform)
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

        label=torch.zeros(100)
        label[self.targets[idx]]=1
        if self.transform:
            transform=Change_Compose_32(self.transform,self.trans,self.size_rate)
            image=transform(image)
        if self.train and idx>0 and idx%self.nums==0:
            mixup_idx=random.randint(0,len(self.data)-1)
            mixup_image, mixup_target = self.data[mixup_idx], self.targets[mixup_idx]
            mixup_image = Image.fromarray(mixup_image)

            mixup_label=torch.zeros(100)
            mixup_label[self.targets[mixup_idx]]=1
            if self.transform:
                transform = Change_Compose_32(self.transform, self.trans,self.size_rate)
                mixup_image=transform(mixup_image)
            beta=self.beta
            lam=np.random.beta(beta,beta)
            image=lam*image+(1-lam)*mixup_image
            label=lam*label+(1-lam)*mixup_label
        return image,label

def Change_Compose_32(compose:torchvision.transforms.Compose,p,size_rate):
    z=torchvision.transforms.Compose([])
    for i in range(len(compose.transforms)):
        if isinstance(compose.transforms[i], torchvision.transforms.Resize):
            z.transforms.append(torchvision.transforms.Resize(int(32 * size_rate)))
        elif isinstance(compose.transforms[i],AutoAugment):
            z.transforms.append(torchvision.transforms.RandomApply([compose.transforms[i]],p))
        elif isinstance(compose.transforms[i],RandomErasing):
            z.transforms.append(torchvision.transforms.RandomApply([compose.transforms[i]],p))
        elif isinstance(compose.transforms[i],ToTensor) or isinstance(compose.transforms[i],Normalize):
            z.transforms.append(compose.transforms[i])
        else:
            z.transforms.append(compose.transforms[i])
    return z



def Change_Compose_224(compose:torchvision.transforms.Compose,p,size_rate):
    z=torchvision.transforms.Compose([])
    for i in range(len(compose.transforms)):
        if isinstance(compose.transforms[i], torchvision.transforms.Resize):
            z.transforms.append(torchvision.transforms.Resize(int(224 * size_rate)))
        elif isinstance(compose.transforms[i],AutoAugment):
            z.transforms.append(torchvision.transforms.RandomApply([compose.transforms[i]],p))
        elif isinstance(compose.transforms[i],RandomErasing):
            z.transforms.append(torchvision.transforms.RandomApply([compose.transforms[i]],p))
        elif isinstance(compose.transforms[i],ToTensor) or isinstance(compose.transforms[i],Normalize):
            z.transforms.append(compose.transforms[i])
        else:
            z.transforms.append(compose.transforms[i])
    return z




class RandomErasing(torch.nn.Module):
    """ Randomly selects a rectangle region in an image and erases its pixels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        super(RandomErasing, self).__init__()
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def forward(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
