
# -*- coding: utf-8 -*-

"""
MNIST,CIFAR10,CIFAR100
"""
import os

from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms, utils
import torch
import torchvision.datasets as datasets
import torchvision
import torch.utils.data
from lib.dataprocess import CarDateset
from lib.dataprocess import STLdataprocess
from torch.utils.data import DataLoader
from lib.ProgressiveLearning import CIFAR10Dataset,ImageNetDataset,CIFAR100Dataset,RandomErasing
import random
import openpyxl

"""
MNIST
"""


class MNISTDataset(Dataset):
    """mnist dataset

    torchvision_mnist: dataset object
    length: number of steps of snn
    max_rate: a scale factor. MNIST pixel value is normalized to [0,1], and them multiply with this value
    flatten: return 28x28 image or a flattened 1d vector
    transform: transform
    """

    def __init__(self, torchvision_mnist, length, max_rate=1, flatten=False, transform=None):
        self.dataset = torchvision_mnist
        self.transform = transform
        self.flatten = flatten
        self.length = length
        self.max_rate = max_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.dataset[idx][0]
        if self.transform:
            img = self.transform(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.1307,), (0.3081,))(img).float()
        if self.flatten == True:
            img = img.view(-1)
        return img, self.dataset[idx][1]


def get_rand_transform(transform_config):
    t1_size = transform_config['RandomResizedCrop']['size']
    t1_scale = transform_config['RandomResizedCrop']['scale']
    t1_ratio = transform_config['RandomResizedCrop']['ratio']
    t1 = transforms.RandomResizedCrop(t1_size, scale=t1_scale, ratio=t1_ratio, interpolation=2)

    t2_angle = transform_config['RandomRotation']['angle']
    t2 = transforms.RandomRotation(t2_angle, resample=False, expand=False, center=None)
    t3 = transforms.Compose([t1, t2])

    rand_transform = transforms.RandomApply([t1, t2, t3], p=transform_config['RandomApply']['probability'])

    return rand_transform


"""
CIFAR10
"""


def load_data_car(train_batch_size, test_batch_size, shuffle=True, transform=True, tmp_size=96, result_size=64, ):
    data_url = os.getcwd()
    train_data = CarDateset(data_url, tmp_size=tmp_size, result_size=result_size, use_transform=transform,
                            training=True)
    test_data = CarDateset(data_url, tmp_size=tmp_size, result_size=result_size, use_transform=transform,
                           training=False)
    train_loader = DataLoader(train_data, train_batch_size, shuffle=shuffle, num_workers=1, drop_last=True)
    test_loader = DataLoader(test_data, test_batch_size, shuffle=shuffle, num_workers=1, drop_last=True)
    return train_loader, test_loader

def load_data_svhn(train_batch_size, test_batch_size, data_url=None):
    if data_url == None:
        data_url = './data'
    RGB2Gray = transforms.Lambda(lambda x: x.convert('L'))
    train_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_data = torchvision.datasets.SVHN(root=data_url,split="train",download=True,transform=train_transforms)
    test_data = torchvision.datasets.SVHN(root=data_url, split="test", download=True,transform=test_transforms)
    train_loader = DataLoader(train_data, train_batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_data, test_batch_size, shuffle=True, num_workers=4, drop_last=True)
    return train_loader, test_loader

def load_data(train_batch_size, test_batch_size, data_url=None ,use_standard=True):
    if data_url == None:
        data_url = './data'
    def get_statistics():
        train_set=torchvision.datasets.CIFAR10(root=data_url,train=True,download=True,transform=transforms.ToTensor())
        data=torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0,2,3]),data.std(dim=[0,2,3])
    if use_standard==False:
        mean,std=get_statistics()
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if use_standard==True else
        transforms.Normalize(mean, std) ,
    ])

    test_trainsform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if use_standard == True else
        transforms.Normalize(mean, std),
    ])
    train_set = CIFAR10Dataset(root=data_url, train=True,
                                             download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                               shuffle=True, num_workers=4, drop_last=False)

    test_set = torchvision.datasets.CIFAR10(root=data_url, train=False,
                                            download=True, transform=test_trainsform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                              shuffle=False, num_workers=4, drop_last=False)

    return train_loader, test_loader
def load_data_stl(train_batch_size,test_batch_size,data_url=None):
    if data_url == None:
        data_url = './data'
    p=STLdataprocess()
    train_set=p(use_training=True)
    test_set=p(use_training=True)
    train_loader=torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                shuffle=True, num_workers=4, drop_last=False)
    test_loader=torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                shuffle=False, num_workers=4, drop_last=False)

    return train_loader, test_loader
def load_data_imagenet(train_batch_size,test_batch_size,data_url=None):
    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    train=ImageNetDataset(os.path.join(data_url,"ILSVRC2012_img_train/"),train_transform)
    val=datasets.ImageFolder(os.path.join(data_url,"val/"),test_transform)
    train_loader=DataLoader(train,batch_size=train_batch_size,shuffle=True,num_workers=8)
    val_loader=DataLoader(val,batch_size=test_batch_size,shuffle=False,num_workers=4)
    return train_loader,val_loader
def load_data_c100(train_batch_size, test_batch_size, data_url=None ,use_standard=True):
    if data_url == None:
        data_url = './data'
    def get_statistics():
        train_set=torchvision.datasets.CIFAR100(root=data_url,train=True,download=True,transform=transforms.ToTensor())
        data=torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0,2,3]),data.std(dim=[0,2,3])
    if use_standard==False:
        mean,std=get_statistics()
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        RandomErasing(mean=(0.4914, 0.4822, 0.4465) if use_standard==True else mean),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if use_standard==True else
        transforms.Normalize(mean, std) ,
    ])

    test_trainsform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if use_standard == True else
        transforms.Normalize(mean, std),
    ])
    train_set = CIFAR100Dataset(root=data_url,train=True,download=True,transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                               shuffle=True, num_workers=4, drop_last=False)
    test_set = torchvision.datasets.CIFAR100(root=data_url,train=False,download=True,transform=test_trainsform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                              shuffle=False, num_workers=4, drop_last=False)

    return train_loader, test_loader
def revertNoramlImgae(image):
    image: torch.Tensor
    if image.shape[-1] == 1:
        mean1 = torch.Tensor([0.1307]).unsqueeze(0).unsqueeze(0)
        std1 = torch.Tensor([0.3081]).unsqueeze(0).unsqueeze(0)
        image = image * std1 + mean1
        return image
    else:
        mean1 = torch.Tensor([0.4914, 0.4822, 0.4465]).unsqueeze(0).unsqueeze(0)
        std1 = torch.Tensor([0.2023, 0.1994, 0.2010]).unsqueeze(0).unsqueeze(0)
        image = image * std1 + mean1
        return image


class EEGDateset(Dataset):
    def __init__(self, random, flatten=False, transform=False, training=True):
        repeat = 0
        scale = 0.1
        label_selection = "ScoreValence"
        self.transform = transform
        self.flatten = flatten
        self.train_X = []
        self.train_y = []
        self.test_X = []
        self.test_y = []
        for i in [100, 200, 300, 400, 500]:
            self.train_X.append(np.load(file='./data/eeg/Xtrain_re%ds%d_%s_modelseed%d.npy' % (
                repeat, scale * 100, label_selection, i)))
            self.train_y.append(np.load(file='./data/eeg/ytrain_re%ds%d_%s_modelseed%d.npy' % (
                repeat, scale * 100, label_selection, i)))
            self.test_X.append(np.load(file='./data/eeg/Xtest_re%ds%d_%s_modelseed%d.npy' % (
                repeat, scale * 100, label_selection, i)))
            self.test_y.append(np.load(file='./data/eeg/ytest_re%ds%d_%s_modelseed%d.npy' % (
                repeat, scale * 100, label_selection, i)))
        seed_indices = random
        self.test_X = self.test_X[seed_indices]
        self.train_X = self.train_X[seed_indices]
        self.test_y = self.test_y[seed_indices]
        self.train_y = self.train_y[seed_indices]

        if self.flatten == True:
            len = self.train_X.shape[0]
            arr = np.arange(0, len)
            np.random.shuffle(arr)
            self.train_X = self.train_X[arr]
            self.train_y = self.train_y[arr]
        if training == True:
            self.data = self.train_X
            self.label = self.train_y
        else:
            self.data = self.test_X
            self.label = self.test_y

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        X = np.reshape(np.repeat(np.expand_dims(np.transpose(self.data[idx], (1, 0)), axis=-1), repeats=16, axis=-1),
                       ([14, 32, 32]))
        y = self.label[idx]
        if self.transform == True:
            import random
            # if random.random() > 0.5:
            #     X = X.transpose((0, 2, 1))
            if random.random() > 0.5:
                m = random.random()
                if m > 0.25:
                    X = np.concatenate((np.expand_dims(X[:, :, -1], axis=-1), X[:, :, :-1]), axis=-1)
                elif m >= 0.25 and m < 0.5:
                    X = np.concatenate((np.expand_dims(X[:, -1, :], axis=1), X[:, :-1, :]), axis=1)
                elif m >= 0.5 and m < 0.75:
                    X = np.concatenate((X[:, 1:, :], np.expand_dims(X[:, 0, :], axis=1)), axis=1)
                else:
                    X = np.concatenate((X[:, :, 1:], np.expand_dims(X[:, :, 0], axis=-1)), axis=-1)
        X = torch.Tensor(X)
        import torch.nn.functional as F
        with torch.no_grad():
            X = X.unsqueeze(0)
            X = F.interpolate(X, size=[32, 32], mode='nearest')
            X = X.squeeze(0)
        y = torch.Tensor([y])
        return X, y.long().squeeze(-1)
