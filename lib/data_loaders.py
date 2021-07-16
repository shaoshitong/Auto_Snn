# -*- coding: utf-8 -*-

"""
MNIST,CIFAR10,CIFAR100
"""

from torch.utils.data import Dataset, DataLoader

import numpy as np
from torchvision import transforms, utils
import torch
import torchvision
import torch.utils.data
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

        img = np.array(self.dataset[idx][0], dtype=np.float32) / 255.0 * self.max_rate
        shape = img.shape
        img_spike = None
        if self.flatten == True:
            img = img.reshape(-1)
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
def load_data(train_batch_size, test_batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_trainsform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                              shuffle=True, num_workers=4,drop_last=False)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_trainsform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                             shuffle=False, num_workers=4,drop_last=False)

    return train_loader, test_loader
