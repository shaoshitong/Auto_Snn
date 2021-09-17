import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import torch.utils.data as data
import PIL.Image as Image
import torchvision.transforms as transforms


def shuffle(len, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(0, len)
    np.random.shuffle(indices)
    return indices


class STLdataset(data.Dataset):
    def __init__(self, data, use_training=True, size=96, use_shuffle=True):
        self.data = data
        self.use_training = use_training
        self.transform = transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30, center=(size // 2, size // 2)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4455), (0.2023, 0.1994, 0.2010))
        ]) if use_training == True else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4455), (0.2023, 0.1994, 0.2010))
        ])
        if use_shuffle == True:
            rdnindex = shuffle(self.data[0].shape[0])
            self.data = (self.data[0][rdnindex, ...], self.data[1][rdnindex, ...])

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, index):
        data, label = self.data[0][index,...], torch.from_numpy(self.data[1][index, :]).squeeze().item()-1
        data = self.transform(Image.fromarray(data.transpose(1,2,0)))
        return data, label

class STLdataprocess(object):
    def __init__(self, path="/home/sst/product/Snn_Auto_master/data/stl/stl10_matlab"):
        """
        shape : 5000,3,96,96 5000,1
        """
        self.train_path = os.path.join(path, "train.mat")
        self.test_path = os.path.join(path, "test.mat")
        self.train_data = scio.loadmat(self.train_path)
        self.test_data = scio.loadmat(self.test_path)
        self.train_data = (self.train_data["X"], self.train_data["y"])
        self.test_data = (self.test_data["X"], self.test_data["y"])
        self.train_data = (self.train_data[0].reshape(self.train_data[0].shape[0], 3, 96, 96), self.train_data[1])
        self.test_data = (self.test_data[0].reshape(self.test_data[0].shape[0], 3, 96, 96), self.test_data[1])

    def __call__(self, use_training=True):
        if use_training == True:
            dataset = STLdataset(self.train_data, use_training=use_training)
        else:
            dataset = STLdataset(self.test_data, use_training=use_training)
        return dataset


