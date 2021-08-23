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


def load_data(train_batch_size, test_batch_size, data_url=None):
    if data_url == None:
        data_url = './data'
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

    train_set = torchvision.datasets.CIFAR10(root=data_url, train=True,
                                             download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size,
                                               shuffle=True, num_workers=4, drop_last=False)

    test_set = torchvision.datasets.CIFAR10(root=data_url, train=False,
                                            download=True, transform=test_trainsform)
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
    def __init__(self, max_rate=1, flatten=False, transform=False, training=True):
        repeat = 0
        scale = 0.1
        label_selection = "ScoreValence"
        self.transform = transform
        self.flatten = flatten
        self.max_rate = max_rate
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
        seed_indices=random.randint(0,4)
        self.test_X = self.test_X[seed_indices]
        self.train_X = self.train_X[seed_indices]
        self.test_y = self.test_y[seed_indices]
        self.train_y =self.train_y[seed_indices]

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
        X = np.repeat(np.expand_dims(np.transpose(self.data[idx],(1,0)), axis=-1), repeats=self.data[idx].shape[-1], axis=-1)
        y = self.label[idx]
        if self.transform==True:
            import random
            if random.random() > 0.5:
                X = X.transpose((0, 2, 1))
            if random.random() > 0.5:
                m = random.random()
                if m > 0.25:
                    X = np.concatenate((np.expand_dims(X[:, :, -1],axis=-1), X[:, :, :-1]), axis=-1)
                elif m >= 0.25 and m < 0.5:
                    X = np.concatenate((np.expand_dims(X[:, -1, :],axis=1), X[:, :-1, :]), axis=1)
                elif m >= 0.5 and m < 0.75:
                    X = np.concatenate((X[:, 1:, :], np.expand_dims(X[:, 0, :],axis=1)), axis=1)
                else:
                    X = np.concatenate((X[:, :, 1:], np.expand_dims(X[:, :, 0],axis=-1)), axis=-1)
        X=torch.Tensor(X)
        import torch.nn.functional as F
        with torch.no_grad():
            X=X.unsqueeze(0)
            X=F.interpolate(X,size=[64,64],mode='nearest')
            X=X.squeeze(0)
        y=torch.Tensor([y])
        return X,y.long().squeeze(-1)