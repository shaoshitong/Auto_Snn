# -*- coding: utf-8 -*-

"""
# File Name : snn_mlp_1.py
# Author: Haowen Fang
# Email: hfang02@syr.edu
# Description: multi-layer snn for MNIST classification. Use dual exponential psp kernel.
"""

import argparse
import pandas as pd
import os
import time
import sys

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms, utils
import omegaconf
from omegaconf import OmegaConf
import torch.nn as nn


class random_layer():
    mode: str
    shape: tuple or int

    def __init__(self, shape, mode=None, seed=None):
        """
        mode="Bernoulli", "Binomial", "Geometric", "Poisson", "Normal", "Power law"
        shape=tuple or int
        seed=int
        """
        self.modelist = ["Bernoulli", "Binomial", "Geometric", "Poisson", "Normal", "Power law"]
        if isinstance(shape, int):
            self.shape = (shape,)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            print(TypeError("the shape type is false"))
            exit()
        if seed is None:
            self.seed = np.random.randint(1, high=int(1e6))
        else:
            self.seed = seed
        self.result = torch.zeros(self.shape, dtype=torch.float32)
        if mode is None:
            self.mode = "Bernoulli"
        else:
            if mode not in self.modelist:
                print(KeyError("Not have this key"))
                exit()
            else:
                self.mode = mode

    def return_Distribution(self, *args):
        self.args = args
        torch.manual_seed(self.seed)
        if self.mode == self.modelist[0]:
            """
            对于伯努利分布而言主要是概率p选择1,1-p选择0
            """
            arglist = list(self.args)
            if len(arglist) == 1:
                arglist = arglist[0]
            if isinstance(arglist, float):
                par_perm = torch.full(self.shape, arglist)
            elif torch.is_tensor(arglist):
                for i in range(len(self.shape)):
                    if arglist.shape[i] != self.shape[i]:
                        exit("Error:size error")
                par_perm = arglist
            else:
                arglist = torch.Tensor(*self.shape).uniform_(0, 1)
                par_perm = arglist
            return torch.bernoulli(par_perm).float()
        elif self.mode == self.modelist[1]:
            args = list(self.args)
            """
            包括二项分布的概率=>p，总次数=>n，单个次数=>m
            """
            if len(args) == 2:
                p, n = args
            else:
                p, n = .5, 2
            print(p, n)
            if isinstance(p, float):
                p = torch.full(self.shape, p)
            s = torch.rand(self.shape)
            result = torch.pow(1 - p, n)
            result2 = torch.ge(result, s).float()
            print(result2)
            for i in range(n):
                result = result + result * p * (n - i) / ((i + 1) * (1 - p))
                result2 = torch.where(torch.eq(result2, 0), torch.ge(result, s).float() * (i + 2), result2)
            result2 -= 1
            return result2.float()
        elif self.mode == self.modelist[2]:
            """
            几何分布:一个人做一件事，第x次成功的概率
            """
            args = list(self.args)
            if len(args) == 2:
                p, n = args
            else:
                p, n = .5, 2
            if isinstance(p, float):
                p = torch.full(self.shape, p)
            result = torch.zeros(self.shape, dtype=torch.float32)
            s = torch.rand(self.shape)
            result2 = torch.zeros(self.shape, dtype=torch.float32)
            for i in range(n):
                result += torch.pow(1 - p, i) * p
                eqtmp = torch.eq(result2, .0)
                getmp = torch.ge(result, s) * (i + 1)
                result2 = torch.where(eqtmp, getmp.float(), result2)
            return result2.float()
        elif self.mode == self.modelist[3]:
            args = list(self.args)
            if len(args) != 1:
                p = torch.rand(self.shape)
            else:
                p = args[0]
            if isinstance(p, float):
                p = torch.full(self.shape, p)
            return torch.poisson(p).float()
            pass
        elif self.mode == self.modelist[4]:
            args = list(self.args)
            if len(args) == 2:
                a, s = args
            else:
                a, s = 0., 1.
            return torch.normal(a, s, self.shape).float()
        elif self.mode == self.modelist[5]:
            """
            幂律分布是指分布服从y=x^(-a-1)
            """
            args = list(self.args)
            if len(args) == 2:
                p, n = args
            else:
                p = torch.full(self.shape, 0)
                n = 100
            s = torch.rand(self.shape)
            result = torch.zeros(self.shape, dtype=torch.float32)
            result2 = torch.zeros(self.shape, dtype=torch.float32)
            for i in range(2, n + 1):
                result += torch.pow(torch.full(self.shape, i).float(), -p - 1)
                getmp = torch.ge(result, s).float() * (i - 1)
                eqtmp = torch.eq(result2, 0.)
                result2 = torch.where(eqtmp, getmp, result2)
            return (result2).float()
        else:
            pass


class Choose_layer(nn.Module):
    def __init__(self, symbol, shape=None, probability=None, sigma=2.):
        super(Choose_layer, self).__init__()
        if shape == None and probability == None:
            print("Error:the parameter's number too little")
            pass
        elif shape != None and probability == None:
            self.p = torch.randn(*shape)
        elif shape == None and probability != None:
            if torch.is_tensor(probability):
                self.p = probability
            else:
                self.p = torch.from_numpy(probability)
        else:
            if tuple(*shape) == tuple(*probability.shape):
                if torch.is_tensor(probability):
                    self.p = probability
                else:
                    self.p = torch.from_numpy(probability)
            else:
                print("Error:Input parameter conflict")
        self.symbollist = ["Bernoulli", "Binomial", "Geometric", "Poisson", "Normal", "Power law"]
        if symbol not in self.symbollist:
            print("Error:symbol error")
            pass
        self.symbol = symbol
        self.softmax = nn.Softmax(dim=1)
        self.sigma = sigma

    def forward(self, x):
        x1, x2, x3 = x
        m = x1.shape
        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        Random_layer = random_layer(m[0], mode=self.symbol)
        p = self.softmax(self.p.view(x1.shape[0], -1))
        mean = torch.mean(p, dim=1, keepdim=True)
        var = torch.var(p, dim=1, keepdim=True)
        sigma = self.sigma
        p = torch.where(torch.ge(p, (mean + torch.pow(var, .5) / sigma)), 1., 0.).float()
        """
        对特征进行筛选
        """
        if self.symbol == "Bernoulli":
            q = Random_layer.return_Distribution(.5) % 3
        elif self.symbol == "Binomial":
            q = Random_layer.return_Distribution(.5, 3) % 3
        elif self.symbol == "Geometric":
            q = (Random_layer.return_Distribution(.5, 3) - 1) % 3
        elif self.symbol == "Poisson":
            q = Random_layer.return_Distribution(.5, 3) % 3
        elif self.symbol == "Normal":
            q = torch.max(torch.min(torch.ceil(Random_layer.return_Distribution(0., 1.)).float(), torch.Tensor([2.])),
                          torch.Tensor([0.]))
        elif self.symbol == "Power law":
            q = Random_layer.return_Distribution(2, 3) % 3
        """
        对来自三维空间的三组数据进行筛选
        """
        print("sample======>", q)

        result = []
        for i in range(q.shape[0]):
            if q[i].item() == 0:
                result.append(x1[i])
            elif q[i].item() == 1:
                result.append(x2[i])
            elif q[i].item() == 2:
                result.append(x3[i])
        result = torch.stack(result, dim=0)
        return result * p


# a = random_layer((3, 3), mode="Power law")
# print(a.return_Distribution(torch.randint(0,2,(3,3)),100))
choose = Choose_layer("Power law", probability=torch.rand(3, 4))
"""
["Bernoulli", "Binomial", "Geometric", "Poisson", "Normal", "Power law"]
"""
x1 = torch.randn(3, 4)
x2 = torch.randn(3, 4)
x3 = torch.randn(3, 4)
print(x1, x2, x3)
print("=" * 120)
print(choose.forward((x1, x2, x3)))
