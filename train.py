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
from Snn_Auto_master.lib.parameters_check import parametersCheck, linearSubUpdate, parametersNameCheck, \
    parametersgradCheck, pd_save
from Snn_Auto_master.lib.data_loaders import MNISTDataset, get_rand_transform, load_data
from Snn_Auto_master.lib.three_dsnn import merge_layer
from Snn_Auto_master.lib.optimizer import get_optimizer
from Snn_Auto_master.lib.scheduler import get_scheduler
from Snn_Auto_master.lib.criterion import criterion
from Snn_Auto_master.lib.accuracy import accuracy
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms, utils
import omegaconf
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description='SNN AUTO MASTER')
parser.add_argument('--config_file', type=str, default='train.yaml',
                    help='path to configuration file')
parser.add_argument('--train', dest='train', default=True, type=bool,
                    help='train model')
parser.add_argument('--test', dest='test', default=True, type=bool,
                    help='test model')
args = parser.parse_args()


class Avgupdate(object):
    def __init__(self):
        self.count = 0.
        self.sum = 0.
        self.avg = 0.

    def reset(self):
        self.count = 0.
        self.sum = 0.
        self.avg = 0.

    def update(self, val, n=1):
        self.sum += val
        self.avg = (self.sum) / (self.count + n)
        self.count += n


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def yaml_config_get(args):
    if args.config_file is None:
        print('No config file provided, use default config file')
    else:
        print('Config file provided:', args.config_file)
    conf = OmegaConf.load(args.config_file)
    return conf


def set_random_seed(conf):
    torch.manual_seed(conf['pytorch_seed'])


def test(path, data, yaml, criterion_loss):
    torch.cuda.empty_cache()
    the_model = merge_layer(set_device(), shape=yaml['shape'], dropout=yaml['parameters']['droupout'], test=True)
    if yaml['data'] == 'mnist':
        the_model.initiate_data(torch.randn(yaml['parameters']['batch_size'], 28 * 28), 0, yaml['parameters']['epoch'],
                                option=False)
        the_model.initiate_layer(torch.randn(yaml['parameters']['batch_size'], 28 * 28), int(10))
    elif yaml['data'] == 'cifar10':
        the_model.initiate_data(torch.randn(yaml['parameters']['batch_size'], 32 * 32 * 3), 0,
                                yaml['parameters']['epoch'],
                                option=False)
        the_model.initiate_layer(torch.randn(yaml['parameters']['batch_size'], 32 * 32 * 3), int(10))
    the_model.load_state_dict(torch.load(path)['snn_state_dict'])
    device = set_device()
    the_model.to(device)
    all_loss = Avgupdate()
    all_prec1 = Avgupdate()
    all_prec5 = Avgupdate()
    it_loss = Avgupdate()
    it_prec1 = Avgupdate()
    it_prec5 = Avgupdate()
    the_model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(data):
            batch_time_stamp = time.strftime("%Y%m%d-%H%M%S")
            if yaml['data']=='mnist':
                input = input.float().to(device)
            elif yaml['data']=='cifar10':
                input = input.float().to(device).view(input.shape[0],-1)
            else:
                raise KeyError()
            target = target.to(device)
            input.requires_grad_()
            the_model.initiate_data(input, 0, yaml['parameters']['epoch'], option=False)
            torch.cuda.synchronize()
            output = the_model(input)
            torch.cuda.synchronize()
            loss = criterion(criterion_loss, output, target)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            all_loss.update(loss.item())
            all_prec1.update(prec1.item())
            all_prec5.update(prec5.item())
            it_loss.update(loss.item())
            it_prec1.update(prec1.item())
            it_prec5.update(prec5.item())
            if i % 100 == 0 and i != 0:
                print(
                    "batch_size_num:{} top1:{:.3f}% top5:{:.3f}% loss:{:.3f} time:{}".format(i, it_prec1.avg
                                                                                             , it_prec5.avg,
                                                                                             it_loss.avg,
                                                                                             batch_time_stamp))
                it_loss.reset()
                it_prec1.reset()
                it_prec5.reset()
        print("====================================>  ths test: top1:{:.3f}% top5:{:.3f}% loss:{:.3f}".format(
            all_prec1.avg,
            all_prec5.avg,
            all_loss.avg))


def train(model, optimizer, scheduler, data, yaml, epoch, criterion_loss, path="./output"):
    device = set_device()
    model.to(device)
    all_loss = Avgupdate()
    all_prec1 = Avgupdate()
    all_prec5 = Avgupdate()
    it_loss = Avgupdate()
    it_prec1 = Avgupdate()
    it_prec5 = Avgupdate()
    for i, (input, target) in enumerate(data):
        batch_time_stamp = time.strftime("%Y%m%d-%H%M%S")
        if yaml['data'] == 'mnist':
            input = input.float().to(device)
        elif yaml['data'] == 'cifar10':
            input = input.float().to(device).view(input.shape[0], -1)
        else:
            raise KeyError()
        target = target.to(device)
        input.requires_grad_()
        model.initiate_data(input, epoch, yaml['parameters']['epoch'], option=False,real_img=True)
        output = model(input)
        loss = criterion(criterion_loss, output, target)
        model.zero_grad()
        loss.backward()
        # linearSubUpdate(model)
        # parametersgradCheck(model)
        model.subWeightGrad(epoch, yaml['parameters']['epoch'], 1.)
        # parametersgradCheck(model)
        # pd_save(model.three_dim_layer.point_layerg+_module[str(0) + '_' + str(0) + '_' + str(0)].tensor_tau_m1.view(28,-1),"tau_m2/"+str(i))
        optimizer.step()
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        all_loss.update(loss.item())
        all_prec1.update(prec1.item())
        all_prec5.update(prec5.item())
        it_loss.update(loss.item())
        it_prec1.update(prec1.item())
        it_prec5.update(prec5.item())
        if i % 100 == 0 and i != 0:
            # parametersgradCheck(model)
            print(
                "batch_size_num:{} top1:{:.3f}% top5:{:.3f}% loss:{:.3f} time:{}".format(i, it_prec1.avg
                                                                                         , it_prec5.avg
                                                                                         , it_loss.avg
                                                                                         , batch_time_stamp))
            it_loss.reset()
            it_prec1.reset()
            it_prec5.reset()
        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()
    print("====================================>  ths epoch {}: top1:{:.3f}% top5:{:.3f}% loss:{:.3f}".format(epoch,
                                                                                                              all_prec1.avg,
                                                                                                              all_prec5.avg,
                                                                                                              all_loss.avg))
    if isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR):
        scheduler.step()
    return all_prec1.avg, all_loss.avg


if __name__ == "__main__":
    torch.cuda.empty_cache()
    yaml = yaml_config_get(args)
    # set_random_seed(yaml)
    model = merge_layer(set_device(), shape=yaml['shape'], dropout=yaml['parameters']['droupout'])
    writer = SummaryWriter()
    rand_transform = get_rand_transform(yaml['transform'])
    if yaml['data'] == 'mnist':
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=rand_transform)
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
        train_data = MNISTDataset(mnist_trainset, max_rate=1, length=yaml['parameters']['length'], flatten=True)
        train_dataloader = DataLoader(train_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                      drop_last=True)
        test_data = MNISTDataset(mnist_testset, max_rate=1, length=yaml['parameters']['length'], flatten=True)
        test_dataloader = DataLoader(test_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                     drop_last=True)
        model.initiate_data(torch.randn(yaml['parameters']['batch_size'], 28 * 28), 0, yaml['parameters']['epoch'],
                            option=False)
        model.initiate_layer(torch.randn(yaml['parameters']['batch_size'], 28 * 28), int(10))
    elif yaml['data'] == 'cifar10':
        train_dataloader, test_dataloader = load_data(yaml['parameters']['batch_size'],
                                                      yaml['parameters']['batch_size'])
        model.initiate_data(torch.randn(yaml['parameters']['batch_size'], 32 * 32 * 3), 0, yaml['parameters']['epoch'],
                            option=False)
        model.initiate_layer(torch.randn(yaml['parameters']['batch_size'], 32 * 32 * 3), int(10))
    else:
        raise KeyError('There is no corresponding dataset')
    params = list(model.parameters())
    params2 = filter(lambda i: i.requires_grad, model.parameters())
    optimizer = get_optimizer(params2, yaml, model)
    scheduler = get_scheduler(optimizer, yaml)
    criterion_loss = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion_loss = criterion_loss.cuda()
    if args.train == True:
        best_acc = .0
        for j in range(yaml['parameters']['epoch']):
            model.train()
            epoch_time_stamp = time.strftime("%Y%m%d-%H%M%S")
            prec1, loss = train(model, optimizer, scheduler, train_dataloader, yaml, j, criterion_loss)
            checkpoint_path = os.path.join(yaml['output'], str(j) + '_' + epoch_time_stamp + str(best_acc))
            if best_acc < prec1:
                best_acc = prec1
                torch.save({
                    'epoch': j,
                    'snn_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path + 'best')
                path = checkpoint_path + 'best'
            else:
                torch.save({
                    'epoch': j,
                    'snn_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)
                path = checkpoint_path
            if args.test == True:
                test(path, test_dataloader, yaml, criterion_loss)
        print("best_acc:{:.3f}%".format(best_acc))
