# -*- coding: utf-8 -*-
"""
# train.py
# mnist,cifar10,fashionmnist
"""
import argparse
import os
import random
import time
import sys
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
sys.path.append("D:\Product")
sys.path.append("D:\Product\Snn_Auto_master")
sys.path.append("F:\Snn_Auto")
sys.path.append("F:\sst")
sys.path.append("/home/sst/product")
from lib.accuracy import *
from lib.criterion import *
from lib.data_loaders import *
from lib.log import *
from lib.optimizer import *
from lib.scheduler import *
from lib.three_dsnn import *
from lib.loss_utils import *
from lib.config import *

from lib.parameters_check import parametersgradCheck, parametersNameCheck

parser = argparse.ArgumentParser(description='SNN AUTO MASTER')
parser.add_argument('--config_file', type=str, default='./config/train_imagenet.yaml',
                    help='path to configuration file')
parser.add_argument('--train', dest='train', default=True, type=bool,
                    help='train model')
parser.add_argument('--test', dest='test', default=True, type=bool,
                    help='test model')
parser.add_argument('--data_url', dest='data_url', default='D:\\Product\\up-detr\\data', type=str,
                    help='test model')
parser.add_argument('--neg_mul', dest='neg_mul', default=0.1, type=float,
                    help='neg_learning')
parser.add_argument('--log_each', dest='log_each', default=100, type=int,
                    help='how many step log once')
args = parser.parse_args()
log = Log(log_each=args.log_each)
scaler = torch.cuda.amp.GradScaler()
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


device = set_device()


def neg_smooth_cross(pred, gold, output,smoothing=0.1, *args, **kwargs):
    n_class = pred.size(1)
    gold = gold.to(pred.device)
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1)).to(pred.device)  # 0.0111111
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1. - smoothing)  # 0.9
    output=torch.softmax(output,dim=1)
    out_one_hot=output.argmax(dim=1)
    with torch.no_grad():
        one_hot=torch.where(torch.eq(out_one_hot.unsqueeze(-1),gold.unsqueeze(-1)),torch.softmax(pred,dim=1),one_hot)
    log_prob = torch.nn.functional.log_softmax(pred, dim=1)
    return torch.nn.functional.kl_div(input=log_prob, target=one_hot, reduction='none').sum(dim=-1).mean()


def get_params_numeric(model):
    sum = 0
    set_id = []
    for parameter in model.named_parameters():
        if id(parameter[1]) not in set_id:
            sum += parameter[1].numel()
            set_id.append(id(parameter[1]))
            # print(f"{parameter[0]},{parameter[1].numel()}")
    print(f"{round(sum / 1e6, 6)}M")


def yaml_config_get(args):
    if args.config_file is None:
        print('No config file provided, use default config file')
    else:
        print('Config file provided:', args.config_file)
    conf = OmegaConf.load(args.config_file)
    return conf


def set_random_seed(conf):
    torch.manual_seed(conf['pytorch_seed'])
    torch.cuda.manual_seed(conf['pytorch_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    import numpy as np
    import random
    np.random.seed(conf['pytorch_seed'])
    random.seed(conf['pytorch_seed'])


def test2(model, data, yaml, criterion_loss):
    log.eval(len_dataset=len(data))
    the_model = model
    the_model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(data):
            batch_time_stamp = time.strftime("%Y%m%d-%H%M%S")
            if yaml['data'] in ['mnist', 'fashionmnist']:
                input = input.float().to(device)
            elif yaml['data'] in ['cifar10', 'cifar100', 'svhn', 'eeg', 'car', 'stl-10']:
                input = input.float().to(device)
            elif yaml['data'] in ["imagenet"]:
                input=input.to(device)
            else:
                raise KeyError('not have this dataset')
            target = target.to(device)
            input.requires_grad_()
            torch.cuda.synchronize()
            output= model(input)
            #z=args.neg_mul * F.mse_loss(potg.squeeze(1), target.float(), reduction="mean")
            loss_list = [criterion(criterion_loss, output, target)]
            loss = model.L2_biasoption(loss_list, yaml["parameters"]['sigma_list'])
            torch.cuda.synchronize()
            if yaml['data'] == 'eeg':
                prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            elif yaml['data'] in ['mnist', 'fashionmnist', 'cifar10', 'car', 'svhn', 'cifar100', 'stl-10',"imagenet"]:
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            log(model, loss.cpu(), prec1.cpu(), prec5.cpu())
    return log.epoch_state["top_1"] / log.epoch_state["steps"], log.epoch_state["loss"] / log.epoch_state["steps"]


def test(path, data, yaml, criterion_loss):
    log.eval(len_dataset=len(data))
    torch.cuda.empty_cache()
    the_model = merge_layer(set_device(), shape=yaml['shape'], dropout=yaml['parameters']['dropout'])
    if yaml['data'] == 'mnist':
        the_model.initiate_layer(dataoption='mnist',
                                 data=torch.randn(yaml['parameters']['batch_size'], 1, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'cifar10':
        the_model.initiate_layer(dataoption='cifar10',
                                 data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'imagenet':
        the_model.initiate_layer(dataoption='imagenet',
                                 data=torch.randn(yaml['parameters']['batch_size'], 3,224, 224),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'cifar100':
        the_model.initiate_layer(dataoption='cifar100',
                                 data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'fashionmnist':
        the_model.initiate_layer(dataoption='fashionmnist',
                                 data=torch.randn(yaml['parameters']['batch_size'], 1, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'eeg':
        the_model.initiate_layer(dataoption='eeg',
                                 data=torch.randn(yaml['parameters']['batch_size'], 14, 64, 64),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'car':
        the_model.initiate_layer(dataoption='car',
                                 data=torch.randn(yaml['parameters']['batch_size'], 3, 64, 64),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'svhn':
        the_model.initiate_layer(dataoption='svhn',
                                 data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'stl-10':
        the_model.initiate_layer(dataoption='stl-10',
                                 data=torch.randn(yaml['parameters']['batch_size'], 3, 96, 96),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    else:
        raise KeyError('not have this dataset')
    the_model.load_state_dict(torch.load(path)['snn_state_dict'])
    device = set_device()
    the_model.to(device)
    the_model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(data):
            batch_time_stamp = time.strftime("%Y%m%d-%H%M%S")
            if yaml['data'] in ['mnist', 'fashionmnist']:
                input = input.float().to(device)
            elif yaml['data'] in ['cifar10', 'cifar100', 'svhn', 'eeg', 'car', 'stl-10']:
                input = input.float().to(device).view(input.shape[0], -1)
            elif yaml['data'] in ["imagenet"]:
                input=input.float().to(device)
            else:
                raise KeyError('not have this dataset')
            target = target.to(device)
            input.requires_grad_()
            torch.cuda.synchronize()
            output= model(input)
            loss_list = [criterion(criterion_loss, output, target)]
            loss = model.L2_biasoption(loss_list, yaml["parameters"]['sigma_list'])
            torch.cuda.synchronize()
            if yaml['data'] == 'eeg':
                prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            elif yaml['data'] in ['mnist', 'fashionmnist', 'cifar10', 'car', 'svhn', 'cifar100', 'stl-10',"imagenet"]:
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            log(model, loss.cpu(), prec1.cpu(), prec5.cpu())
    return log.epoch_state["top_1"] / log.epoch_state["steps"], log.epoch_state["loss"] / log.epoch_state["steps"]


def train(model, optimizer, scheduler, data, yaml, epoch, criterion_loss, path="./output"):
    log.train(len_dataset=len(data))
    sigma = yaml['parameters']['sigma']
    if yaml['data'] == "stl-10":
        res = 0.
    for i, (input, target) in enumerate(data):
        iter=epoch*(len(data))+i
        if yaml['data'] in ['mnist', 'fashionmnist']:
            input = input.float().to(device)
        elif yaml['data'] in ['cifar10', 'cifar100', 'svhn', 'eeg', 'car', 'stl-10']:
            input = input.float().to(device)
        elif yaml['data'] in ["imagenet"]:
            input = input.to(device)
        else:
            raise KeyError('not have this dataset')
        target = target.to(device)
        input.requires_grad_()
        with torch.cuda.amp.autocast():
            output= model(input)
            # z = args.neg_mul * F.mse_loss(potg.squeeze(1), target.float(), reduction="mean")
            loss_list = [criterion(criterion_loss, output, target)]
            loss = model.L2_biasoption(loss_list,yaml["parameters"]['sigma_list'])
        if yaml['optimizer']['optimizer_choice'] == 'SAM':
            loss.backward(retain_graph=False)
            optimizer.first_step(zero_grad=True)
            output= model(input)
            #z=args.neg_mul * F.mse_loss(potg.squeeze(1), target.float(), reduction="mean")
            loss_list = [criterion(criterion_loss, output, target)]
            loss = model.L2_biasoption(loss_list, yaml["parameters"]['sigma_list'])
            loss.backward()
            optimizer.second_step(zero_grad=False)
        else:
            if iter%8==0:
                optimizer.zero_grad()
            # unscale 梯度，可以不影响clip的threshol
            scaler.scale(loss).backward(retain_graph=False)
            if iter%8==1:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if type(parameter.grad) != type(None):
                        parameter.grad/=8
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                scaler.step(optimizer)
                scaler.update()
        if yaml['data'] == 'eeg':
            prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
        elif yaml['data'] in ['mnist', 'fashionmnist', 'cifar10', 'car', 'svhn', 'cifar100', 'stl-10',"imagenet"]:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        log(model, loss.cpu(), prec1.cpu(), prec5.cpu(), scheduler.lr())
        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()
    if isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR):
        scheduler.step()
    elif isinstance(scheduler, SchedulerLR):
        scheduler(epoch)
    return log.epoch_state["top_1"] / log.epoch_state["steps"], log.epoch_state["loss"] / log.epoch_state["steps"]


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    config_=config()
    yaml = yaml_config_get(args)
    if yaml['set_seed'] is True:
        set_random_seed(yaml)
    model = merge_layer(set_device(), shape=yaml['shape'], dropout=yaml['parameters']['dropout'])
    writer = SummaryWriter()
    rand_transform = get_rand_transform(yaml['transform'])
    if yaml['data'] == 'mnist':
        mnist_trainset = datasets.MNIST(root=args.data_url, train=True, download=True, transform=rand_transform)
        mnist_testset = datasets.MNIST(root=args.data_url, train=False, download=True, transform=None)
        train_data = MNISTDataset(mnist_trainset, max_rate=1, length=yaml['parameters']['length'], flatten=True)
        train_dataloader = DataLoader(train_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                      num_workers=4,
                                      drop_last=True)
        test_data = MNISTDataset(mnist_testset, max_rate=1, length=yaml['parameters']['length'], flatten=True)
        test_dataloader = DataLoader(test_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                     num_workers=4,
                                     drop_last=True)
        model.initiate_layer(    dataoption='mnist',
                                 data=torch.randn(yaml['parameters']['batch_size'], 1, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'svhn':
        train_dataloader, test_dataloader = load_data_svhn(yaml['parameters']['batch_size'],
                                                           yaml['parameters']['batch_size'], args.data_url)
        model.initiate_layer(    dataoption='svhn',
                                 data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'cifar10':
        train_dataloader, test_dataloader = load_data(yaml['parameters']['batch_size'],
                                                      yaml['parameters']['batch_size'], args.data_url,
                                                      yaml['use_standard'])
        model.initiate_layer(    dataoption='cifar10',
                                 data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'imagenet':
        train_dataloader, test_dataloader = load_data_imagenet(yaml['parameters']['batch_size'],
                                                      yaml['parameters']['batch_size'], args.data_url)
        model.initiate_layer(dataoption='imagenet',
                             data=torch.randn(yaml['parameters']['batch_size'], 3, 224, 224),
                             num_classes=yaml["num_classes"],
                             feature_list=yaml["feature_list"],
                             size_list=yaml["size_list"],
                             hidden_size_list=yaml["hidden_size_list"],
                             path_nums_list=yaml["path_nums_list"],
                             nums_layer_list=yaml["nums_layer_list"],
                             breadth_threshold=yaml["breadth_threshold"],
                             down_rate=yaml["down_rate"],
                             mult_k=yaml["mult_k"])
    elif yaml['data'] == 'cifar100':
        train_dataloader, test_dataloader = load_data_c100(yaml['parameters']['batch_size'],
                                                           yaml['parameters']['batch_size'], args.data_url,
                                                           yaml['use_standard'])
        model.initiate_layer(    dataoption='cifar100',
                                 data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'stl-10':
        train_dataloader, test_dataloader = load_data_stl(yaml['parameters']['batch_size'],
                                                          yaml['parameters']['batch_size'], args.data_url)
        model.initiate_layer(dataoption='stl-10',
                             data=torch.randn(yaml['parameters']['batch_size'], 3, 96, 96),
                             num_classes=yaml["num_classes"],
                             feature_list=yaml["feature_list"],
                             size_list=yaml["size_list"],
                             hidden_size_list=yaml["hidden_size_list"],
                             path_nums_list=yaml["path_nums_list"],
                             nums_layer_list=yaml["nums_layer_list"],
                             breadth_threshold=yaml["breadth_threshold"],
                             down_rate=yaml["down_rate"],
                             mult_k=yaml["mult_k"])
    elif yaml['data'] == 'car':
        train_dataloader, test_dataloader = load_data_car(yaml['parameters']['batch_size'],
                                                          yaml['parameters']['batch_size'])
        model.initiate_layer(dataoption='car',
                             data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                             num_classes=yaml["num_classes"],
                             feature_list=yaml["feature_list"],
                             size_list=yaml["size_list"],
                             hidden_size_list=yaml["hidden_size_list"],
                             path_nums_list=yaml["path_nums_list"],
                             nums_layer_list=yaml["nums_layer_list"],
                             breadth_threshold=yaml["breadth_threshold"],
                             down_rate=yaml["down_rate"],
                             mult_k=yaml["mult_k"])
    elif yaml['data'] == 'fashionmnist':
        fashionmnist_trainset = datasets.FashionMNIST(root=args.data_url, train=True, download=True,
                                                      transform=rand_transform)
        fashionmnist_testset = datasets.FashionMNIST(root=args.data_url, train=False, download=True, transform=None)
        train_data = MNISTDataset(fashionmnist_trainset, max_rate=1, length=yaml['parameters']['length'], flatten=True)
        train_dataloader = DataLoader(train_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                      num_workers=4,
                                      drop_last=True)
        test_data = MNISTDataset(fashionmnist_testset, max_rate=1, length=yaml['parameters']['length'], flatten=True)
        test_dataloader = DataLoader(test_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                     num_workers=4,
                                     drop_last=True)
        model.initiate_layer(dataoption='fashionmnist',
                             data=torch.randn(yaml['parameters']['batch_size'], 1, 32, 32),
                             num_classes=yaml["num_classes"],
                             feature_list=yaml["feature_list"],
                             size_list=yaml["size_list"],
                             hidden_size_list=yaml["hidden_size_list"],
                             path_nums_list=yaml["path_nums_list"],
                             nums_layer_list=yaml["nums_layer_list"],
                             breadth_threshold=yaml["breadth_threshold"],
                             down_rate=yaml["down_rate"],
                             mult_k=yaml["mult_k"])
    elif yaml['data'] == 'eeg':
        p = random.randint(0, 4)
        train_data = EEGDateset(random=p, flatten=True, transform=True, training=True)
        test_data = EEGDateset(random=p, flatten=False, transform=False, training=False)
        train_dataloader = DataLoader(train_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                      num_workers=4,
                                      drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                     num_workers=4,
                                     drop_last=True)
        model.initiate_layer(    dataoption='eeg',
                                 data=torch.randn(yaml['parameters']['batch_size'], 14, 64, 64),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 down_rate=yaml["down_rate"],
                                 mult_k=yaml["mult_k"])

        """
        (self, data, num_classes, fn_channels,feature_list, size_list, hidden_size_list, path_nums_list,
                       nums_layer_list,down_rate,breadth_threshold, mult_k=2,drop_rate=2):
        """
    else:
        raise KeyError('There is no corresponding dataset')
    params4 = list(filter(lambda i: i.requires_grad, model.parameters()))
    dict_list4 = dict(params=params4,weight_decay=yaml['optimizer'][yaml['optimizer']['optimizer_choice']]['weight_decay'])
    optimizer = get_optimizer([dict_list4], yaml, model)
    scheduler = get_scheduler(optimizer, yaml)
    criterion_loss = make_loss(yaml['parameters'],yaml['num_classes'],None)
    model.load_state_dict(torch.load("./output/imagenet 3_4_5_6 best")['snn_state_dict'])
    model.to(set_device())
    optimizer.load_state_dict(torch.load("./output/imragenet 3_4_5_6 best")["optimizer_state_dict"])
    get_params_numeric(model)  # 5.261376
    if torch.cuda.is_available():
        criterion_loss = criterion_loss.cuda()
    if args.train == True:
        best_acc = .0
        for j in range(yaml['parameters']['epoch']):
            model.train()
            """======================"""
            for i, e in enumerate(config_.iter_epoch):
                if e<=j and i!=len(config_.iter_epoch)-1 and j<=config_.iter_epoch[i+1]:
                    a1,b1,c1,d1=config_.iter_beta[i],config_.iter_size[i],config_.iter_drop[i],config_.iter_epoch[i]
                    a2,b2,c2,d2=config_.iter_beta[i+1],config_.iter_size[i+1],config_.iter_drop[i+1],config_.iter_epoch[i+1]
                    p=(j-config_.iter_epoch[i])/(config_.iter_epoch[i+1]-config_.iter_epoch[i])
                    a,b,c=(a2-a1)*p+a1,(b2-b1)*p+b1,(c2-c1)*p+c1
                    model.set_dropout(c)
                    train_dataloader.dataset.reset_beta(a,b)
                    break
            epoch_time_stamp = time.strftime("%Y%m%d-%H%M%S")
            prec1, loss = train(model, optimizer, scheduler, train_dataloader, yaml, j, criterion_loss)
            """======================"""

            # params1.assert_buffer_is_valid()
            # params2.assert_buffer_is_valid()
            if args.test == True:
                prec1, loss = test2(model, test_dataloader, yaml, criterion_loss)
                if best_acc < prec1:
                    best_acc = prec1
                    torch.save({
                        'epoch': j,
                        'snn_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, "./output/imagenet 3_4_5_6 best")
                    path = "./output/imagenet 3_4_5_6 best 2"
                # else:
                #     torch.save({
                #         'epoch': j,
                #         'snn_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'loss': loss,
                #     }, checkpoint_path)
                #     path = checkpoint_path
        log.flush()
        print("best_acc:{:.3f}%".format(best_acc))
