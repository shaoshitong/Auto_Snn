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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import torch.nn.functional as F
sys.path.append("D:\Product")

sys.path.append("F:\sst")
sys.path.append("/home/sst/product")
from lib.accuracy import accuracy
from lib.criterion import criterion
from lib.data_loaders import MNISTDataset, get_rand_transform, load_data, EEGDateset, load_data_car, \
    load_data_svhn, load_data_c100, load_data_stl,load_data_imagenet
from lib.log import Log
from lib.optimizer import get_optimizer
from lib.scheduler import get_scheduler, SchedulerLR
from lib.three_dsnn import merge_layer, filename
from lib.parameters_check import parametersgradCheck, parametersNameCheck

parser = argparse.ArgumentParser(description='SNN AUTO MASTER')
parser.add_argument('--config_file', type=str, default='train_c10.yaml',
                    help='path to configuration file')
parser.add_argument('--train', dest='train', default=True, type=bool,
                    help='train model')
parser.add_argument('--test', dest='test', default=True, type=bool,
                    help='test model')
parser.add_argument('--data_url', dest='data_url', default='./data', type=str,
                    help='test model')
parser.add_argument('--neg_mul', dest='neg_mul', default=0.1, type=float,
                    help='neg_learning')
parser.add_argument('--log_each', dest='log_each', default=100, type=int,
                    help='how many step log once')
args = parser.parse_args()
args.config_file = filename
log = Log(log_each=args.log_each)
scaler = torch.cuda.amp.GradScaler()
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


device = set_device()


def Loss_get(name="cross"):
    class smooth_crossentropy(object):
        def __init__(self):
            pass

        def __call__(self, pred, gold, smoothing=0.1, *args, **kwargs):
            n_class = pred.size(1)
            gold = gold.to(pred.device)
            one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1)).to(pred.device)  # 0.0111111
            one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1. - smoothing)  # 0.9
            log_prob = torch.nn.functional.log_softmax(pred, dim=1)
            return torch.nn.functional.kl_div(input=log_prob, target=one_hot, reduction='none').sum(dim=-1).mean()

        def cuda(self, ):
            return self

    if name == "cross":
        return torch.nn.CrossEntropyLoss()
    elif name == "smooth_cross":
        return smooth_crossentropy()
    else:
        return None


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
    torch.backends.cudnn.benchmark = False
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
                input = input.float().to(device).view(input.shape[0], -1)
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
        the_model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 1, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'cifar10':
        the_model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'imagenet':
        the_model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 3,224, 224),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'cifar100':
        the_model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'fashionmnist':
        the_model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 1, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'eeg':
        the_model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 14, 64, 64),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'car':
        the_model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 3, 64, 64),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'svhn':
        the_model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'stl-10':
        the_model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 3, 96, 96),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
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
        if yaml['data'] in ['mnist', 'fashionmnist']:
            input = input.float().to(device)
        elif yaml['data'] in ['cifar10', 'cifar100', 'svhn', 'eeg', 'car', 'stl-10']:
            input = input.float().to(device).view(input.shape[0], -1)
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
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.first_step(zero_grad=True)
            output= model(input)
            #z=args.neg_mul * F.mse_loss(potg.squeeze(1), target.float(), reduction="mean")
            loss_list = [criterion(criterion_loss, output, target)]
            loss = model.L2_biasoption(loss_list, yaml["parameters"]['sigma_list'])
            loss.backward()
            optimizer.second_step(zero_grad=False)
        else:
            optimizer.zero_grad()
            # unscale 梯度，可以不影响clip的threshol
            scaler.scale(loss).backward(retain_graph=False)
            scaler.unscale_(optimizer)

            # clip梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer)
            scaler.update()


        #     print("\r",f"{(torch.eq(torch.argmax(potg_a,dim=-1),target).sum()/potg_a.size()[0]).item()},"
        #           f"{(torch.eq(torch.argmax(potg_b,dim=-1),target).sum()/potg_b.size()[0]).item()},"
        #           f"{(torch.eq(torch.argmax(potg_c,dim=-1),target).sum()/potg_c.size()[0]).item()},"
        #           f"{(torch.eq(torch.argmax(output, dim=-1), target).sum() / output.size()[0]).item()},",end="",flush=True)
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
        model.initiate_layer(    data=torch.randn(yaml['parameters']['batch_size'], 1, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'svhn':
        train_dataloader, test_dataloader = load_data_svhn(yaml['parameters']['batch_size'],
                                                           yaml['parameters']['batch_size'], args.data_url)
        model.initiate_layer(    data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'cifar10':
        train_dataloader, test_dataloader = load_data(yaml['parameters']['batch_size'],
                                                      yaml['parameters']['batch_size'], args.data_url,
                                                      yaml['use_standard'])
        model.initiate_layer(    data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'imagenet':
        train_dataloader, test_dataloader = load_data_imagenet(yaml['parameters']['batch_size'],
                                                      yaml['parameters']['batch_size'], args.data_url)
        model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 3, 224, 224),
                             num_classes=yaml["num_classes"],
                             feature_list=yaml["feature_list"],
                             size_list=yaml["size_list"],
                             hidden_size_list=yaml["hidden_size_list"],
                             path_nums_list=yaml["path_nums_list"],
                             nums_layer_list=yaml["nums_layer_list"],
                             breadth_threshold=yaml["breadth_threshold"],
                             mult_k=yaml["mult_k"])
    elif yaml['data'] == 'cifar100':
        train_dataloader, test_dataloader = load_data_c100(yaml['parameters']['batch_size'],
                                                           yaml['parameters']['batch_size'], args.data_url,
                                                           yaml['use_standard'])
        model.initiate_layer(    data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    elif yaml['data'] == 'stl-10':
        train_dataloader, test_dataloader = load_data_stl(yaml['parameters']['batch_size'],
                                                          yaml['parameters']['batch_size'], args.data_url)
        model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 3, 96, 96),
                             num_classes=yaml["num_classes"],
                             feature_list=yaml["feature_list"],
                             size_list=yaml["size_list"],
                             hidden_size_list=yaml["hidden_size_list"],
                             path_nums_list=yaml["path_nums_list"],
                             nums_layer_list=yaml["nums_layer_list"],
                             breadth_threshold=yaml["breadth_threshold"],
                             mult_k=yaml["mult_k"])
    elif yaml['data'] == 'car':
        train_dataloader, test_dataloader = load_data_car(yaml['parameters']['batch_size'],
                                                          yaml['parameters']['batch_size'])
        model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                             num_classes=yaml["num_classes"],
                             feature_list=yaml["feature_list"],
                             size_list=yaml["size_list"],
                             hidden_size_list=yaml["hidden_size_list"],
                             path_nums_list=yaml["path_nums_list"],
                             nums_layer_list=yaml["nums_layer_list"],
                             breadth_threshold=yaml["breadth_threshold"],
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
        model.initiate_layer(data=torch.randn(yaml['parameters']['batch_size'], 1, 32, 32),
                             num_classes=yaml["num_classes"],
                             feature_list=yaml["feature_list"],
                             size_list=yaml["size_list"],
                             hidden_size_list=yaml["hidden_size_list"],
                             path_nums_list=yaml["path_nums_list"],
                             nums_layer_list=yaml["nums_layer_list"],
                             breadth_threshold=yaml["breadth_threshold"],
                             mult_k=yaml["mult_k"])
    elif yaml['data'] == 'eeg':
        p = random.randint(0, 4)
        print(p)
        train_data = EEGDateset(random=p, flatten=True, transform=True, training=True)
        test_data = EEGDateset(random=p, flatten=False, transform=False, training=False)
        train_dataloader = DataLoader(train_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                      num_workers=4,
                                      drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                     num_workers=4,
                                     drop_last=True)
        model.initiate_layer(    data=torch.randn(yaml['parameters']['batch_size'], 14, 64, 64),
                                 num_classes=yaml["num_classes"],
                                 feature_list=yaml["feature_list"],
                                 size_list=yaml["size_list"],
                                 hidden_size_list=yaml["hidden_size_list"],
                                 path_nums_list=yaml["path_nums_list"],
                                 nums_layer_list=yaml["nums_layer_list"],
                                 breadth_threshold=yaml["breadth_threshold"],
                                 mult_k=yaml["mult_k"])
    else:
        raise KeyError('There is no corresponding dataset')
    params4 = list(filter(lambda i: i.requires_grad, model.parameters()))
    dict_list4 = dict(params=params4,weight_decay=yaml['optimizer'][yaml['optimizer']['optimizer_choice']]['weight_decay'])
    optimizer = get_optimizer([dict_list4], yaml, model)
    scheduler = get_scheduler(optimizer, yaml)
    criterion_loss = Loss_get(yaml["parameters"]["loss_option"])
    model.to(set_device())

    get_params_numeric(model)  # 5.261376
    if torch.cuda.is_available():
        criterion_loss = criterion_loss.cuda()
    if args.train == True:
        best_acc = .0
        for j in range(yaml['parameters']['epoch']):
            model.train()
            epoch_time_stamp = time.strftime("%Y%m%d-%H%M%S")
            prec1, loss = train(model, optimizer, scheduler, train_dataloader, yaml, j, criterion_loss)
            # params1.assert_buffer_is_valid()
            # params2.assert_buffer_is_valid()
            if args.test == True:
                checkpoint_path = os.path.join(yaml['output'], str(j) + '_' + epoch_time_stamp + str(best_acc))
                prec1, loss = test2(model, test_dataloader, yaml, criterion_loss)
                if best_acc < prec1:
                    best_acc = prec1
                #     torch.save({
                #         'epoch': j,
                #         'snn_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'loss': loss,
                #     }, checkpoint_path + 'best')
                #     path = checkpoint_path + 'best'
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
