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

sys.path.append("F:\Snn_Auto")
sys.path.append("/home/sst/product")
from Snn_Auto_master.lib.accuracy import accuracy
from Snn_Auto_master.lib.criterion import criterion
from Snn_Auto_master.lib.data_loaders import MNISTDataset, get_rand_transform, load_data,EEGDateset,load_data_car,load_data_svhn
from Snn_Auto_master.lib.log import Log
from Snn_Auto_master.lib.optimizer import get_optimizer
from Snn_Auto_master.lib.scheduler import get_scheduler, SchedulerLR
from Snn_Auto_master.lib.three_dsnn import merge_layer,filename
from Snn_Auto_master.lib.parameters_check import parametersgradCheck,parametersNameCheck

parser = argparse.ArgumentParser(description='SNN AUTO MASTER')
parser.add_argument('--config_file', type=str, default='train.yaml',
                    help='path to configuration file')
parser.add_argument('--train', dest='train', default=True, type=bool,
                    help='train model')
parser.add_argument('--test', dest='test', default=True, type=bool,
                    help='test model')
parser.add_argument('--data_url', dest='data_url',default='./data', type=str,
                    help='test model')
parser.add_argument('--log_each', dest='log_each',default=100, type=int,
                    help='how many step log once')
args = parser.parse_args()
args.config_file=filename
log = Log(log_each=args.log_each)
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

device = set_device()
def get_params_numeric(model):
    sum=0
    set_id=[]
    for parameter in model.named_parameters():
        if id(parameter[1]) not in set_id:
            sum+=parameter[1].numel()
            set_id.append(id(parameter[1]))
            # print(f"{parameter[0]},{parameter[1].numel()}")
    print(f"{round(sum/1e6,6)}M")
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
    the_model.settest(True)
    the_model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(data):
            batch_time_stamp = time.strftime("%Y%m%d-%H%M%S")
            if yaml['data'] == 'mnist':
                input = input.float().to(device)
            elif yaml['data'] == 'cifar10':
                input = input.float().to(device).view(input.shape[0], -1)
            elif yaml['data'] == 'svhn':
                input = input.float().to(device).view(input.shape[0], -1)
            elif yaml['data']=='fashionmnist':
                input = input.float().to(device)
            elif yaml['data'] == 'eeg':
                input = input.float().to(device).view(input.shape[0], -1)
            elif yaml['data'] == 'car':
                input = input.float().to(device).view(input.shape[0], -1)
            else:
                raise KeyError('not have this dataset')
            target = target.to(device)
            input.requires_grad_()
            torch.cuda.synchronize()
            output = the_model(input)
            torch.cuda.synchronize()
            loss = criterion(criterion_loss, output, target)
            if yaml['data'] == 'eeg':
                prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            elif yaml['data'] in ['mnist', 'fashionmnist', 'cifar10','car','svhn']:
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            log(model, loss.cpu(), prec1.cpu(), prec5.cpu())
    return log.epoch_state["top_1"] / log.epoch_state["steps"],log.epoch_state["loss"] / log.epoch_state["steps"]


def test(path, data, yaml, criterion_loss):
    log.eval(len_dataset=len(data))
    torch.cuda.empty_cache()
    the_model = merge_layer(set_device(), shape=yaml['shape'], dropout=yaml['parameters']['dropout'], test=True)
    if yaml['data'] == 'mnist':
        the_model.initiate_layer(torch.randn(yaml['parameters']['batch_size'],1,28,28), 1,1,int(10),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    elif yaml['data'] == 'cifar10':
        the_model.initiate_layer(torch.randn(yaml['parameters']['batch_size'],3,32,32),3,3, int(10),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    elif yaml['data']=='fashionmnist':
        the_model.initiate_layer(torch.randn(yaml['parameters']['batch_size'],1,28,28),1,1,int(10),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    elif yaml['data']=='eeg':
        the_model.initiate_layer(torch.randn(yaml['parameters']['batch_size'],14,64,64),14,14,int(2),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    elif yaml['data']=='car':
        the_model.initiate_layer(torch.randn(yaml['parameters']['batch_size'],3,64,64),3,3,int(10),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    elif yaml['data']=='svhn':
        the_model.initiate_layer(torch.randn(yaml['parameters']['batch_size'],3,32,32),3,3,int(10),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    else:
        raise KeyError('not have this dataset')
    the_model.load_state_dict(torch.load(path)['snn_state_dict'])
    device = set_device()
    the_model.to(device)
    the_model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(data):
            batch_time_stamp = time.strftime("%Y%m%d-%H%M%S")
            if yaml['data'] == 'mnist':
                input = input.float().to(device)
            elif yaml['data'] == 'cifar10':
                input = input.float().to(device).view(input.shape[0], -1)
            elif yaml['data']=='fashionmnist':
                input = input.float().to(device)
            elif yaml['data'] == 'eeg':
                input = input.float().to(device).view(input.shape[0], -1)
            elif yaml['data'] == 'car':
                input = input.float().to(device).view(input.shape[0], -1)
            elif yaml['data'] == 'svhn':
                input = input.float().to(device).view(input.shape[0], -1)
            else:
                raise KeyError()
            target = target.to(device)
            input.requires_grad_()
            torch.cuda.synchronize()
            output = the_model(input)
            torch.cuda.synchronize()
            loss = criterion(criterion_loss, output, target)
            if yaml['data'] == 'eeg':
                prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            elif yaml['data'] in ['mnist', 'fashionmnist', 'cifar10','car','svhn']:
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            log(model, loss.cpu(), prec1.cpu(), prec5.cpu())
    return log.epoch_state["top_1"] / log.epoch_state["steps"],log.epoch_state["loss"] / log.epoch_state["steps"]


def train(model, optimizer, scheduler, data, yaml, epoch, criterion_loss, path="./output"):
    log.train(len_dataset=len(data))
    sigma=yaml['parameters']['sigma']
    model.settest(False)
    for i, (input, target) in enumerate(data):

        if yaml['data'] == 'mnist':
            input = input.float().to(device)
        elif yaml['data'] == 'cifar10':
            input = input.float().to(device).view(input.shape[0], -1)
        elif yaml['data'] == 'fashionmnist':
            input = input.float().to(device)
        elif yaml['data'] == 'eeg':
            input = input.float().to(device).view(input.shape[0], -1)
        elif yaml['data'] == 'car':
            input = input.float().to(device).view(input.shape[0], -1)
        elif yaml['data'] == 'svhn':
            input = input.float().to(device).view(input.shape[0], -1)
        else:
            raise KeyError()
        target = target.to(device)
        input.requires_grad_()
        output = model(input)
        L2=model.L2_biasoption()
        loss = criterion(criterion_loss, output, target)+L2
        model.zero_grad()
        if yaml['optimizer']['optimizer_choice']=='SAM':
            loss.backward(retain_graph=False)
            # model.subWeightGrad(epoch, yaml['parameters']['epoch'], .5)
            optimizer.first_step(zero_grad=True)
            (criterion(criterion_loss,model(input),target)+model.L2_biasoption()).backward()
            # model.subWeightGrad(epoch, yaml['parameters']['epoch'], .5)
            optimizer.second_step(zero_grad=False)
        else:
            loss.backward(retain_graph=False)
            # model.subWeightGrad(epoch, yaml['parameters']['epoch'], .5)
            optimizer.step()
            # if i==600:
            #      parametersgradCheck(model)
        # pd_save(model.three_dim_layer.point_layerg+_module[str(0) + '_' + str(0) + '_' + str(0)].tensor_tau_m1.view(28,-1),"tau_m2/"+str(i))
        if yaml['data']=='eeg':
            prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
        elif yaml['data'] in ['mnist','fashionmnist','cifar10','car','svhn']:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        log(model, loss.cpu(), prec1.cpu(),prec5.cpu(),scheduler.lr())
        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()
    if isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR):
        scheduler.step()
    elif isinstance(scheduler, SchedulerLR):
        scheduler(epoch)
    return log.epoch_state["top_1"] / log.epoch_state["steps"],log.epoch_state["loss"] / log.epoch_state["steps"]


if __name__ == "__main__":
    torch.cuda.empty_cache()
    yaml = yaml_config_get(args)
    if yaml['set_seed'] is True:
        set_random_seed(yaml)
    model = merge_layer(set_device(), shape=yaml['shape'], dropout=yaml['parameters']['dropout'],test=False)
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
        model.initiate_layer(torch.randn(yaml['parameters']['batch_size'], 1,28 ,28),1,1, int(10),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    elif yaml['data'] == 'svhn':
        train_dataloader, test_dataloader = load_data_svhn(yaml['parameters']['batch_size'],
                                                      yaml['parameters']['batch_size'],args.data_url)
        model.initiate_layer(torch.randn(yaml['parameters']['batch_size'],3,32,32),3,3, int(10),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    elif yaml['data'] == 'cifar10':
        train_dataloader, test_dataloader = load_data(yaml['parameters']['batch_size'],
                                                      yaml['parameters']['batch_size'],args.data_url,yaml['use_standard'])
        model.initiate_layer(torch.randn(yaml['parameters']['batch_size'],3,32,32),3,3, int(10),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    elif yaml['data'] == 'car':
        train_dataloader, test_dataloader = load_data_car(yaml['parameters']['batch_size'],
                                                      yaml['parameters']['batch_size'])
        model.initiate_layer(torch.randn(yaml['parameters']['batch_size'], 3,32,32),3,3, int(10),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    elif yaml['data'] == 'fashionmnist':
        fashionmnist_trainset = datasets.FashionMNIST(root=args.data_url, train=True, download=True, transform=rand_transform)
        fashionmnist_testset = datasets.FashionMNIST(root=args.data_url, train=False, download=True, transform=None)
        train_data = MNISTDataset(fashionmnist_trainset, max_rate=1, length=yaml['parameters']['length'], flatten=True)
        train_dataloader = DataLoader(train_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                      num_workers=4,
                                      drop_last=True)
        test_data = MNISTDataset(fashionmnist_testset, max_rate=1, length=yaml['parameters']['length'], flatten=True)
        test_dataloader = DataLoader(test_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                     num_workers=4,
                                     drop_last=True)
        model.initiate_layer(torch.randn(yaml['parameters']['batch_size'],1,28,28),1,1, int(10),tmp_feature=yaml['parameters']['tmp_feature'],tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'],use_share_layer=yaml['set_share_layer'],  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])
    elif yaml['data'] == 'eeg':
        p=random.randint(0,4)
        print(p)
        train_data=EEGDateset(random=p,flatten=True, transform=True,training=True)
        test_data=EEGDateset(random=p,flatten=False,transform=False,training=False)
        train_dataloader=DataLoader(train_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                      num_workers=4,
                                      drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size=yaml['parameters']['batch_size'], shuffle=True,
                                     num_workers=4,
                                     drop_last=True)
        model.initiate_layer(torch.randn(yaml['parameters']['batch_size'], 14,32,32),14,14, int(2),tmp_feature=yaml['parameters']['tmp_feature']*2,tau_m=yaml['parameters']['filter_tau_m'],tau_s=yaml['parameters']['filter_tau_s'],use_gauss=False,mult_k=yaml['mult_k'],p=yaml['parameters']['dropout'] ,  push_num=yaml['parameters']['push_num'], s=yaml['parameters']['s'])

    else:
        raise KeyError('There is no corresponding dataset')
    params1 = list(filter(lambda i: i.requires_grad, model.out_classifier.parameters()))
    params_sub=list(map(id,model.out_classifier.parameters()))
    params2 = list(filter(lambda i: i.requires_grad and id(i) not in params_sub, model.parameters()))
    dict_list1=dict(params=params1,weight_decay=yaml['optimizer'][yaml['optimizer']['optimizer_choice']]['weight_decay'])
    dict_list2=dict(params=params2,
                    # weight_decay=yaml['optimizer'][yaml['optimizer']['optimizer_choice']]['weight_decay']/2
                    )
    optimizer = get_optimizer([dict_list1,dict_list2], yaml, model)
    scheduler = get_scheduler(optimizer, yaml)
    criterion_loss = torch.nn.CrossEntropyLoss()
    model.to(set_device())
    get_params_numeric(model) # 5.261376
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
                prec1,loss=test2(model, test_dataloader, yaml, criterion_loss)
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
