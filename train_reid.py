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
sys.path.append("F:\Snn_Auto")
sys.path.append("F:\sst")
sys.path.append("/home/sst/product")
from lib.accuracy import *
from lib.criterion import *
from lib.data_loaders import *
from lib.optimizer import *
from lib.scheduler import *
from lib.three_dsnn import *
from lib.loss_utils import *
from reid_utils import *
from lib.config import *
from utils.getmemcpu import getMemCpu
from lib.parameters_check import parametersgradCheck, parametersNameCheck

# torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser(description='SNN AUTO MASTER')
parser.add_argument('--config_file', type=str, default='./config/train_cuhk.yaml',
                    help='path to configuration file')
parser.add_argument('--train', dest='train', default=True, type=bool,
                    help='train model')
parser.add_argument('--test', dest='test', default=True, type=bool,
                    help='test model')
parser.add_argument('--data_url', dest='data_url', default='/data', type=str,
                    help='test model')
parser.add_argument('--neg_mul', dest='neg_mul', default=0.1, type=float,
                    help='neg_learning')
parser.add_argument('--log_each', dest='log_each', default=100, type=int,
                    help='how many step log once')
args = parser.parse_args()
scaler = torch.cuda.amp.GradScaler()


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


device = set_device()


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


def test2(model, data, yaml, criterion_loss, mAP):
    mAP.reset()
    the_model = model
    the_model.eval()
    with torch.no_grad():
        for i, (data, pids, camids, _) in enumerate(data):
            data = data.float().to(device)
            feat = model(data)
            batch = (feat, pids, camids)
            mAP.update(batch)
    cmc, map, _, _, _, _, _ = mAP.compute()
    log_print = "CMC curve, "
    for r in [1, 5, 10]:
        log_print += "ank-{:<3}:{:.1%} |".format(r, cmc[r - 1].item())
    log_print += "mAP--{:.1%} |".format(map)
    print(f"The test " + log_print)
    return cmc[0].item(), cmc[4].item()


def train(model, optimizer, scheduler, data, yaml, epoch, criterion_loss, mAP, optimizer_center=None):
    model.train()
    total_prec1 = 0.
    total_prec5 = 0.
    total_loss = 0.
    len = 0
    for i, (input, target) in enumerate(data):
        input = input.float().to(device)
        target = target.to(device)
        input.requires_grad_()
        score, feat = model(input)
        loss_list = [criterion(criterion_loss, score, feat, target)]
        loss = model.L2_biasoption(loss_list, yaml["parameters"]['sigma_list'])
        if yaml['optimizer']['optimizer_choice'] == 'SAM':
            if optimizer_center != None:
                optimizer_center.zero_grad()
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.first_step(zero_grad=True)
            score, feat = model(input)
            # z=args.neg_mul * F.mse_loss(potg.squeeze(1), target.float(), reduction="mean")
            loss_list = [criterion(criterion_loss, score, feat, target)]
            loss = model.L2_biasoption(loss_list, yaml["parameters"]['sigma_list'])
            loss.backward()
            total_loss += loss.clone().detach().cpu().item()
            optimizer.second_step(zero_grad=False)
            for param in criterion_loss.center.parameters():
                param.grad.data *= (1. / yaml["center_loss_weight"])
            optimizer_center.step()
        else:
            if optimizer_center != None:
                optimizer_center.zero_grad()
            optimizer.zero_grad()
            #loss.backward()
            scaler.scale(loss).backward(retain_graph=False)
            total_loss += loss.clone().detach().cpu().item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()
            if optimizer_center != None:
                for param in criterion_loss.center.parameters():
                    param.grad.data *= (1. / yaml["center_loss_weight"])
                optimizer_center.step()
            # torch.nn.utils.clip_grad_norm_(model.parameters(),20.)
            # scaler.step(optimizer)
            # scaler.update()
        prec1, prec5 = accuracy(score.data, target, topk=(1, 5))
        len += 1
        total_prec1 += prec1.cpu().item()
        total_prec5 += prec5.cpu().item()
        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()
    if isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR):
        scheduler.step()
    elif isinstance(scheduler, SchedulerLR):
        scheduler(epoch)
    print(
        f"The train prec1 is {round(total_prec1 / len, 3)}%,prec5 is {round(total_prec5 / len, 3)}%,loss is {round(total_loss / len, 4)}")
    return round(total_prec1 / len, 3), round(total_loss / len, 4)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    yaml = yaml_config_get(args)
    if yaml['set_seed'] is True:
        set_random_seed(yaml)
    if yaml['data'] == 'cuhk03':
        model = Backone(set_device(), shape=yaml['shape'], dropout=yaml['parameters']['dropout'])
        cuhk03_config = Config("./reid_utils/config/cuhk.yaml")
        train_loader, test_loader, num_query, num_classes = get_dataset_and_dataloader(cuhk03_config)
        model.initiate_layer_reid(dataoption='cuhk03',
                                  data=torch.randn(yaml['parameters']['batch_size'], 3, 32, 32),
                                  num_classes=num_classes,
                                  feature_list=yaml["feature_list"],
                                  size_list=yaml["size_list"],
                                  hidden_size_list=yaml["hidden_size_list"],
                                  path_nums_list=yaml["path_nums_list"],
                                  nums_layer_list=yaml["nums_layer_list"],
                                  breadth_threshold=yaml["breadth_threshold"],
                                  down_rate=yaml["down_rate"],
                                  mult_k=yaml["mult_k"],
                                  drop_rate=yaml["drop_rate"],
                                  neck=cuhk03_config.neck,
                                  neck_feat=cuhk03_config.neck_feat,
                                  )
        criterion_loss = make_loss(yaml, num_classes, model.h)
        optimizer = get_optimizer([model.parameters()], yaml, model)
        if yaml["loss_type"] == "triplet_center":
            optimizer_center = torch.optim.AdamW(criterion_loss.center.parameters(), lr=yaml["optimizer"][yaml["optimizer"]["optimizer_choice"]]["center_lr"])
        else:
            optimizer_center = None
        scheduler = get_scheduler(optimizer, yaml)
        mAP = R1_mAP(num_query, max_rank=50, feat_norm=cuhk03_config.test_feat_norm)
    elif yaml['data'] == 'market1501':
        raise NotImplementedError("Not Import Dataset market1501")
    else:
        raise KeyError('There is no corresponding dataset')
    model.to(device)
    get_params_numeric(model)  # 5.261376
    if torch.cuda.is_available():
        criterion_loss = criterion_loss.cuda()
    if args.train == True:
        best_acc = .0
        for j in range(yaml['parameters']['epoch']):
            print(f"Epoch {j} Stated:")

            prec1, loss = train(model, optimizer, scheduler, train_loader, yaml, j, criterion_loss, mAP,
                                optimizer_center)
            if args.test == True:
                cmc, amp = test2(model, test_loader, yaml, criterion_loss, mAP)
            if best_acc < prec1:
                best_acc = prec1
                torch.save({"parameter": model.state_dict()}, "/data/best model")
        print("best_acc:{:.3f}%".format(best_acc))
