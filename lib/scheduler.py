# -*- coding: utf-8 -*-

"""
# File Name : scheduler.py
# Author: Haowen Fang
# Email: hfang02@syr.edu
# Description: schedulers.
"""

import torch


def get_scheduler(optimizer, conf):
    scheduler_conf = conf['scheduler']
    scheduler_choice = scheduler_conf['scheduler_choice']

    if scheduler_choice == 'MultiStepLR':
        milesones = list(scheduler_conf[scheduler_choice]['milestones'])
        print('scheduler:', scheduler_choice, 'milesones:', milesones)
        if 'gamma' in scheduler_conf[scheduler_choice]:
            gamma = scheduler_conf[scheduler_choice]['gamma']
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milesones, gamma)
        else:
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milesones)
    if scheduler_choice == 'SchedulerLR':
        trun_list = list(scheduler_conf[scheduler_choice]['milestones'])
        epochs=conf['parameters']['epoch']
        lr=conf['optimizer'][conf['optimizer']['optimizer_choice']]['lr']
        if 'gamma' in scheduler_conf[scheduler_choice]:
            gamma = scheduler_conf[scheduler_choice]['gamma']
            return SchedulerLR(optimizer,lr,epochs, trun_list, gamma)
        else:
            return SchedulerLR(optimizer,lr,epochs, trun_list)
    elif scheduler_choice == 'CosineAnnealingWarmRestarts':
        T_0 = scheduler_conf[scheduler_choice]['T_0']
        print('scheduler:', scheduler_choice, 'T_0:', T_0)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0)

    elif scheduler_choice == 'CyclicLR':
        base_lr = scheduler_conf[scheduler_choice]['base_lr']
        max_lr = scheduler_conf[scheduler_choice]['max_lr']
        step_size_up = scheduler_conf[scheduler_choice]['step_size_up']
        print('scheduler:', scheduler_conf['scheduler_choice'], 'base_lr:', base_lr,
              'max_lr:', max_lr, 'step_size_up:', step_size_up)
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up)

    elif scheduler_choice == 'none':
        return None
class SchedulerLR(object):
    def __init__(self, optimizer, learning_rate: float, epochs: int,turnlist:list,gamma=0.2):
        gamma:float
        self.optimizer = optimizer
        self.epochs = epochs
        self.base = learning_rate
        self.gamma=gamma
        self.turnlist=turnlist

    def __call__(self, epoch):
        lr=self.base
        for i,nums in enumerate(self.turnlist):
            if i==len(self.turnlist)-1:
                lr=self.base*self.gamma**(i+1)
                break
            if self.epochs*self.turnlist[i]<=epoch and epoch <self.epochs*self.turnlist[i+1]:
                lr=self.base*self.gamma**(i+1)
                break
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
