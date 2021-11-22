# -*- coding: utf-8 -*-

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
    elif scheduler_choice== 'WarmupMultiStepLR':
        return WarmupMultiStepLR(optimizer,scheduler_conf[scheduler_choice]["steps"],
                                 scheduler_conf[scheduler_choice]["gamma"],scheduler_conf[scheduler_choice]["warmup_factor"],
                                 scheduler_conf[scheduler_choice]["warmup_iters"],scheduler_conf[scheduler_choice]["warmup_method"])
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
    def __init__(self, optimizer, learning_rate: float, epochs: int,turnlist:list,gamma:list):
        self.optimizer = optimizer
        self.epochs = epochs
        self.base = learning_rate
        self.gamma=gamma
        self.turnlist=[float(_) for _ in turnlist]
    def __call__(self, epoch):
        lr=self.base
        now_gamma=1.
        for i,nums in enumerate(self.turnlist):
            now_gamma*=self.gamma[i]
            if i==len(self.turnlist)-1 and self.epochs*self.turnlist[i]<=epoch:
                lr=self.base*now_gamma
                break
            if self.epochs*self.turnlist[i]<=epoch and epoch <self.epochs*self.turnlist[i+1]:
                lr=self.base*now_gamma
                break
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """
steps: [30, 55]
  gamma: 0.1
  warmup_factor: 0.3
  warmup_iters: 500
  warmup_method: "linear"
    """
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid]: hi = mid
        else: lo = mid+1
    return lo