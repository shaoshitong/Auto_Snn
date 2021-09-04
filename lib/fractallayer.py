import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F

"""
Take a number composed of the lowest one of binary and the following 0
such as 7=0111->1
such as 20=10100->100
"""
lowbit = lambda x: (x) & (-x)
"""
override the class enumerate
"""


class myenumerate:
    def __init__(self, wrapped, end=None):
        self.wrapped = wrapped
        self.offset = 0
        if end == None:
            self.end = len(wrapped)
        else:
            self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset >= len(self.wrapped) or self.offset >= self.end:
            raise (StopIteration)
        else:
            item = self.wrapped[self.offset]
            self.offset += 1
            return self.offset - 1, item


"""
Randomly choose one of several tensors
"""

def rand_one_in_array(count, seed=None):
    if seed is None:
        seed = np.random.randint(1, high=int(1e6))
    global gts
    assert count > 0
    arr = torch.tensor([1.] + [.0] * (count - 1), dtype=torch.float64)
    if seed is not None:
        gts = torch.manual_seed(seed)
    index = torch.randperm(count, dtype=torch.long, generator=gts)
    return torch.index_select(arr, 0, index)

"""
mul Weighted layer
"""
"""
layer api
"""


class JoinLayer(nn.Module):
    def __init__(self, drop_p, is_global, global_path, force_path):
        super(JoinLayer, self).__init__()
        self.p = 1. - drop_p
        self.is_global = is_global
        self.global_path = global_path
        if torch.is_tensor(global_path) == False:
            self.global_path = torch.tensor(global_path, dtype=torch.float32)
        else:
            self.global_path = global_path.to(torch.float32)
        self.global_path.requires_grad = True
        self.force_path = force_path
        self.average_shape = None
    def _mulLayer(self,x):
        while self.global_path.ndim != x.ndim:
            self.global_path =self.global_path.unsqueeze(-1)
        if self.global_path.device!=x.device:
            self.global_path=self.global_path.to(x.device)
        x = torch.mean(x * self.global_path[:x.shape[0]], dim=0, keepdim=True)
        return x
    def _weights_init_list(self, x):
        self.average_shape = x.shape.tolist[1:]

    def _random_arr(self, count, p):
        return torch.distributions.Binomial(1, torch.tensor([p], dtype=torch.float32).expand(count)).sample()

    def _arr_with_one(self, count):
        return rand_one_in_array(count=count)

    def _gen_local_drops(self, count, p):
        arr = self._random_arr(count, p)
        if torch.sum(arr).item() == 0:
            return self._arr_with_one(count)
        else:
            return arr

    def _gen_global_path(self, count):
        return self.global_path[:count]

    def _drop_path(self, inputs):
        if self.average_shape is None:
            self.average_shape = inputs.shape[1:]
        count = inputs.shape[0]
        if self.is_global == True:
            drops = self.global_path[:inputs.shape[0]].cuda()
            ave = self._mulLayer(inputs)
        else:
            drops = self._gen_local_drops(count, self.p).cuda()
            while drops.ndim != inputs.ndim:
                drops = drops.unsqueeze(-1)
            ave = torch.sum(inputs * drops[:inputs.shape[0]], dim=0, keepdim=True)
        indexsum = torch.sum(drops).item()
        return ave.squeeze(0) / indexsum if indexsum and self.is_global==False else ave.squeeze(0)

    def _ave(self, inputs):
        ave=inputs[0]
        for input in inputs[1:]:
            ave+=input
        return ave/inputs.shape[0]

    def forward(self, inputs):

        inputs = self._drop_path(inputs) if (self.force_path or inputs.requires_grad or self.is_global) else self._ave(inputs)
        inputs = inputs.to(torch.float32)
        # if not inputs.requires_grad:print(1)
        return inputs


"""
layer api
"""


class JoinLayerGen:
    def __init__(self, width, global_p=0.5, deepest=False):
        self.global_p = global_p
        self.width = width
        self.switch_seed = np.random.randint(1, int(1e6))
        self.path_seed = np.random.randint(1, int(1e6))
        self.deepest = deepest
        if deepest:
            self.is_global = True
            self.path_array = torch.tensor([1.] + [.0] * (width - 1), dtype=torch.float64)
        else:
            self.is_global = self._build_global_switch()
            self.path_array = self._build_global_path_arr()

    def _build_global_path_arr(self):
        # The path the block will take when using global droppath
        return rand_one_in_array(seed=self.path_seed, count=self.width)

    def _build_global_switch(self):
        # A randomly sampled tensor that will signal if the batch
        # should use global or local droppath
        torch.manual_seed(self.switch_seed)
        p=torch.distributions.Binomial(1, torch.tensor([self.global_p], dtype=torch.float64).expand(
            1)).sample().item() == True
        return p

    def get_join_layer(self, drop_p):
        global_switch = self.is_global
        global_path = self.path_array
        return JoinLayer(drop_p=drop_p, is_global=global_switch, global_path=global_path, force_path=self.deepest)
def get_Joinlayer(width,drop_p=0.15,global_p=0.5,deepest=False,):
    join_gen=JoinLayerGen(width=width, global_p=global_p, deepest=deepest)
    merged = join_gen.get_join_layer(drop_p=drop_p)
    return merged
class LastJoiner(nn.Module):
    def __init__(self,c):
        super(LastJoiner,self).__init__()
        self.merged=get_Joinlayer(width=c,)
    def forward(self,x):
        merging = torch.cat(tuple([i.unsqueeze(0) for i in x]), dim=0)
        merged = self.merged(merging)
        return merged