import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import plotly as py
import chart_studio
import plotly.io as pio
import plotly.graph_objects as go
from plotly.graph_objs import *
import torch
import math
import plotly.express as px
import os
pio.templates.default = "simple_white"
pyplt = py.offline.plot
p = pio.renderers['png']
p.width = 800
p.height = 600
chart_studio.tools.set_config_file(world_readable=True, sharing='public')

def parametersCheck(parameters):
    resultList = []
    Check = True
    for parameter in parameters:
        if len(torch.where(torch.isnan(parameter.data) == True)[0]) > 0:
            resultList.append(len(torch.where(torch.isnan(parameter.data) == True)[0]))
            Check = False
        else:
            resultList.append(0)
    return resultList, Check


def linearSubUpdate(model):
    model.three_dim_layer.subWeightGrad()


def parametersNameCheck(model):
    for name, param in model.named_parameters():
        if type(param.grad) == type(None):
            print(name, param.requires_grad, type(param.grad), type(param.data))


def parametersgradCheck(model):
    paramdict={}
    for name, param in model.named_parameters():
        if type(param.grad) is not type(None):
            paramdict[name]=abs(torch.abs(param.grad).mean().cpu().item())
            # print('=' * 120, '\n', name, torch.max(param.grad).item(), torch.min(param.grad).item(),
            #       torch.max(param.data).item(), torch.min(param.data).item())
        else:
            paramdict[name] = -1
            # print(name, param.requires_grad, type(param.grad), torch.max(param.data).item(),
            #       torch.min(param.data).item())
    for key in paramdict.keys():
        print(key,paramdict[key])

def pd_save(tensor, name):
    df = pd.DataFrame()
    data = tensor.clone().detach().cpu().numpy()
    for i, d in enumerate(data):
        df[str(i)] = d
    df.to_csv('./output/error/' + name + '.csv', index=False, header=True)
