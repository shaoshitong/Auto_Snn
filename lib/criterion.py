import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def criterion(criterion_loss, *args):
    loss = criterion_loss(*args)
    return loss
