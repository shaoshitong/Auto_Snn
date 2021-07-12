import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def criterion(criterion_loss, output, label):
    loss = criterion_loss(output, label)
    return loss
