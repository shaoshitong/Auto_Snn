import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import os,sys,math
class MultiAttention(nn.Module):
