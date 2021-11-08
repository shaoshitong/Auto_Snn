import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
# from lib.memory import  MemTracker
from lib.GRU import multi_GRU, multi_block_eq, Cat, DenseBlock, cat_result_get,return_tensor_add,numeric_get,\
    aplha_decay,token_numeric_get
from lib.utils import *
from lib.DenseNet import DenseBlock as DenseDeepBlock
import math
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
# frame = inspect.currentframe()          # define a frame to track
# gpu_tracker = MemTracker(frame)         # define a GPU tracker

def yaml_config_get(yamlname):
    """
    该函数是为了获取模型的配置信息
    """

    conf = OmegaConf.load(yamlname)
    return conf


class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x, *args):
        return self.function(x, x.size()[-1], *args)


def batch_norm(input):
    input_linear = input.view(input.shape[0], -1)
    mean = input_linear.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    std = input_linear.std(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    return torch.div(torch.sub(input, mean), std)


filename = "./train_c10_hr.yaml"
yaml = yaml_config_get(filename)
# yaml = yaml_config_get("./train.yaml")
dataoption = yaml['data']

def size_change(f, s):
    def change(xx):
        xx: torch.Tensor
        xx = F.interpolate(xx, (s, s), mode='bilinear', align_corners=True)
        g = f // xx.shape[1]
        if (int(g) == 0):
            xx = xx[:, ::int(xx.shape[1] // f), :, :]
        else:
            xx = xx.repeat(1, f // xx.shape[1], 1, 1)
        return xx

    return change


class Shortcut(nn.Module):
    """
    该层是为了残差连接而设计，从输入数据剪切一块后进行填充
    目前在特征从初始变换到64大小时会用到
    """

    def __init__(self, in_feature, out_feature, use_same=False, proportion=2):
        in_feature: int
        out_feature: int
        super(Shortcut, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        if use_same == False:
            self.shortcut = lambda x: F.pad(x[:, :, ::proportion, ::proportion],
                                            (0, 0, 0, 0, (self.out_feature - x.shape[1]) // 2,
                                             (self.out_feature - x.shape[1]) // 2 + (
                                                     self.out_feature - x.shape[1]) % 2),
                                            "constant", 0)
        else:
            self.shortcut = lambda x: F.pad(x,
                                            (0, 0, 0, 0, (self.out_feature - x.shape[1]) // 2,
                                             (self.out_feature - x.shape[1]) // 2 + (
                                                     self.out_feature - x.shape[1]) % 2),
                                            "constant", 0)

    def forward(self, x):
        return self.shortcut(x)


class block_out(nn.Module):
    def __init__(self, feature, classes, size, use_pool='none'):
        """
        size=[32,16,8,4]
        feature=[a,b,c,d]
        classes=10
        """
        super(block_out, self).__init__()


        self.classifiar = nn.Sequential(nn.Flatten(), nn.Linear(sum(feature), classes))
        self.avg = nn.Sequential(*[Lambda(F.avg_pool2d)])
        size=size[1:]
        self.transition_layer=nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(feature[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature[i], feature[i], (1, 1), (size[i]//size[-1],size[i]//size[-1]), (0, 0), bias=False)
            ) for i in range(len(feature[:-1]))
        ])
        self.training = False
        self.use_pool = use_pool
        self.size = size
        self.classes = classes
        self._initialize()

    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1.)
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                # layer.weight.data.zero_()
                layer.bias.data.zero_()

    def forward(self, inputs):
        res=[]
        for i in range(len(inputs[:-1])):
            res.append(self.transition_layer[i](inputs[i]))
        res.append(inputs[-1])
        res=torch.cat(res,dim=1)
        res=self.avg(res)
        return self.classifiar(res)

# class block_eq(nn.Module):
#     def __init__(self, eq_feature,tmp_feature,dropout):
#         super(block_eq, self).__init__()
#         self.eq_feature = eq_feature
#         self.tmp_feature=tmp_feature
#         self.convz1 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (1, 5), padding=(0, 2))
#         self.convr1 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (1, 5), padding=(0, 2))
#         self.convq1 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (1, 5), padding=(0, 2))
#         self.convz2 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (5, 1), padding=(2, 0))
#         self.convr2 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (5, 1), padding=(2, 0))
#         self.convq2 = nn.Conv2d(eq_feature + tmp_feature, eq_feature, (5, 1), padding=(2, 0))
#         self.convo = nn.Sequential(*[
#             nn.Conv2d(eq_feature, eq_feature, (3, 3), (1, 1), (1, 1), bias=False),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout),
#             nn.BatchNorm2d(eq_feature),
#         ])
#         self.xgru=nn.Sequential(nn.Conv2d(eq_feature,tmp_feature,(1,1),(1,1),padding=0,bias=False))
#         self._initialize()
#     def _initialize(self):
#         for layer in self.modules():
#             if isinstance(layer, nn.Conv2d):
#                 nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="sigmoid")
#                 if layer.bias is not None:
#                     nn.init.zeros_(layer.bias.data)
#             if isinstance(layer, nn.BatchNorm2d):
#                 nn.init.ones_(layer.weight.data)
#                 nn.init.zeros_(layer.bias.data)
#         for layer in self.convo.modules():
#             if isinstance(layer, nn.Conv2d):
#                 nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
#                 if layer.bias is not None:
#                     nn.init.zeros_(layer.bias.data)
#
#         nn.init.kaiming_normal_(self.convq1.weight.data, mode="fan_in", nonlinearity="tanh")
#         nn.init.kaiming_normal_(self.convq2.weight.data, mode="fan_in", nonlinearity="tanh")
#
#     def forward(self, m):
#         h, x = m
#         hx = torch.cat([h, x], dim=1)
#         z = torch.sigmoid(self.convz1(hx))
#         r = torch.sigmoid(self.convr1(hx))
#         q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
#         h = F.relu((1 + z) * h + (1 - z) * q)
#         hx  = torch.cat([h, x], dim=1)
#         z = torch.sigmoid(self.convz2(hx))
#         r = torch.sigmoid(self.convr2(hx))
#         q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
#         h = F.relu((1 + z) * h + (1 - z) * q)
#         x =  self.xgru(h)
#         h =  self.convo(h)
#         del  m, z, r, q
#         return (h, x)
#
#
# class multi_block_eq(nn.Module):
#     def __init__(self, in_feature, out_feature,hidden_size, multi_k=1, stride=1,dropout=0.1):
#         super(multi_block_eq, self).__init__()
#         if in_feature != out_feature or stride != 1:
#             self.sample = nn.Sequential(
#                 nn.Conv2d(in_feature, out_feature, (stride, stride), (stride, stride), bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(out_feature))
#             self.psample = nn.Sequential(
#                 nn.Conv2d(in_feature, hidden_size, (stride, stride), (stride, stride), bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(hidden_size))
#             self.qsample= nn.Sequential(
#                 nn.Conv2d(hidden_size, out_feature, (1,1),(1,1), bias=False))
#             for layer in self.sample.modules():
#                 if isinstance(layer, nn.Conv2d):
#                     nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
#                 if isinstance(layer, nn.BatchNorm2d):
#                     nn.init.ones_(layer.weight.data)
#                     nn.init.zeros_(layer.bias.data)
#             for layer in self.psample.modules():
#                 if isinstance(layer, nn.Conv2d):
#                     nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
#                 if isinstance(layer, nn.BatchNorm2d):
#                     nn.init.ones_(layer.weight.data)
#                     nn.init.zeros_(layer.bias.data)
#         self.xgru=nn.Sequential(nn.Conv2d(in_feature,in_feature,(1,1),(1,1),padding=0,bias=False))
#         self.model = nn.Sequential(*[
#             block_eq(out_feature,hidden_size,dropout) for _ in range(multi_k)
#         ])
#
#     def forward(self, x):
#         x,h=x
#         if h == None:
#             h = self.xgru(x)
#         if hasattr(self, "sample"):
#             x = self.sample(x)
#             h = self.psample(h)
#         x,h=self.model((x, h))
#         h=self.qsample(h)
#         return (x,h)
#

class Trinomial_operation(object):
    def __init__(self, max_n, tau_m=1., tau_s=4.):
        self.max_n = max_n
        self.tau_s = tau_s
        self.tau_m = tau_m
        self.Trinomial_list()

    def Trinomial_list(self):
        max_n = self.max_n
        self.diag = [1 for i in range(self.max_n * 3)]
        for i in range(1, max_n * 3):
            self.diag[i] = self.diag[i - 1] * (i)
        self.diag_T = torch.ones(self.max_n, self.max_n, self.max_n, dtype=torch.float32)
        for i in range(self.max_n):
            for j in range(self.max_n):
                for k in range(self.max_n):
                    self.diag_T[i][j][k] = self.diag[i + j + k] / (self.diag[i] * self.diag[j] * self.diag[k])

    def get_value(self, i, j, k):
        if i >= self.max_n or j >= self.max_n or k >= self.max_n:
            exit(-1)
        return self.diag_T[i][j][k]



class point_cul_Layer(nn.Module):
    def __init__(self,is_diag, in_feature, out_feature, hidden_size, in_size, out_size, true_out, cat_x, cat_y,b,d,x,y, STuning=True,
                 grad_lr=0.1, dropout=0.3,use_gauss=True, mult_k=2):
        """
        输入的张量维度为（batch_size,64,x//2,y//2）
        该层通过门机制后进行卷积与归一化
        """
        super(point_cul_Layer, self).__init__()
        self.is_diag=is_diag
        if is_diag==False:
            self.cat_feature =  (out_feature) + in_feature
            if cat_x==cat_y:
                fusion=1
            elif (cat_x>cat_y)==0:
                fusion=0
            else:
                fusion=2
            self.DoorMach = DenseBlock(self.cat_feature, max(1,int(true_out/(d**min(1,abs(cat_x-cat_y))))), hidden_size, cat_x, cat_y,
                                       dropout,fusion,in_size)
            self.STuning = STuning
            self.b=b
            self.grad_lr = grad_lr
            self.sigma = 1
            self.cat_x,self.cat_y=cat_x,cat_y
            self.norm = None
        else:
            self.cat_feature = (out_feature) + in_feature
            self.DoorMach= DenseBlock(self.cat_feature, max(1,int(true_out/(d**min(1,abs(cat_x-cat_y))))), hidden_size, cat_x, cat_y,
                                       dropout,1,in_size)
            self.b = b
            self.grad_lr = grad_lr
            self.sigma = 1
            self.cat_x, self.cat_y = cat_x, cat_y
            self.norm = None

    def forward(self, x):
        if self.is_diag==False:
            tensor_prev, (i, j),tag,(pre_i,pre_j) = x
            return self.DoorMach(cat_result_get(tensor_prev, i, j ,self.b,tag,pre_i,pre_j))
        else:
            tensor_prev, (i, j),tag,(pre_i,pre_j) = x
            return self.DoorMach(cat_result_get(tensor_prev, i, j, self.b,tag,pre_i,pre_j))
class non_two_dim_layer(nn.Module):
    def __init__(self, in_feature, out_feature, hidden_size, in_size, out_size, x, y,b,down_rate, mult_k=2, p=0.2):
        super(non_two_dim_layer, self).__init__()
        self.np_last=in_feature
    def forward(self,x):
        return x
class two_dim_layer(nn.Module):
    def __init__(self, in_feature, out_feature, hidden_size, in_size, out_size, x, y,b,down_rate, mult_k=2, p=0.2):
        super(two_dim_layer, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.hidden_size = hidden_size
        self.in_pointnum = in_size
        self.out_pointnum = out_size
        self.x = x
        self.y = y
        self.b=b
        assert self.x>=self.b
        assert self.x>=1 and self.y>=1
        """
        +3 +7 +11 +15
        +2 +5 +8 +11
        +1 +3 +5 +7
        +1 +1 +2 +3
        """
        self.point_cul_layer = {}
        self.test = False
        self.tensor_check,self.push_list=token_numeric_get(x,y,b,out_feature,down_rate)
        # for col in self.tensor_check:
        #     print(col)
        # print("\n\n")
        if self.x>0 and self.y>0:
            for i in range(0,self.x):
                for j in range(0,self.y):
                    if i!=0 or j!=0:
                        if abs(i-j)<self.b:
                            if not (i==self.x-1 and j==self.y-1):
                                """
                                in_feature, out_feature, hidden_size, in_size, out_size, 
                                true_out, cat_x, cat_y,b,d, STuning=True,
                                grad_lr=0.1, dropout=0.3,use_gauss=True, mult_k=2,
                                """
                                self.point_cul_layer[str(i) + "_" + str(j)] = point_cul_Layer(
                                    False,
                                    in_feature,
                                    self.tensor_check[i][j],
                                    hidden_size,
                                    in_size,
                                    out_size,
                                    out_feature,
                                    i ,
                                    j ,
                                    b ,
                                    down_rate,
                                    x,
                                    y,
                                    dropout=p,
                                    mult_k=mult_k)
                            else:
                                self.point_cul_layer[str(i) + "_" + str(j)] = point_cul_Layer(
                                    True,
                                    in_feature,
                                    self.tensor_check[i][j],
                                    hidden_size,
                                    in_size,
                                    out_size,
                                    out_feature,
                                    i ,
                                    j ,
                                    b ,
                                    down_rate,
                                    x,
                                    y,
                                    dropout=p,
                                    mult_k=mult_k)
            self.point_layer_module = nn.ModuleDict(self.point_cul_layer)
            self.np_last = self.tensor_check[self.x-1][self.y-1]+out_feature+in_feature
            self.cross_loss = nn.CrossEntropyLoss()
        """self.dimixloss= nn.ModuleList([Linear_adaptive_loss(out_feature,out_size) for _ in range(1)])"""
    def forward(self, z):
        if self.x==0 and self.y==0:
            return z
        tensor_prev = [[z for i in range(self.x)] for j in range(self.y)]
        for i in range(len(self.push_list[1:])):
            pre_a,pre_b=self.push_list[i]
            a,b = self.push_list[i + 1]
            if pre_a<=a and pre_b<=b:
                tensor_prev[a][b] = self.point_layer_module[str(a) + "_" + str(b)]((tensor_prev, (a, b),False,(pre_a,pre_b)))
            else:
                tensor_prev[a][b] = self.point_layer_module[str(a) + "_" + str(b)]((tensor_prev, (a, b),True,(pre_a,pre_b)))
        result = []
        for i in range(self.x):
            for j in range(self.y):
                if abs(i-j)<self.b:
                    result.append(tensor_prev[i][j])
        tensor_prev = torch.cat(result, dim=1)
        return tensor_prev

    def settest(self, test=True):
        self.test = test
    def reset(self,random_float):
        with torch.no_grad():
            for i in range(1, self.x):
                for j in range(1, self.y):
                    if abs(i - j) < self.b:
                        if random.random() > random_float:
                            for layer in self.point_layer_module[str(i) + "_" + str(j)].modules():
                                if isinstance(layer,nn.Conv2d):
                                    nn.init.kaiming_normal_(layer.weight.data,mode="fan_in",nonlinearity="relu")


class turn_layer(nn.Module):
    def __init__(self, in_feature, out_feature, bn_size, num_layer, decay_rate=2, stride=1, dropout=0.1):
        super(turn_layer, self).__init__()
        if num_layer != 0:
            self.downsample = nn.Sequential(*[])
            self.downsample.add_module('norm', nn.BatchNorm2d(in_feature))
            self.downsample.add_module("relu", nn.ReLU(inplace=True))
            self.downsample.add_module("conv",
                                       nn.Conv2d(in_feature,int(in_feature/decay_rate), (1, 1), (1, 1), (0, 0), bias=False))
            self.xsample = nn.Sequential(*[])
            self.xsample.add_module('pool', nn.AvgPool2d(kernel_size=(stride, stride), stride=(stride, stride)))

            in_feature = int(in_feature / decay_rate)
            if num_layer!=1:
                self.dense_deep_block = DenseDeepBlock([in_feature] + [out_feature] * num_layer, bn_size, dropout,
                                                       num_layer)
                in_feature = in_feature + out_feature * num_layer
            self.origin_out_feature =in_feature
            self.num_layer = num_layer
        else:
            self.origin_out_feature =  int(in_feature)
            self.num_layer = num_layer
            self.downsample = nn.Sequential(*[])
            self.downsample.add_module('norm', nn.BatchNorm2d(in_feature))
            self.downsample.add_module("relu", nn.ReLU(inplace=True))
            self.downsample.add_module('pool', nn.AvgPool2d(kernel_size=(stride, stride), stride=(stride, stride)))
            in_feature = in_feature + out_feature * num_layer
            self.origin_out_feature =in_feature
        self._initialize()

    def _initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
            elif isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)

    def forward(self, x):
        if self.num_layer != 0:
            if self.num_layer!=1:
                return self.dense_deep_block(self.xsample(self.downsample(x)))
            else:
                return self.xsample(self.downsample(x))
        else:
            # gpu_tracker.track()
            return self.downsample(x)


class Iter_Layer(nn.Module):
    def __init__(self, shape, device, p=0.1):
        super(Iter_Layer, self).__init__()
        """
        该层便是three-dim层
        x维度代表原始数据通过卷积变换至[batchsize,64,x//2,y//2]
        y维度代表原始数据先获得grad后经过卷积变换到[batchsize,64,x//2,y//2]
        z维度目前用0向量填充，未来可以改进
        """
        self.a, self.b, self.c = shape[0], shape[1], shape[2]
        self.shape = shape
        self.device = device
        self.dropout = p
    def forward(self, x):
        """
        x,y=>[batchsize,64,x_pointnum,y_pointnum]
        """
        for i in range(self.len):
            x=self.turn_layer_module[str(i)](x)
            x=[self.point_layer_module[str(i)][j](x[j]) for j in range(len(self.size_list))]
        return x

    def initiate_layer(self,
                       data,
                       fn_channels,
                       feature_list,
                       size_list,
                       hidden_size_list,
                       path_nums_list,
                       nums_layer,
                       decay_rate,
                       down_rate,
                       breadth_threshold):
        """
        three-dim层初始化节点
        """
        self.point_layer = {}
        self.turn_layer = {}

        self.in_channels=data.shape[1]
        self.fn_channels=fn_channels

        _Multi_Fusion_Create=Multi_Fusion_Create(    in_channels=self.in_channels,
                                                     channnels=self.fn_channels,
                                                     size_list=size_list[1:],
                                                     in_size=size_list[0])

        """
        size_list=[ 32 , 16 , 8 , 4 ]
        feature_list=   [
                          
                        [32],
                        [32,32],
                        [32,32,32],
                        [32,32,32,32],
                        
                        ]
        
        hidden_size_list=   [
        
                            [4],
                            [4,4],
                            [4,4,4],
                            [4,4,4,4]
                            
                            ]
        
        path_nums_list =    [
                            
                            [4],
                            [4,4],
                            [4,4,4],
                            [4,4,4,4],
                            
                            ]
        
        """
        self.in_shape = data.shape
        size_list=size_list[1:]
        assert len(feature_list) == len(size_list) and len(hidden_size_list) == len(path_nums_list) and len(
            path_nums_list) == len(nums_layer) and len(breadth_threshold)==len(nums_layer)

        feature_list,path_nums_list,nums_layer,breadth_threshold=balance_ilist(feature_list, path_nums_list, nums_layer, breadth_threshold)
        self.feature_list,self.path_nums_list,self.nums_layer,self.breadth_threshold=feature_list,path_nums_list,nums_layer,breadth_threshold
        self.size_list=size_list
        assert len(feature_list[0])>0

        for i in range(len(feature_list[0])):

            if i==0:
                self.turn_layer[str(i)]=_Multi_Fusion_Create
                index=[_i for _i in range(len(size_list))]
                mm = self.turn_layer[str(i)].np_last
            else:
                tmp_f=copy.deepcopy(f)
                for j in range(len(size_list)):
                    if feature_list[j][i]==None:
                        tmp_f[j]=None
                now_size_list,now_f,index=filter_list(size_list,tmp_f)
                self.turn_layer[str(i)]=Multi_Fusion(now_size_list,now_f,decay_rate_list(now_f,decay_rate,index),index)
                mm=self.turn_layer[str(i)].np_last
                for q,k in enumerate(index):
                    f[k]=mm[q]
                mm=f
            print(mm)
            point_mode=nn.ModuleList([])
            for j in range(len(size_list)):
                m=mm[j]
                f2=feature_list[j][i]
                h=hidden_size_list[i]
                s=size_list[j]
                p=path_nums_list[j][i]
                n=nums_layer[j][i]
                b=breadth_threshold[j][i]
                if f2==None:
                    point_mode.append(
                        non_two_dim_layer(m,f2,h,s,s,p,p,b,down_rate,0,self.dropout))
                else:
                    point_mode.append(
                        two_dim_layer(m, f2, h, s, s, p, p, b, down_rate, 0, self.dropout))
            self.point_layer[str(i)]=point_mode
            f=[self.point_layer[str(i)][z].np_last for z in range(len(self.point_layer[str(i)]))]

        self.turn_layer_module = nn.ModuleDict(self.turn_layer)
        self.point_layer_module = nn.ModuleDict(self.point_layer)
        self.len = len(self.feature_list[0])
        del self.point_layer, self.turn_layer
        return f


class merge_layer(nn.Module):
    def __init__(self, device, shape=None, dropout=0.3):
        """
        该层是basic层,包含了特征变换层和three-dim路径选择层
        """
        super(merge_layer, self).__init__()
        if shape == None:
            self.shape = [2, 2, 2]
        else:
            self.shape = shape
        self.device = device
        self.iter=0.
        self.InputGenerateNet = Iter_Layer(self.shape, self.device, dropout).to(device)

    def forward(self, x):
        # x, y = self.initdata(x)
        with torch.no_grad():
            if hasattr(self, 'input_shape'):
                x = x.view(self.input_shape)
            else:
                if dataoption in ['cifar10', 'cifar100']:
                    x = x.view(x.shape[0], 3, 32, 32)
                elif dataoption == 'mnist':
                    x: torch.Tensor
                    x = x.view(x.shape[0], 1, 28, 28)
                    x = F.interpolate(x, (32, 32), mode='bilinear', align_corners=True)
                elif dataoption == 'imagenet':
                    pass
                elif dataoption == 'fashionmnist':
                    x = x.view(x.shape[0], 1, 28, 28)
                    x = F.interpolate(x, (32, 32), mode='bilinear', align_corners=True)
                elif dataoption == 'eeg':
                    x = x.view(x.shape[0], 14, 32, 32)
                    # 64,16,16
                elif dataoption == 'car':
                    x = x.view(x.shape[0], 3, 64, 64)
                    # 64,16,16
                elif dataoption == 'svhn':
                    x = x.view(x.shape[0], 3, 32, 32)

                elif dataoption == "stl-10":
                    x = x.view(x.shape[0], 3, 96, 96)
                else:
                    raise KeyError()
        x = self.InputGenerateNet(x)
        x = self.out_classifier(x)
        return x

    def initiate_layer(self, data, num_classes, fn_channels,feature_list, size_list, hidden_size_list, path_nums_list,
                       nums_layer_list,down_rate,breadth_threshold, mult_k=2,drop_rate=2):
        """
        配置相应的层
        """
        b, c, h, w = data.shape
        input_shape = (b, c, h, w)
        """
        if dataoption=="imagenet":
            self.inf = nn.Sequential(*[nn.Conv2d(c, feature_list[0], (7, 7), (2, 2), (2, 2), bias=False),])
        else:
            self.inf = nn.Conv2d(c, feature_list[0], (3, 3), (1,1), (1, 1), bias=False)
        self._initialize()
        
                        data,
                       fn_channels,
                       feature_list,
                       size_list,
                       hidden_size_list,
                       path_nums_list,
                       nums_layer,
                       decay_rate,
                       down_rate,
                       breadth_threshold):
        """
        h = self.InputGenerateNet.initiate_layer(data,fn_channels, feature_list, size_list, hidden_size_list, path_nums_list,
                                                 nums_layer_list, drop_rate,down_rate,breadth_threshold)
        self.out_classifier = block_out(h, num_classes, size_list)
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    @staticmethod
    def _list_build():
        return [0.1, 0.1]

    @staticmethod
    def _list_print(list):
        for i in list:
            print(i.squeeze().item(), end=",")
        print("")
    def reset(self,random_float=1.0):
        for layer in self.modules():
            if isinstance(layer,two_dim_layer):
                layer.reset(random_float)

    def L2_biasoption(self, loss_list, sigma=None):
        if sigma == None:
            sigma = self._list_build()
        loss_bias = [torch.tensor(0.).float().cuda()]
        loss_feature = torch.tensor([0.]).float().cuda()
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) and layer.bias is not None:
                layer: nn.Conv2d
                loss_bias.append(torch.norm(torch.abs(layer.weight.data), p=2) / layer.weight.data.numel())
            # elif isinstance(layer, point_cul_Layer):
            #     layer: point_cul_Layer
            #     if hasattr(layer,"dis_loss"):
            #         loss_feature+=layer.dis_loss
        loss_feature = (loss_feature.squeeze(-1)) * sigma[0]
        loss_bias = torch.stack(loss_bias, dim=-1).mean() * sigma[1]
        loss_list = loss_list + [loss_bias, loss_feature]
        loss = torch.stack(loss_list, dim=-1).sum()
        return loss