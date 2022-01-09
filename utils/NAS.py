import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# 3,4,5,6  4626432.0
class NAS:
    def __init__(self, yaml):
        self.yaml = yaml
        self.x = [2, 3, 4, 5, 6, 7, 8, 9]
        self.y = [3, 23, 78, 192, 395, 723, 1218, 1928]
        self.p = [10484520.1, 4626432.1]
        self.feature_map_list=yaml["feature_list"][1:]
        self.size_list=yaml["size_list"][1:]
        self.input_feature=yaml["feature_list"][0]
        self.psi_list=yaml["path_nums_list"]
        self.calcuate_num=self.calcuate(self.input_feature,self.feature_map_list,self.size_list,self.psi_list)
        self.min_num=max(0.,self.calcuate_num-1000000)
        self.max_num=min(self.p[0],self.calcuate_num+1000000)
        self.initialize_choose_list()
    def initialize_choose_list(self):
        psi_num=np.arange(2,10,1)
        feature_map_num=np.arange(24,51,1)
        input_feature=np.arange(48,81,1)
        choose_list=[]
        for i in psi_num:
            for j in feature_map_num:
                for k in input_feature:
                    choose_list.append((i,j,k))
        self.choose_list=choose_list
    def create_like_input(self,size_list,nums=5000):
        l=len(self.choose_list)
        set_num=set()
        np.random.shuffle(self.choose_list)
        choose_index=np.random.randint(0,l,(1000,len(size_list)),dtype=int)
        chould_choose=[]
        for i in range(choose_index.shape[0]):
            z=choose_index[i].tolist()
            feature_map_list=[]
            psi_list=[]
            for j in z:
                psi,feature_map,input=self.choose_list[j]
                feature_map_list.append(feature_map)
                psi_list.append(psi)
            if psi_list[0]>=6:
                continue
            num=self.calcuate(input,feature_map_list,self.size_list,psi_list)
            str_num=str(input)+str(feature_map_list)+str(psi_list)
            if num>=self.min_num and num<=self.max_num and str_num not in set_num:
                set_num.add(str_num)
                chould_choose.append({"input":input,"path_nums_list":psi_list,"size_list":self.size_list,"feature_list":feature_map_list})
        return chould_choose[:nums]
    def calcuate(self, input_feature, feature_map_list, size_list, psi_list):
        sum = 0.
        p = input_feature
        for feature_map, size, psi in zip(feature_map_list, size_list, psi_list):
            if isinstance(size,int):
                size=(size,size)
            sum += self.y[psi - 2] * feature_map + p * (psi ** 2 - 1) * size[0] * size[1]
            p = p + (psi ** 2 - 1) * feature_map
        return sum