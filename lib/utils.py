import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Interpolate(nn.Module):
    def __init__(self,in_channel,out_channel,stride):
        super(Interpolate, self).__init__()
        assert stride>=1
        self.stride=stride
        if in_channel!=out_channel:
            self.turn_channel=nn.Conv2d(in_channel,out_channel,(1,1),(1,1),(0,0),bias=False)
        self.in_channel,self.out_channel=in_channel,out_channel
    def forward(self,x):
        if self.in_channel!=self.out_channel:
            x=self.turn_channel(x)
        x=F.interpolate(x,scale_factor=self.stride,mode="bilinear",align_corners=False)
        return x


class None_Do(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(None_Do, self).__init__()
        if in_channel != out_channel:
            self.turn_channel = nn.Conv2d(in_channel, out_channel, (1, 1), (1, 1), (0, 0), bias=False)
        self.in_channel, self.out_channel = in_channel, out_channel
    def forward(self,x):
        if self.in_channel != self.out_channel:
            x = self.turn_channel(x)
        return x

def tensor_list_add(tensor_list):
    if len(tensor_list)==0:
        return None
    a= 0. if tensor_list[0]==None else tensor_list[0]
    for tensor in tensor_list[1:]:
        a+=tensor
    a=None if isinstance(a,float) else a
    return a
class Multi_Fusion(nn.Module):
    def __init__(self,size_list,feature_list_in,feature_list_out,index):
        super(Multi_Fusion,self).__init__()
        """
        for example:
        
            size_list=[32,16,8,4]
            feature_list=[32,32,32,32]
            
            so the input is a list:
            [Tensor(32,32),Tensor(16,16),Tensor(8,8),Tensor(4,4)]
        """
        assert len(size_list)==len(feature_list_out) and len(feature_list_in)==len(feature_list_out)
        self.size_list=size_list
        self.total_model=nn.ModuleList([])
        self.feature_list_in=feature_list_in
        self.in_norm=nn.ModuleList([
            nn.Sequential(nn.BatchNorm2d(feature_list_in[i]),
                          nn.ReLU(inplace=False)) for i in range(len(feature_list_in))
        ])
        for i,out_size in enumerate(self.size_list):
            mode=nn.ModuleList([])
            for j,in_size in enumerate(self.size_list):
                stride=int(out_size//in_size)
                if stride==1:
                    mode.append(None_Do(feature_list_in[j],feature_list_out[i]))
                elif stride<1:
                    stride=int(in_size//out_size)
                    mode.append(nn.Conv2d(feature_list_in[j],feature_list_out[i],(stride,stride),(stride,stride),(0,0),bias=False))
                elif stride>1:
                    mode.append(Interpolate(feature_list_in[j],feature_list_out[i],stride))
            self.total_model.append(mode)
        self.index=index
        self.np_last=feature_list_out
    def forward(self,inputs):
        index=self.index
        assert isinstance(inputs,list) or isinstance(inputs,tuple)
        return_list=[]
        for i,ind in enumerate(index):
            inputs[ind] = self.in_norm[i](inputs[ind])
        for i,in_size in enumerate(self.size_list):
            out_list=[]
            for j,out_size in enumerate(self.size_list):
                out_list.append(self.total_model[i][j](inputs[index[j]]))
            return_list.append(tensor_list_add(out_list))
        for i,ind in enumerate(index):
            inputs[ind]=return_list[i]

        return inputs


class Multi_Fusion_Create(nn.Module):
    def __init__(self,in_channels,channnels,in_size,size_list):
        super(Multi_Fusion_Create, self).__init__()
        self.conv=nn.ModuleList([
            nn.Conv2d(in_channels,channnels,(3,3),(int(in_size//size),int(in_size//size)),(1,1),bias=False) for size in size_list
        ])
        self.np_last=[channnels]*len(size_list)
    def forward(self,x):
        return_list=[]
        for conv in self.conv:
            return_list.append(conv(x))
        return return_list

def filling_data(list_i,l):
    while len(list_i)<l:
        list_i.insert(0,None)
    return list_i
def balance_ilist(feature_list,path_nums_list,nums_layer,breadth_threshold):
    L=max(len(h) for h in feature_list)
    for i in range(len(feature_list)):
        if len(feature_list[i])<L:
            feature_list[i]=filling_data(feature_list[i],L)

    for i in range(len(path_nums_list)):
        if len(path_nums_list[i])<L:
            path_nums_list[i]=filling_data(path_nums_list[i],L)

    for i in range(len(nums_layer)):
        if len(nums_layer[i])<L:
            nums_layer[i]=filling_data(nums_layer[i],L)

    for i in range(len(breadth_threshold)):
        if len(breadth_threshold[i])<L:
            breadth_threshold[i]=filling_data(breadth_threshold[i],L)
    return feature_list,path_nums_list,nums_layer,breadth_threshold
def decay_rate_list(feature_list,decay_rate,index):
    _feature_list=copy.deepcopy(feature_list)
    for i in range(len(feature_list)):
        if i in index:
            _feature_list[i]=int(feature_list[i]//decay_rate)
    return _feature_list
def filter_list(a,b):
    c,d,i=[],[],[]
    iter=0
    for n,m in zip(a,b):
        if m!=None:
            c.append(n)
            d.append(m)
            i.append(iter)
        iter+=1
    return c,d,i