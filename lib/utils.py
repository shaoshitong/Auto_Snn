import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self,size_list,feature_list):
        super(Multi_Fusion,self).__init__()
        """
        for example:
        
            size_list=[32,16,8,4]
            feature_list=[32,32,32,32]
            
            so the input is a list:
            [Tensor(32,32),Tensor(16,16),Tensor(8,8),Tensor(4,4)]
        """
        assert len(size_list)==len(feature_list)
        self.size_list=size_list
        self.feature_list=feature_list
        self.total_model=nn.ModuleList([])
        self.in_norm=nn.ModuleList([
            nn.Sequential(nn.BatchNorm2d(feature_list[i]),
                          nn.LeakyReLU(inplace=True)) for i in range(len(self.feature_list))
        ])
        for i,out_size in enumerate(self.size_list):
            mode=nn.ModuleList([])
            for j,in_size in enumerate(self.size_list):
                stride=int(out_size//in_size)
                if stride==1:
                    mode.append(None_Do(self.feature_list[i],self.feature_list[j]))
                elif stride<1:
                    stride=int(in_size//out_size)
                    mode.append(nn.Conv2d(self.feature_list[i],self.feature_list[j],(stride,stride),(stride,stride),(0,0),bias=False))
                elif stride>1:
                    mode.append(Interpolate(self.feature_list[i],self.feature_list[j],stride))
            self.total_model.append(mode)
    def forward(self,inputs):
        assert isinstance(inputs,list) or isinstance(inputs,tuple)
        return_list=[]

        for i in range(len(inputs)):
            inputs[i]=self.in_norm[i](inputs[i])

        for i,in_size in enumerate(self.size_list):
            out_list=[]
            for j,out_size in enumerate(self.size_list):
                out_list.append(self.total_model[i][j](inputs[j]))
            return_list.append(tensor_list_add(out_list))

        return return_list




