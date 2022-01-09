import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class PointConv(nn.Module):
    def __init__(self, kernel_size, in_feature, in_point, out_point, padding="same", stride=None,
                 grouping=None, bias=True):
        super(PointConv, self).__init__()
        self.in_feature = in_feature
        self.in_point = in_point
        self.out_point = out_point
        self.kernel_size = kernel_size
        if stride == None:
            self.stride = kernel_size
        else:
            self.stride = stride
        if padding == "same":
            k_h, k_w = kernel_size
            in_h, in_w = in_point
            out_h, out_w = out_point
            if out_h % self.kernel_size[0] == 0:
                r_h = 0
            else:
                r_h = 1
            if out_w % self.kernel_size[1] == 0:
                r_w = 0
            else:
                r_w = 1
            self.pad_h = (k_h + self.stride[0] * (out_h // self.kernel_size[0] + r_h - 1) - in_h) // 2 + 0.5 if (k_h +
                                                                                                                 self.stride[
                                                                                                                     0] * (
                                                                                                                         out_h //
                                                                                                                         self.kernel_size[
                                                                                                                             0] + r_h - 1) - in_h) % 2 != 0 else 0
            self.pad_w = (k_w + self.stride[1] * (out_w // self.kernel_size[1] + r_w - 1) - in_w) // 2 + 0.5 if (k_w +
                                                                                                                 self.stride[
                                                                                                                     1] * (
                                                                                                                         out_w //
                                                                                                                         self.kernel_size[
                                                                                                                             1] + r_w - 1) - in_w) % 2 != 0 else 0
            assert (self.pad_h >= 0 and self.pad_w >= 0)
        else:
            self.pad_h, self.pad_w = padding
        if grouping == None:
            self.grouping = in_feature
        else:
            self.grouping = grouping
        self.bias = bias
        self.usepad = nn.ReflectionPad2d((int(self.pad_h) + int(-2 * (self.pad_h // 1 - self.pad_h)), int(self.pad_h),
                                          int(self.pad_w) + int(-2 * (self.pad_w // 1 - self.pad_w)), int(self.pad_w)))
        self.unfold = lambda image: F.unfold(image, self.kernel_size,
                                             stride=self.stride, )  # [B, C* kH * kW, L]
        assert (((self.in_point[0] + self.pad_h * 2 - self.kernel_size[0]) // self.stride[0] + 1) * self.kernel_size[
            0] == self.out_point[0] and
                ((self.in_point[1] + self.pad_w * 2 - self.kernel_size[1]) // self.stride[1] + 1) * self.kernel_size[
                    1] == self.out_point[1])
        self.L = int(((self.in_point[0] + self.pad_h * 2 - self.kernel_size[0]) // self.stride[0] + 1) * (
                (self.in_point[1] + self.pad_w * 2 - self.kernel_size[1]) // self.stride[1] + 1))
        # self.weight = []
        # for i in range(self.L):
        #     weight = Parameter(
        #         torch.Tensor( self.kernel_size[0] * self.kernel_size[1],self.kernel_size[0] * self.kernel_size[1]),
        #         requires_grad=True)
        #     stdv = 6. / math.sqrt(weight.data.numel())
        #     weight.data.uniform_(-stdv, stdv)
        #     self.weight.append(weight)
        # if self.bias == True:
        #     self.bias = []
        #     for i in range(self.L):
        #         bias = Parameter(torch.zeros(1, 1, self.kernel_size[0] * self.kernel_size[1]), requires_grad=True)
        #         self.bias.append(bias)
        # else:
        #     self.bias = []
        # self.weight=nn.ParameterList(self.weight)
        # self.bias=nn.ParameterList(self.bias)

        self.weight = Parameter(
            torch.Tensor(self.L,self.kernel_size[0] * self.kernel_size[1]*self.grouping, self.kernel_size[0] * self.kernel_size[1]*self.grouping),
            requires_grad=True)
        stdv = 6. / math.sqrt(self.weight.data.numel()//self.weight.data.shape[0])
        self.weight.data.uniform_(-stdv, stdv)
        self.bias = Parameter(torch.zeros(1, 1, self.kernel_size[0] * self.kernel_size[1]*self.grouping,self.L), requires_grad=True)

    def add_pad(self, x):
        return self.usepad(x)

    def reshape(self, patch_image):
        B, C_kh_kw, L = patch_image.size()
        assert self.in_feature%self.grouping==0
        patch_image = patch_image.view(B, self.in_feature//self.grouping, -1, L)
        return patch_image

    def patch_image_conv(self, patch_image_tuple):
        B,C,H,L=patch_image_tuple.size()
        m=(torch.einsum("bcnl,lnp->bcpl",patch_image_tuple,self.weight)+self.bias).view(B,-1,L)
        r=F.fold(m,self.out_point,self.kernel_size,stride=self.kernel_size)

        return r

    def forward(self, x):
        x = self.add_pad(x)
        patch_image = self.unfold(x)
        patch_image = self.reshape(patch_image)
        return self.patch_image_conv(patch_image)
#
# c = PointConv((4, 4), 10, (8, 8), (8, 8))
# X=torch.tensor([[1,0,1,1,1,0,0,1],
#                           [2,3,4,5,6,7,8,9],
#                           [3,2,1,0,-1,-2,-3,-4],
#                           [1,2,3,4,5,6,7,8],
#                           [0,0,0,0,0,0,0,0],
#                           [1,1,1,1,1,1,1,1],
#                           [2,2,2,2,2,2,2,2],
#                           [3,3,3,3,3,3,3,3]],dtype=torch.float32)
# X=X.unsqueeze(0).unsqueeze(0).repeat(1,10,1,1)
# print(c(X))
