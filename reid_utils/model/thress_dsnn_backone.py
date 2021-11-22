from lib.three_dsnn import merge_layer
from utils.getmemcpu import getMemCpu
import torch.nn as nn
import torch
import torch.nn.functional as F
class Backone(merge_layer):
    def __init__(self,device, shape=None, dropout=0.3):
        super(Backone, self).__init__(device,shape,dropout)

    def initiate_layer_reid(self, data, dataoption,num_classes, feature_list, size_list, hidden_size_list, path_nums_list,
                       nums_layer_list,down_rate,breadth_threshold, mult_k,drop_rate,neck,neck_feat):
        """
        配置相应的层
        """
        b, c, h, w = data.shape
        input_shape = (b, c, h, w)
        self.inf=nn.Sequential(*[nn.Conv2d(3, feature_list[0], kernel_size=(7,7), stride=(2,2), padding=3,bias=False),
                                 nn.BatchNorm2d(feature_list[0]),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
        self._initialize()
        h = self.InputGenerateNet.initiate_layer(data, feature_list, size_list, hidden_size_list, path_nums_list,
                                                 nums_layer_list, drop_rate,mult_k,down_rate,breadth_threshold)
        self.num_classes=num_classes
        self.neck=neck
        self.neck_feat=neck_feat
        self.bottleneck = nn.BatchNorm1d(h)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(
            h,self.num_classes, bias=False)
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.bottleneck.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)
    def weights_init_kaiming(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)
    def forward(self, x):
        with torch.no_grad():
            if hasattr(self, 'input_shape'):
                x = x.view(self.input_shape)
            else:
                pass
        x = self.inf(x)
        x = self.InputGenerateNet(x)
        x = self.gap(x)
        x = x.view(x.shape[0],x.shape[1])
        if self.neck=='no':
            feat=x
        elif self.neck=='bnneck':
            feat=self.bottleneck(x)
        else:
            raise NotImplementedError("Not Important Neck!")

        if feat.requires_grad==True:
            cls_score=self.classifier(feat)
            return cls_score,x
        else:
            if self.neck_feat=='after':
                return feat
            else:
                return x




