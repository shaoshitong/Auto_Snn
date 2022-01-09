import certifi

from lib.loss_utils.arcface import *
from lib.loss_utils.center import *
from lib.loss_utils.triplet import *
from lib.loss_utils.smooth import *

class reid_cross_entropy(object):
    def __init__(self):
        pass
    def __call__(self,score,feat,target):
        return F.cross_entropy(score,target)
    def cuda(self,):
        return self

class reid_triplet(object):
    def __init__(self,triplet):
        self.triplet=triplet
    def __call__(self,score,feat,target):
        return self.triplet(feat,target)[0]
    def cuda(self,):
        return self

class reid_xent_triplet(object):
    def __init__(self,xent,triplet):
        self.xent=xent
        self.triplet=triplet
    def __call__(self,score,feat,target):
        return self.xent(score,target)+self.triplet(feat,target)[0]
    def cuda(self,):
        return self

class reid_cross_triplet(object):
    def __init__(self,cross,triplet):
        self.cross=cross
        self.triplet=triplet
    def __call__(self,score,feat,target):
        return self.cross(score,target)+self.triplet(feat,target)[0]
    def cuda(self,):
        return self

class reid_xent_triplet_center(object):
    def __init__(self,xent,triplet,config,center):
        self.xent=xent
        self.triplet=triplet
        self.weight=config.center_loss_weight
        self.center=center
    def __call__(self,score,feat,target):
        xent=self.xent(score, target)
        triplet=self.triplet(feat, target)[0]
        center=self.weight *self.center(feat, target)
        return xent+triplet+center*0.00001
    def cuda(self,):
        return self

class reid_cross_triplet_center(object):
    def __init__(self,cross,triplet,config,center):
        self.cross=cross
        self.triplet=triplet
        self.weight=config.center_loss_weight
        self.center=center
    def __call__(self,score,feat,target):
        return self.cross(score, target) + \
               self.triplet(feat, target)[0] + \
               self.weight * \
               self.center(feat, target)
    def cuda(self,):
        return self
def make_loss(config, num_classes,feed_dim):    # modified by gu

    if config['loss_type'] == 'triplet':
        sampler = config['sampler']
        triplet = TripletLoss(config.margin)  # triplet loss
        if config['label_smooth'] == 'on':
            xent = CrossEntropyLabelSmooth(
                num_classes=num_classes)     # new add by luo
            print("label smooth on, numclasses:", num_classes)

        if sampler == 'softmax':
            return reid_cross_entropy()
        elif config['sampler'] == 'triplet':
            return reid_triplet(triplet)
        elif config['sampler'] == 'softmax_triplet':
            if config['loss_type'] == 'triplet':
                if config['label_smooth'] == 'on':
                    return reid_xent_triplet(xent,triplet)
                else:
                    return reid_cross_triplet(F.cross_entropy,triplet)
        else:
            print('expected sampler should be softmax, triplet or softmax_triplet, '
                  'but got {}'.format(config.sampler))

    elif config['loss_type'] == 'triplet_center':
        center_criterion = CenterLoss(
            num_classes=num_classes, feat_dim=feed_dim, use_gpu=True)  # center loss
        triplet = TripletLoss(config.margin)  # triplet loss
        if config['label_smooth'] == 'on':
            xent = CrossEntropyLabelSmooth(
                num_classes=num_classes)     # new add by luo
            print("label smooth on, numclasses:", num_classes)
            return reid_xent_triplet_center(xent,triplet,config,center_criterion)
        else:
            cross = F.cross_entropy
            print("label smooth on, numclasses:", num_classes)
            return reid_cross_triplet_center(cross,triplet,config,center_criterion)
        """=======================reid=============================="""
    elif config['loss_type'] == 'cross':
        return torch.nn.CrossEntropyLoss()
    elif config['loss_type'] == 'smooth_cross':
        return smooth_crossentropy()
    elif config['loss_type'] == 'noonehot_cross':
        return crossentropy()
    else:
        raise NotImplementedError('expected METRIC_LOSS_TYPE should be triplet'
                            'but got {}'.format(config.loss_type))

def Loss_get(name="cross"):
    if name == "cross":
        return torch.nn.CrossEntropyLoss()
    elif name == "smooth_cross":
        return smooth_crossentropy()
    elif name=="noonehot_cross":
        return crossentropy()
    else:
        return None
