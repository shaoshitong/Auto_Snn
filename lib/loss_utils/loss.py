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

def make_loss(config, num_classes):    # modified by gu

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
    elif config['loss_type'] == 'cross':
        return torch.nn.CrossEntropyLoss()
    elif config['loss_type'] == 'smooth_cross':
        return smooth_crossentropy()
    elif config['loss_type'] == 'noonehot_cross':
        return crossentropy()
    else:
        raise NotImplementedError('expected METRIC_LOSS_TYPE should be triplet'
                            'but got {}'.format(config.loss_type))


def make_loss_with_center(config, num_classes):    # modified by gu
    if config.model_name == 'resnet18' or config.model_name == 'resnet34':
        feat_dim = 512

    elif config.model_name == 'efficientnet_v2':
        feat_dim = 1280

    else:
        feat_dim = 2048

    if config.loss_type == 'center':
        center_criterion = CenterLoss(
            num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif config.loss_type == 'triplet_center':
        triplet = TripletLoss(config.margin)  # triplet loss
        center_criterion = CenterLoss(
            num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(config.loss_type))

    if config.label_smooth == 'on':
        xent = CrossEntropyLabelSmooth(
            num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if config.loss_type == 'center':
            if config.label_smooth == 'on':
                return xent(score, target) + \
                    config.center_loss_weight * \
                    center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                    config.center_loss_weight * \
                    center_criterion(feat, target)

        elif config.loss_type == 'triplet_center':
            if config.label_smooth == 'on':
                return xent(score, target) + \
                    triplet(feat, target)[0] + \
                    config.center_loss_weight * \
                    center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                    triplet(feat, target)[0] + \
                    config.center_loss_weight * \
                    center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(config.loss_type))
    return loss_func, center_criterion


def Loss_get(name="cross"):
    if name == "cross":
        return torch.nn.CrossEntropyLoss()
    elif name == "smooth_cross":
        return smooth_crossentropy()
    elif name=="noonehot_cross":
        return crossentropy()
    else:
        return None
