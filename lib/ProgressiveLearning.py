import math
import torch
import random
import numpy as np
import torchvision.datasets
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageEnhance, ImageOps


class ImageNetDataset(torchvision.datasets.ImageFolder):
    def __init__(self,root,transform):
        super(ImageNetDataset, self).__init__(root=root,transform=transform)
        self.nums=10
        self.beta=0.2
        self.trans=self.nums
        self.size_rate=1
    def reset_beta(self,beta,size_rate):
        self.nums=int((1-beta)*10)
        self.trans=beta
        self.size_rate=size_rate
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        label=torch.zeros(1000)
        label[target]=1
        if self.transform is not None:
            transform = Change_Compose_224(self.transform, self.trans, self.size_rate)
            sample = transform(sample)
        if  index > 0 and index % self.nums == 0:
            mixup_idx=random.randint(0,len(self.samples)-1)
            mixup_path, mixup_target = self.samples[mixup_idx]
            mixup_sample = self.loader(mixup_path)
            mixup_label = torch.zeros(1000)
            mixup_label[mixup_target] = 1
            if self.transform is not None:
                transform = Change_Compose_224(self.transform, self.trans, self.size_rate)
                mixup_sample = transform(mixup_sample)
            beta = self.beta
            lam = np.random.beta(beta, beta)
            sample = lam * sample + (1 - lam) * mixup_sample
            label = lam * label + (1 - lam) * mixup_label
        return sample, label
class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self,root, train,download, transform):
        super(CIFAR10Dataset, self).__init__(root=root,train=train,download=download,transform=transform)
        self.nums=10
        self.beta=0.2
        self.trans=self.nums
        self.size_rate=1
    def reset_beta(self,beta,size_rate):
        self.nums=int((1-beta)*10)
        self.trans=beta
        self.size_rate=size_rate
    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        label=torch.zeros(10)
        label[self.targets[idx]]=1
        if self.transform:
            transform=Change_Compose_32(self.transform,self.trans,self.size_rate)
            image=transform(image)
        if self.train and idx>0 and idx%self.nums==0:
            mixup_idx=random.randint(0,len(self.data)-1)
            mixup_image, mixup_target = self.data[mixup_idx], self.targets[mixup_idx]
            mixup_image = Image.fromarray(mixup_image)

            mixup_label=torch.zeros(10)
            mixup_label[self.targets[mixup_idx]]=1
            if self.transform:
                transform = Change_Compose_32(self.transform, self.trans,self.size_rate)
                mixup_image=transform(mixup_image)
            beta=self.beta
            lam=np.random.beta(beta,beta)
            image=lam*image+(1-lam)*mixup_image
            label=lam*label+(1-lam)*mixup_label
        return image,label


class CIFAR100Dataset(torchvision.datasets.CIFAR100):
    def __init__(self,root, train,download, transform):
        super(CIFAR100Dataset, self).__init__(root=root,train=train,download=download,transform=transform)
        self.nums=10
        self.beta=0.2
        self.trans=self.nums
        self.size_rate=1
    def reset_beta(self,beta,size_rate):
        self.nums=int((1-beta)*10)
        self.trans=beta
        self.size_rate=size_rate
    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)

        label=torch.zeros(100)
        label[self.targets[idx]]=1
        if self.transform:
            transform=Change_Compose_32(self.transform,self.trans,self.size_rate)
            image=transform(image)
        if self.train and idx>0 and idx%self.nums==0:
            mixup_idx=random.randint(0,len(self.data)-1)
            mixup_image, mixup_target = self.data[mixup_idx], self.targets[mixup_idx]
            mixup_image = Image.fromarray(mixup_image)

            mixup_label=torch.zeros(100)
            mixup_label[self.targets[mixup_idx]]=1
            if self.transform:
                transform = Change_Compose_32(self.transform, self.trans,self.size_rate)
                mixup_image=transform(mixup_image)
            beta=self.beta
            lam=np.random.beta(beta,beta)
            image=lam*image+(1-lam)*mixup_image
            label=lam*label+(1-lam)*mixup_label
        return image,label

def Change_Compose_32(compose:torchvision.transforms.Compose,p,size_rate):
    z=torchvision.transforms.Compose([])
    for i in range(len(compose.transforms)):
        if isinstance(compose.transforms[i], torchvision.transforms.Resize):
            z.transforms.append(torchvision.transforms.Resize(int(32 * size_rate)))
        elif isinstance(compose.transforms[i],RandomErasing):
            z.transforms.append(torchvision.transforms.RandomApply([compose.transforms[i]],p))
        elif isinstance(compose.transforms[i],ToTensor) or isinstance(compose.transforms[i],Normalize):
            z.transforms.append(compose.transforms[i])
        else:
            z.transforms.append(compose.transforms[i])
    return z



def Change_Compose_224(compose:torchvision.transforms.Compose,p,size_rate):
    z=torchvision.transforms.Compose([])
    for i in range(len(compose.transforms)):
        if isinstance(compose.transforms[i], torchvision.transforms.Resize):
            z.transforms.append(torchvision.transforms.Resize(int(224 * size_rate)))
        elif isinstance(compose.transforms[i],AutoAugment):
            z.transforms.append(torchvision.transforms.RandomApply([compose.transforms[i]],p))
        elif isinstance(compose.transforms[i],RandomErasing):
            z.transforms.append(torchvision.transforms.RandomApply([compose.transforms[i]],p))
        elif isinstance(compose.transforms[i],ToTensor) or isinstance(compose.transforms[i],Normalize):
            z.transforms.append(compose.transforms[i])
        else:
            z.transforms.append(compose.transforms[i])
    return z




class RandomErasing(torch.nn.Module):
    """ Randomly selects a rectangle region in an image and erases its pixels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        super(RandomErasing, self).__init__()
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def forward(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img



class Cutout:

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        ''' img: Tensor image of size (C, H, W) '''
        _, h, w = img.size()
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = int(np.clip(y - self.length // 2, 0, h))
            y2 = int(np.clip(y + self.length // 2, 0, h))
            x1 = int(np.clip(x - self.length // 2, 0, w))
            x2 = int(np.clip(x + self.length // 2, 0, w))
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


class ImageNetPolicy:
    ''' Randomly choose one of the best 25 Sub-policies on ImageNet. '''
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, 'posterize', 8, 0.6, 'rotate', 9, fillcolor),
            SubPolicy(0.6, 'solarize', 5, 0.6, 'autocontrast', 5, fillcolor),
            SubPolicy(0.8, 'equalize', 8, 0.6, 'equalize', 3, fillcolor),
            SubPolicy(0.6, 'posterize', 7, 0.6, 'posterize', 6, fillcolor),
            SubPolicy(0.4, 'equalize', 7, 0.2, 'solarize', 4, fillcolor),

            SubPolicy(0.4, 'equalize', 4, 0.8, 'rotate', 8, fillcolor),
            SubPolicy(0.6, 'solarize', 3, 0.6, 'equalize', 7, fillcolor),
            SubPolicy(0.8, 'posterize', 5, 1.0, 'equalize', 2, fillcolor),
            SubPolicy(0.2, 'rotate', 3, 0.6, 'solarize', 8, fillcolor),
            SubPolicy(0.6, 'equalize', 8, 0.4, 'posterize', 6, fillcolor),

            SubPolicy(0.8, 'rotate', 8, 0.4, 'color', 0, fillcolor),
            SubPolicy(0.4, 'rotate', 9, 0.6, 'equalize', 2, fillcolor),
            SubPolicy(0.0, 'equalize', 7, 0.8, 'equalize', 8, fillcolor),
            SubPolicy(0.6, 'invert', 4, 1.0, 'equalize', 8, fillcolor),
            SubPolicy(0.6, 'color', 4, 1.0, 'contrast', 8, fillcolor),

            SubPolicy(0.8, 'rotate', 8, 1.0, 'color', 2, fillcolor),
            SubPolicy(0.8, 'color', 8, 0.8, 'solarize', 7, fillcolor),
            SubPolicy(0.4, 'sharpness', 7, 0.6, 'invert', 8, fillcolor),
            SubPolicy(0.6, 'shearX', 5, 1.0, 'equalize', 9, fillcolor),
            SubPolicy(0.4, 'color', 0, 0.6, 'equalize', 3, fillcolor),

            SubPolicy(0.4, 'equalize', 7, 0.2, 'solarize', 4, fillcolor),
            SubPolicy(0.6, 'solarize', 5, 0.6, 'autocontrast', 5, fillcolor),
            SubPolicy(0.6, 'invert', 4, 1.0, 'equalize', 8, fillcolor),
            SubPolicy(0.6, 'color', 4, 1.0, 'contrast', 8, fillcolor),
            SubPolicy(0.8, 'equalize', 8, 0.6, 'equalize', 3, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return 'AutoAugment ImageNet Policy'


class CIFAR10Policy:
    ''' Randomly choose one of the best 25 Sub-policies on CIFAR10. '''
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, 'invert', 7, 0.2, 'contrast', 6, fillcolor),
            SubPolicy(0.7, 'rotate', 2, 0.3, 'translateX', 9, fillcolor),
            SubPolicy(0.8, 'sharpness', 1, 0.9, 'sharpness', 3, fillcolor),
            SubPolicy(0.5, 'shearY', 8, 0.7, 'translateY', 9, fillcolor),
            SubPolicy(0.5, 'autocontrast', 8, 0.9, 'equalize', 2, fillcolor),

            SubPolicy(0.2, 'shearY', 7, 0.3, 'posterize', 7, fillcolor),
            SubPolicy(0.4, 'color', 3, 0.6, 'brightness', 7, fillcolor),
            SubPolicy(0.3, 'sharpness', 9, 0.7, 'brightness', 9, fillcolor),
            SubPolicy(0.6, 'equalize', 5, 0.5, 'equalize', 1, fillcolor),
            SubPolicy(0.6, 'contrast', 7, 0.6, 'sharpness', 5, fillcolor),

            SubPolicy(0.7, 'color', 7, 0.5, 'translateX', 8, fillcolor),
            SubPolicy(0.3, 'equalize', 7, 0.4, 'autocontrast', 8, fillcolor),
            SubPolicy(0.4, 'translateY', 3, 0.2, 'sharpness', 6, fillcolor),
            SubPolicy(0.9, 'brightness', 6, 0.2, 'color', 8, fillcolor),
            SubPolicy(0.5, 'solarize', 2, 0.0, 'invert', 3, fillcolor),

            SubPolicy(0.2, 'equalize', 0, 0.6, 'autocontrast', 0, fillcolor),
            SubPolicy(0.2, 'equalize', 8, 0.6, 'equalize', 4, fillcolor),
            SubPolicy(0.9, 'color', 9, 0.6, 'equalize', 6, fillcolor),
            SubPolicy(0.8, 'autocontrast', 4, 0.2, 'solarize', 8, fillcolor),
            SubPolicy(0.1, 'brightness', 3, 0.7, 'color', 0, fillcolor),

            SubPolicy(0.4, 'solarize', 5, 0.9, 'autocontrast', 3, fillcolor),
            SubPolicy(0.9, 'translateY', 9, 0.7, 'translateY', 9, fillcolor),
            SubPolicy(0.9, 'autocontrast', 2, 0.8, 'solarize', 3, fillcolor),
            SubPolicy(0.8, 'equalize', 8, 0.1, 'invert', 3, fillcolor),
            SubPolicy(0.7, 'translateY', 9, 0.9, 'autocontrast', 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return 'AutoAugment CIFAR10 Policy'


class SVHNPolicy:
    ''' Randomly choose one of the best 25 Sub-policies on SVHN. '''
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, 'shearX', 4, 0.2, 'invert', 3, fillcolor),
            SubPolicy(0.9, 'shearY', 8, 0.7, 'invert', 5, fillcolor),
            SubPolicy(0.6, 'equalize', 5, 0.6, 'solarize', 6, fillcolor),
            SubPolicy(0.9, 'invert', 3, 0.6, 'equalize', 3, fillcolor),
            SubPolicy(0.6, 'equalize', 1, 0.9, 'rotate', 3, fillcolor),

            SubPolicy(0.9, 'shearX', 4, 0.8, 'autocontrast', 3, fillcolor),
            SubPolicy(0.9, 'shearY', 8, 0.4, 'invert', 5, fillcolor),
            SubPolicy(0.9, 'shearY', 5, 0.2, 'solarize', 6, fillcolor),
            SubPolicy(0.9, 'invert', 6, 0.8, 'autocontrast', 1, fillcolor),
            SubPolicy(0.6, 'equalize', 3, 0.9, 'rotate', 3, fillcolor),

            SubPolicy(0.9, 'shearX', 4, 0.3, 'solarize', 3, fillcolor),
            SubPolicy(0.8, 'shearY', 8, 0.7, 'invert', 4, fillcolor),
            SubPolicy(0.9, 'equalize', 5, 0.6, 'translateY', 6, fillcolor),
            SubPolicy(0.9, 'invert', 4, 0.6, 'equalize', 7, fillcolor),
            SubPolicy(0.3, 'contrast', 3, 0.8, 'rotate', 4, fillcolor),

            SubPolicy(0.8, 'invert', 5, 0.0, 'translateY', 2, fillcolor),
            SubPolicy(0.7, 'shearY', 6, 0.4, 'solarize', 8, fillcolor),
            SubPolicy(0.6, 'invert', 4, 0.8, 'rotate', 4, fillcolor),
            SubPolicy(0.3, 'shearY', 7, 0.9, 'translateX', 3, fillcolor),
            SubPolicy(0.1, 'shearX', 6, 0.6, 'invert', 5, fillcolor),

            SubPolicy(0.7, 'solarize', 2, 0.6, 'translateY', 7, fillcolor),
            SubPolicy(0.8, 'shearY', 4, 0.8, 'invert', 8, fillcolor),
            SubPolicy(0.7, 'shearX', 9, 0.8, 'translateY', 3, fillcolor),
            SubPolicy(0.8, 'shearY', 5, 0.7, 'autocontrast', 3, fillcolor),
            SubPolicy(0.7, 'shearX', 2, 0.1, 'invert', 5, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return 'AutoAugment SVHN Policy'


class SubPolicy:

    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            'shearX': np.linspace(0, 0.3, 10),
            'shearY': np.linspace(0, 0.3, 10),
            'translateX': np.linspace(0, 150 / 331, 10),
            'translateY': np.linspace(0, 150 / 331, 10),
            'rotate': np.linspace(0, 30, 10),
            'color': np.linspace(0.0, 0.9, 10),
            'posterize': np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            'solarize': np.linspace(256, 0, 10),
            'contrast': np.linspace(0.0, 0.9, 10),
            'sharpness': np.linspace(0.0, 0.9, 10),
            'brightness': np.linspace(0.0, 0.9, 10),
            'autocontrast': [0] * 10,
            'equalize': [0] * 10,
            'invert': [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert('RGBA').rotate(magnitude)
            return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            'shearX': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            'shearY': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            'translateX': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            'translateY': lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            'rotate': lambda img, magnitude: rotate_with_fill(img, magnitude),
            'color': lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            'posterize': lambda img, magnitude: ImageOps.posterize(img, magnitude),
            'solarize': lambda img, magnitude: ImageOps.solarize(img, magnitude),
            'contrast': lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            'sharpness': lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            'brightness': lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            'autocontrast': lambda img, magnitude: ImageOps.autocontrast(img),
            'equalize': lambda img, magnitude: ImageOps.equalize(img),
            'invert': lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img