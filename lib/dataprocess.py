import cv2 as cv
import numpy as np
import torch
import os,sys,math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import os.path as path
import PIL.Image as Image
import os.path

import torch
import scipy.io as scio
import torch.utils.data as data
import torchvision.transforms as transforms


class dataprocess(object):
    def __init__(self,dir,train_path=None,val_path=None,test_path=None):
        self.root=dir
        if dir==os.getcwd():
            if val_path == None:
                self.val_path = "val-car"
            else:
                self.val_path = val_path
            if train_path == None:
                self.train_path = "train-car"
            else:
                self.train_path = train_path
            if test_path == None:
                self.test_path = "test-car"
            else:
                self.test_path = test_path
        else:
            if val_path == None:
                self.val_path = os.path.join(dir,"val-car")
            else:
                self.val_path = val_path
            if train_path == None:
                self.train_path = os.path.join(dir,"train-car")
            else:
                self.train_path = train_path
            if test_path == None:
                self.test_path = os.path.join(dir,"test-car")
            else:
                self.test_path = test_path
        self.call_mode = "return_list"
    def convert_to_rgb(self,path):
        return Image.open(path).convert('RGB')
    def flist_read_iterator(self,file_list):
        imlist=[]
        with open(file_list,"r") as f:
            for line in f.readline():
                image_path=line.strip()
                imlist.append(image_path)
        return imlist
    def IMG_CAR_GET(self):
        image_format_list=[".jpg",".JPG",".png",".PNG","jpeg","JPEG",".ppm",".PPM",".bmp",".BMP"]
        def is_image_file(file_path):
            return any(file_path.endswith(img_format) for img_format in image_format_list)
        def list_data(mode="train"):
            if mode is not None:
                self.mode=mode
            image=[]
            if self.mode=="train":
                dir=self.train_path
            elif self.mode=="test":
                dir=self.test_path
            elif self.mode=="val":
                dir=self.val_path
            else:
                raise NotImplementedError("not import this mode!")
            dir=os.path.join(os.getcwd(),dir)
            print(dir)


            assert path.isdir(dir),"{} is not a valid image dir".format(dir)
            for root,dirs,fname in sorted(os.walk(dir)):
                for label_name in dirs:
                    new_dir=os.path.join(root,label_name)
                    for c_root,_,c_fname in sorted(os.walk(new_dir)):
                        for c_f in c_fname:
                            if is_image_file(c_f):
                                image_c_path=os.path.join(c_root,c_f)
                                total_name=image_c_path+","+label_name.strip()
                                image.append(total_name)
            return image
        self.train_list=list_data("train")
        self.val_list=list_data("val")
        for root,dirs,fname in sorted(os.walk(self.test_path)):
            test_dir_len=len(dirs)
        if test_dir_len==0:
            self.test_list=list_data("val")
        else:
            self.test_list = list_data("test")
    def __call__(self,mode="train"):
        tag=0
        if os.path.isfile(os.path.join(self.root,"car-train-list.txt")):
            tag+=1
        if os.path.isfile(os.path.join(self.root,"car-val-list.txt")):
            tag+=1
        if os.path.isfile(os.path.join(self.root,"car-test-list.txt")):
            tag+=1
        if tag==3:
            print("\ntrain data,val data,test data is alreadly!\n")
        else:
            print("\ncome to get the img list!\n")
        if tag==3:
            self.train_list=[]
            self.val_list=[]
            self.test_list=[]
            with open(os.path.join(self.root,"car-train-list.txt"),"r") as f:
                for line in f.readlines():
                    self.train_list.append(line.strip())
            with open(os.path.join(self.root,"car-val-list.txt"),"r") as f:
                for line in f.readlines():
                    self.val_list.append(line.strip())
            with open(os.path.join(self.root,"car-test-list.txt"),"r") as f:
                for line in f.readlines():
                    self.test_list.append(line.strip())
        else:
            self.IMG_CAR_GET()
        with open(os.path.join(self.root,"car-train-list.txt"),"w") as f:
            for line in self.train_list:
                f.writelines(line+"\n")
        with open(os.path.join(self.root,"car-val-list.txt"),"w") as f:
            for line in self.val_list:
                f.writelines(line+"\n")
        with open(os.path.join(self.root,"car-test-list.txt"),"w") as f:
            for line in self.test_list:
                f.writelines(line+"\n")
        if self.call_mode=="return_list":
            return self.train_list,self.val_list,self.test_list
        elif self.call_mode=="return_image":
            tag=0
            label_dict={}
            test_data = []
            test_label = []
            for path_add_label in self.test_list:
                path_add_label = path_add_label.strip()
                path, label = path_add_label.split(",")
                if label not in label_dict.keys():
                    now_tag = tag
                    label_dict[label] = tag
                    tag += 1
                else:
                    now_tag = label_dict[label]
                image = self.convert_to_rgb(path)
                test_data.append(image)
                test_label.append(now_tag)
            train_data = []
            train_label = []
            for path_add_label in self.train_list:
                path_add_label = path_add_label.strip()
                path, label = path_add_label.split(",")
                if label not in label_dict.keys():
                    now_tag = tag
                    label_dict[label] = tag
                    tag += 1
                else:
                    now_tag = label_dict[label]
                image = self.convert_to_rgb(path)
                train_data.append(image)
                train_label.append(now_tag)
            return train_data,train_label,test_data,test_label


        else:
            raise NotImplementedError
    def setmode(self,mode):
        self.call_mode=mode
class CarDateset(data.Dataset):
    def __init__(self,path,mode="return_image",tmp_size=72,result_size=64,use_transform=True,training=True):
        path_1=os.path.join(path,"/data/data/car")
        if os.path.isdir(path_1):
            print(path_1)
            sys.path.append(path_1)
        path_1=os.path.join(path,"/data/car")
        if os.path.isdir(path_1):
            print(path_1)
            sys.path.append(path_1)
        DATA=dataprocess(sys.path[-1])
        DATA.setmode(mode)
        self.training=training
        self.use_transform=use_transform
        self.train_data,self.train_label,self.test_data,self.test_label=DATA()
        import torchvision.transforms as transforms
        self.transform=transforms.Compose([
            transforms.Resize((tmp_size,tmp_size)),
        ])
        self.transform_test=transforms.Compose([
            transforms.Resize((result_size,result_size)),
        ])
        self.transform2=transforms.Compose([
            transforms.RandomSizedCrop((result_size, result_size), scale=(0.9, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(30, center=(result_size // 2, result_size // 2)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4455), (0.2023, 0.1994, 0.2010))
        ])
        self.transform2_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4455), (0.2023, 0.1994, 0.2010))
        ])
        if training == True:
            del self.test_label, self.test_data
            if use_transform==True:
                for i in range(len(self.train_data)):
                    self.train_data[i] = self.transform(self.train_data[i])
        else:
            del self.train_data, self.train_label
            if use_transform==True:
                for i in range(len(self.test_data)):
                    self.test_data[i] = self.transform_test(self.test_data[i])
    def __len__(self):
        if self.training==True:
            return len(self.train_label)
        else:
            return len(self.test_label)
    def __getitem__(self,index):
       if self.training==True:
           return self.transform2(self.train_data[index]),self.train_label[index]
       else:
           return self.transform2_test(self.test_data[index]), self.test_label[index]








def shuffle(len, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(0, len)
    np.random.shuffle(indices)
    return indices


class STLdataset(data.Dataset):
    def __init__(self, data, use_training=True, size=96, use_shuffle=True):
        self.data = data
        self.use_training = use_training
        self.transform = transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30, center=(size // 2, size // 2)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4455), (0.2023, 0.1994, 0.2010))
        ]) if use_training == True else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4455), (0.2023, 0.1994, 0.2010))
        ])
        if use_shuffle == True:
            rdnindex = shuffle(self.data[0].shape[0])
            self.data = (self.data[0][rdnindex, ...], self.data[1][rdnindex, ...])

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, index):
        data, label = self.data[0][index,...], torch.from_numpy(self.data[1][index, :]).squeeze().item()-1
        data = self.transform(Image.fromarray(data.transpose(1,2,0)))
        return data, label

class STLdataprocess(object):
    def __init__(self, path="/data/data/stl/stl10_matlab"):
        """
        shape : 5000,3,96,96 5000,1
        """
        self.train_path = os.path.join(path, "train.mat")
        self.test_path = os.path.join(path, "test.mat")
        self.train_data = scio.loadmat(self.train_path)
        self.test_data = scio.loadmat(self.test_path)
        self.train_data = (self.train_data["X"], self.train_data["y"])
        self.test_data = (self.test_data["X"], self.test_data["y"])
        self.train_data = (self.train_data[0].reshape(self.train_data[0].shape[0], 3, 96, 96), self.train_data[1])
        self.test_data = (self.test_data[0].reshape(self.test_data[0].shape[0], 3, 96, 96), self.test_data[1])

    def __call__(self, use_training=True):
        if use_training == True:
            dataset = STLdataset(self.train_data, use_training=use_training)
        else:
            dataset = STLdataset(self.test_data, use_training=use_training)
        return dataset




































































































































































































































































































































































































