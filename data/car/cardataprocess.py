import cv2 as cv
import numpy as np
import torch
import os,sys,math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import os.path as path
import PIL.Image as Image
class dataprocess(object):
    def __init__(self,train_path=None,val_path=None,test_path=None):
        if val_path==None:
            self.val_path="val-car"
        else:
            self.val_path=val_path
        if train_path==None:
            self.train_path="train-car"
        else:
            self.train_path=train_path
        if test_path==None:
            self.test_path="test-car"
        else:
            self.test_path=test_path
        self.call_mode="return_list"
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
    def __call__(self):
        tag=0
        if os.path.isfile("car-train-list.txt"):
            tag+=1
        if os.path.isfile("car-val-list.txt"):
            tag+=1
        if os.path.isfile("car-test-list.txt"):
            tag+=1
        print(tag)
        if tag==3:
            self.train_list=[]
            self.val_list=[]
            self.test_list=[]
            with open("car-train-list.txt","r") as f:
                for line in f.readlines():
                    self.train_list.append(line.strip())
            with open("car-val-list.txt","r") as f:
                for line in f.readlines():
                    self.val_list.append(line.strip())
            with open("car-test-list.txt","r") as f:
                for line in f.readlines():
                    self.test_list.append(line.strip())
        else:
            self.IMG_CAR_GET()
        with open("car-train-list.txt","w") as f:
            for line in self.train_list:
                f.writelines(line+"\n")
        with open("car-val-list.txt","w") as f:
            for line in self.val_list:
                f.writelines(line+"\n")
        with open("car-test-list.txt","w") as f:
            for line in self.test_list:
                f.writelines(line+"\n")
        if self.call_mode=="return_list":
            return self.train_list,self.val_list,self.test_list
        elif self.call_mode=="return_image":
            label_dict={}
            tag=0
            val_data = []
            val_label = []
            for path_add_label in self.val_list:
                path_add_label=path_add_label.strip()
                path,label=path_add_label.split(",")
                if label not in label_dict.keys():
                    now_tag=tag
                    label_dict[label]=tag
                    tag+=1
                else:
                    now_tag=label_dict[label]
                image=self.convert_to_rgb(path)
                val_data.append(image)
                val_label.append(now_tag)
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
            return train_data,train_label,val_data,val_label,test_data,test_label


        else:
            raise NotImplementedError
    def setmode(self,mode):
        self.call_mode=mode
class CarDateset(data.Dataset):
    def __init__(self,path,mode="return_image",tmp_size=72,result_size=64,use_transform=True,training=True):
        path=os.path.join(path,"../data")
        if os.path.isdir(path):
            sys.path.append(path)
        path=os.path.join(path,"data")
        if os.path.isdir(path):
            sys.path.append(path)
        DATA=dataprocess()
        DATA.setmode(mode)
        self.training=training
        self.use_transform=use_transform
        self.train_data,self.train_label,self.val_data,self.val_label,self.test_data,self.test_label=DATA()
        import torchvision.transforms as transforms
        self.transform=transforms.Compose([
            transforms.Resize((tmp_size,tmp_size)),
            transforms.RandomSizedCrop((result_size,result_size),scale=(0.9,1.0),ratio=(0.85,1.15)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4455),(0.2023,0.1994,0.2010))
        ])
        self.transform_img=transforms.Compose([
            transforms.Resize((tmp_size,tmp_size)),
            transforms.RandomSizedCrop((result_size,result_size),scale=(0.9,1.0),ratio=(0.85,1.15)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ])
    def __len__(self):
        if self.training==True:
            return len(self.train_label)
        else:
            return len(self.test_label)
    def __getitem__(self,index):
        # data_t=self.transform_img(self.train_data[index])
        # import matplotlib.pyplot as plt
        # plt.imshow(data_t,"viridis")
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis("off")
        # plt.show()
        if self.training==False:
            data,label=self.test_data[index],self.test_label[index]
        elif self.training==True and self.use_transform==False:
            data,label=self.train_data[index],self.train_label[index]
        else:
            data,label=self.train_data[index],self.train_label[index]
            data=self.transform(data)

        return data,label











































































































































































































































































































































































































