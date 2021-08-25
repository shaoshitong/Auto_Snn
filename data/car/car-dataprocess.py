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

        self.IMG_CAR_GET()
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
DATA=dataprocess()
DATA.setmode("return_image")
print(DATA())





















































































\




















































































































































































































































































































