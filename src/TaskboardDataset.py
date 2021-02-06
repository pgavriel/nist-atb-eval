#!/usr/bin/env python
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

# Dataset Definition
class TaskboardDataset(Dataset):
    def __init__(self,data_root,component=None,transform=None):
        self.annotations = []
        self.transform = transform
        if component is not None:
            c_folder = None
            for f in os.listdir(data_root):
                if f.startswith("{:02d}".format(component)):
                    c_folder = f
                    print("Found folder {}".format(f))
                    break
            self.data_path = os.path.join(data_root,c_folder)
        else:
            self.data_path = data_root
        print("Dataset Path: {}".format(self.data_path))


        #print(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path,self.annotations[idx][0])
        image = Image.open(img_path)
        label = torch.tensor(int(self.annotations[idx][1]))

        if self.transform:
            image = self.transform(image)

        #print("Dataset Get ID: {}  File: {}  Xform: {}".format(idx,self.annotations[idx][0],self.transform))
        return (image, label)

    def addAll(self):
        for f in os.listdir(self.data_path):
            if f.endswith(".png"):
                self.annotations.append([f,f[0]])

    def add(self,file,label=None):
        if label is None:
            label = file[0]
        label = str(label)
        self.annotations.append([file,label])
