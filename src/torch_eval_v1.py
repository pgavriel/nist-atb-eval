#!/usr/bin/env python
import os, sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import rospy
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Set Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

# Dataset Definition
class TaskboardDataset(Dataset):
    def __init__(self,data_root,component,transform=None):
        self.annotations = []
        self.transform = transform
        c_folder = None
        for f in os.listdir(data_root):
            if f.startswith("{:02d}".format(component)):
                c_folder = f
                print("Found folder {}".format(f))
                break
        self.data_path = os.path.join(data_root,c_folder)
        for f in os.listdir(self.data_path):
            if f.endswith(".png"):
                self.annotations.append([f,f[0]])

        #print(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path,self.annotations[idx][0])
        image = Image.open(img_path)
        label = torch.tensor(int(self.annotations[idx][1]))

        if self.transform:
            image = self.transform(image)

        #print("TBDataset Get ID: {}  File: {}  Xform: {}".format(idx,self.annotations[idx][0],self.transform))
        return (image, label)

# Network Definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#Hyperparameters
batch_size = 4
num_epochs = 3

# Load Data
transform = transforms.Compose(
    [transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = TaskboardDataset("/home/pgavriel/tb_data",20,transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = TaskboardDataset("/home/pgavriel/tb_data/test",20,transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
classes = (0,1)


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape)
# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# Create Model
net = Net()
net = net.to(device)

# Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train Network
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')



















#
