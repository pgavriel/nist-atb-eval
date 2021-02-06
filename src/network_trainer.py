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
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from TaskboardDataset import TaskboardDataset
from TaskboardNetwork import Net

# Function for displaying images from the dataset
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Set Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

# Hyperparameters
batch_size = 1 # TODO: Research tuning this
num_epochs = 20

# Settings
save_path = '/home/pgavriel/ros_ws/src/nist_atb_eval/models'
component = 9
version = 'v1'
model_name = '{}_{:02d}.pth'.format(version,component)



# Load Data
transform = transforms.Compose(
    [transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = TaskboardDataset("/home/pgavriel/tb_data",component,transform=transform)
train_set.addAll()
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = TaskboardDataset("/home/pgavriel/tb_data/test",component,transform=transform)
test_set.addAll()
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
classes = (0,1)


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape)
# show images
imshow(torchvision.utils.make_grid(images))
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
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

# Save Model
PATH = os.path.join(save_path,model_name)
torch.save(net.state_dict(), PATH)
print("Saved {}".format(PATH))

# Test the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data

        print(images)
        print(type(images))
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

















#
