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

if __name__ == '__main__':
    # Setup ROS Node
    rospy.init_node('taskboard_network_trainer', log_level=rospy.INFO)
    # Load ROS Parameters
    training_path = rospy.get_param("~training_path")
    rospy.loginfo("Training Path: %s", training_path)
    test = rospy.get_param("~test")
    if test:
        test_path = rospy.get_param("~test_path")
        rospy.loginfo("Test Path: %s", test_path)
    else:
        rospy.loginfo("Test Model: FALSE")
    save_path = rospy.get_param("~save_path")
    rospy.loginfo("Model Save Path: %s", save_path)
    model_version = rospy.get_param("~model_version")
    rospy.loginfo("Model Version: %s", model_version)
    batch_size = rospy.get_param("~batch_size") # TODO: Research tuning this
    rospy.loginfo("Batch Size: %s", batch_size)
    num_epochs = rospy.get_param("~epochs")
    rospy.loginfo("Training Epochs: %s", num_epochs)
    show_loss = rospy.get_param("~show_loss_every")
    rospy.loginfo("Show Loss Every: %s steps", show_loss)
    components = rospy.get_param("~components")
    if components == 0:
        components = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    else:
        components = str(components).split(' ')
        components = map(int,components)
    rospy.loginfo("Components: %s", components)
    verify = rospy.get_param("~verify_component")
    rospy.loginfo("Verify Components: %s", "TRUE" if verify else "FALSE")

    # Set Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rospy.loginfo("Device: %s",device)


    # Go through training/testing for each component requested
    for component in components:
        print("\nNOW TRAINING COMPONENT {} ----- ----- ----- ----- -----".format(component))
        model_name = '{}_{:02d}.pth'.format(model_version,component)

        # Load Data
        transform = transforms.Compose(
            [transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        print("\nInitializing Training Dataset: ")
        train_set = TaskboardDataset(training_path,component,transform=transform)
        train_set.addAll()
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        if test:
            print("\nInitializing Test Dataset: ")
            test_set = TaskboardDataset(test_path,component,transform=transform)
            test_set.addAll()
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        # Define classes
        classes = (0,1)

        if verify:
            print("\nVerify training sample is correct: ")
            print("Grabbing batch... (Size {})".format(batch_size))
            # Get random training images
            dataiter = iter(train_loader)
            images, labels = dataiter.next()
            # Print shape and labels
            print("Shape: {}".format(images.shape))
            lbls = []
            for i in range(batch_size):
                lbls.append(classes[labels[i]])
            print('Labels: {}'.format(lbls))
            # Show Images
            imshow(torchvision.utils.make_grid(images))


        # Create Model
        print("\nCreating Model...")
        net = Net()
        net = net.to(device)

        # Function and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # Train Network
        print("Training Network...")
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
                if i % show_loss == show_loss-1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished Training Component {}'.format(component))

        # Save Model
        PATH = os.path.join(save_path,model_name)
        torch.save(net.state_dict(), PATH)
        print("Saved Model: {}".format(PATH))

        if test:
            print("\nTesting network...")
            # Test the model on the test set
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data

                    #print(images)
                    #print(type(images))
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))

















#
