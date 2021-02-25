#!/usr/bin/env python
import os, sys
import csv
import cv2
import rospy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from TaskboardDataset import TaskboardDataset
from TaskboardNetwork import Net
from roi_config import getROI
from roi_config import cropRegion

# Set Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class TaskboardEvaluator(object):
    def __init__(self, model_dir, eval_dir, roi_csv, transform=None, model_ver='v1'):
        # Path to find .pth models
        self.model_dir = model_dir
        # Model version (.pth file prefix)
        self.model_ver = model_ver
        # Path to find images to evaluate
        self.eval_dir = eval_dir
        # .csv file definining taskboard ROIs
        self.roi_csv = roi_csv
        # Data transform
        self.transform = transform

        self.save_dir = None
        self.current_board = None
        self.components = []
        self.scores = []
        print(self)


    def __str__(self):
        str = "EVALUATOR:\nModel Dir: {}\nVersion: {}\nEval Dir: {}\nROI CSV: {}\nTransform: {}\n".format(self.model_dir,self.model_ver,self.eval_dir,self.roi_csv,self.transform)
        return str

    def load_board(self,image):
        self.current_board = image
        img = cv2.imread(os.path.join(self.eval_dir,image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi = getROI(self.roi_csv,img.shape[0],img.shape[1],False)
        self.components = []
        # TODO: Image splitter should be made into a class with a function to easily do this
        for r in roi:
            crop = cropRegion(r,img)
            crop = cv2.resize(crop, (96,96), interpolation = cv2.INTER_AREA)
            image = Image.fromarray(crop)
            if self.transform:
                image = self.transform(image)
            self.components.append(image)




    def eval_component(self,component,debug=True):
        if len(self.components) != 20:
            print("Component list doesn't look right. Load a taskboard first.")
            return
        # Load network model for relevant component
        model = "{}_{:02d}.pth".format(self.model_ver,component)
        net = Net()
        net = net.to(device)
        net.load_state_dict(torch.load(os.path.join(self.model_dir,model)))


        image = self.components[component-1]
        #imshow(image)

        var_image = torch.Tensor(image)
        var_image = var_image.unsqueeze(0)
        var_image = var_image.to(device)
        output = net(var_image)
        _, predicted = torch.max(output.data, 1)

        if predicted.item() == 0:
            state = "NOT PRESENT"
        elif predicted.item() == 1:
            state = "PRESENT"
        else:
            state = "UNKNOWN?"
        if debug:
            print("Tested component {:02d} with model {} | {}".format(component,model,state))
        #imshow(image)
        return predicted.item()

    def eval_board(self,debug=True):
        if len(self.components) != 20:
            print("Component list doesn't look right. Load a taskboard first.")
            return
        self.scores = []
        for i in range(1,21):
            comp_score = self.eval_component(i,debug)
            self.scores.append(comp_score)

    def save_results(self):
        if self.save_dir is None:
            print("Please set the save_dir property of the evaluator first.")
            return
        if self.current_board is None:
            print("You must load a board and evaluate it first.")
            return
        if len(self.scores) != 20:
            print("You must evaluate the board to generate scores first.")
            return

        img = cv2.imread(os.path.join(self.eval_dir,self.current_board))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi = getROI(self.roi_csv,img.shape[0],img.shape[1],False)
        save_file = self.current_board[:-4] + "_scored.png"

        size = 50
        for i in range(0,20):
            if self.scores[i] == 1:
                color = (0,255,0)
            else:
                color = (255,0,0)
            x = int(roi[i][1] * img.shape[0])
            y = int(roi[i][2] * img.shape[1])
            s = int(size * roi[i][3])
            img = cv2.rectangle(img,(x-s,y-s),(x+s,y+s),color,2)

        save_img = Image.fromarray(img)
        save_img.save(os.path.join(self.save_dir,save_file))




if __name__ == '__main__':
    # Setup ROS Node
    rospy.init_node('taskboard_network_evaluator', log_level=rospy.INFO)
    # Load ROS Parameters
    model_dir = rospy.get_param("~model_dir")
    rospy.loginfo("Model Dir: %s", model_dir)
    model_version = rospy.get_param("~model_version")
    rospy.loginfo("Model Version: %s", model_version)
    roi_csv = rospy.get_param("~roi_csv")
    rospy.loginfo("ROI CSV: %s", roi_csv)
    eval_dir = rospy.get_param("~eval_dir")
    rospy.loginfo("Eval Dir: %s", eval_dir)
    # Set up list of images to evaluate
    eval_images = rospy.get_param("~eval_images")
    eval_list = []
    if eval_images.upper() == "ALL":
        for f in os.listdir(eval_dir):
            if f.endswith(".png"):
                eval_list.append(f)
        eval_list.sort()
    else:
        eval_list = eval_images.split(' ')
    rospy.loginfo("Eval List: %s", eval_list)
    rospy.loginfo("Eval List Length: %s", len(eval_list))
    # Ground truth parameters
    use_ground_truth = rospy.get_param("~use_ground_truth")
    if use_ground_truth:
        gt_csv = rospy.get_param("~ground_truth_csv")
        rospy.loginfo("Ground Truth CSV: %s", gt_csv)
    else:
        rospy.loginfo("Use Ground Truth: FALSE")
    # Image output parameters
    save_img_output = rospy.get_param("~save_img_output")
    if save_img_output:
        save_dir = rospy.get_param("~save_dir")
        rospy.loginfo("Image Output Dir: %s", save_dir)
    else:
        rospy.loginfo("Save Image Output: FALSE")

    # Transform should be the same as the one used when training
    data_transform = transforms.Compose(
        [transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Create our evaluator
    evaluator = TaskboardEvaluator(model_dir,eval_dir,roi_csv,data_transform,model_version)

    if save_img_output:
        evaluator.save_dir = save_dir
    if use_ground_truth:
        groundtruth = open(gt_csv,'r')
        gt_lines = groundtruth.readlines()
        labeled_files = []
        for l in gt_lines:
            labeled_files.append(l.split(',')[0])
        counter = 0
        percent_sum = 0

    # Evaluate all specified image files
    for file in eval_list:
        print("Evaluating '{}'".format(file))
        evaluator.load_board(file)
        evaluator.eval_board(debug=False)

        if save_img_output:
            evaluator.save_results()
        print("{} EVAL SCORES".format(evaluator.scores))

        if use_ground_truth:
            if file in labeled_files:
                for line in gt_lines:
                    l = line.split(',')
                    labeled_file = l[0]
                    if labeled_file == file:
                        truth = list(map(int, l[1:]))
                        print("{} GROUND TRUTH".format(truth))
                        diff = []
                        diff_sum = 0
                        for i in range(0,20):
                            diff.append(int(truth[i] == evaluator.scores[i]))
                        diff_sum = sum(diff)
                        percent = (float(diff_sum)/20)*100
                        counter = counter + 1
                        percent_sum = percent_sum + percent
                        print("{} COMPARED, {}/20 - {}%".format(diff,diff_sum,percent))
            else:
                rospy.logwarn("File '%s' not found in ground truth list, skipping.",file)
    if use_ground_truth:
        print("\nOVERALL PERFORMANCE\nAverage Accuracy on Test Set: {}%".format(percent_sum/counter))
    print("\nEvaluation Complete.")
