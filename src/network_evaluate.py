#!/usr/bin/env python
import os, sys
import csv
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from TaskboardDataset import TaskboardDataset
from TaskboardNetwork import Net
from roi_config import getROI

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
        size = 50
        for r in roi:
            x = int(r[1] * img.shape[0])
            y = int(r[2] * img.shape[1])
            s = int(size * r[3])
            crop = img[y-s:y+s, x-s:x+s]
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
    model_dir = '/home/pgavriel/ros_ws/src/nist_atb_eval/models'
    model_version = 'v1'
    eval_dir = '/home/pgavriel/ros_ws/src/nist_atb_eval/data/test'
    roi_csv = '/home/pgavriel/ros_ws/src/nist_atb_eval/config/tb_roi.csv'
    # Transform should be the same as the one used when training
    data_transform = transforms.Compose(
        [transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    evaluator = TaskboardEvaluator(model_dir,eval_dir,roi_csv,data_transform,model_version)


    # EVALUATE SINGLE IMAGE
    eval_img = "test11.png"
    evaluator.load_board(eval_img)
    evaluator.eval_board(debug=False)
    evaluator.save_dir = '/home/pgavriel/ros_ws/src/nist_atb_eval/data/test/scored'
    evaluator.save_results()
    print("SCORES: {}".format(evaluator.scores))

    # EVALUATE ENTIRE FOLDER WITH LABELED GROUND TRUTH CSV
    # eval_list = []
    # for f in os.listdir(eval_dir):
    #     if f.endswith(".png"):
    #         eval_list.append(f)
    # eval_list.sort()
    #
    # groundtruth = open('/home/pgavriel/ros_ws/src/nist_atb_eval/data/test/labels.txt','r')
    # lines = groundtruth.readlines()
    # counter = 0
    # percent_sum = 0
    # for f in eval_list:
    #     evaluator.load_board(f)
    #     evaluator.eval_board(debug=False)
    #     print("\nFile: {}".format(f))
    #     print("{} EVAL SCORES".format(evaluator.scores))
    #     # For Testing
    #     for line in lines:
    #         l = line.split(',')
    #         if l[0] == f:
    #             truth = list(map(int, l[1:]))
    #             print("{} GROUND TRUTH".format(truth))
    #             diff = []
    #             diff_sum = 0
    #             for i in range(0,20):
    #                 diff.append(int(truth[i] == evaluator.scores[i]))
    #             diff_sum = sum(diff)
    #             percent = (float(diff_sum)/20)*100
    #             counter = counter + 1
    #             percent_sum = percent_sum + percent
    #             print("{} COMPARED, {}/20 - {}%".format(diff,diff_sum,percent))
    #             break
    #
    # print("\nOVERALL PERFORMANCE\nAverage Accuracy on Test Set: {}%".format(percent_sum/counter))
