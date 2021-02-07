#!/usr/bin/env python
import os, sys
import rospy
import csv
import cv2
from PIL import Image
import numpy as np
from roi_config import getROI


srcpath = '/home/pgavriel/ros_ws/src/nist_atb_eval/data/empty'
os.chdir(srcpath)
img1 = cv2.imread('picam2-taskboard-14-04-53-236640.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
srcpath = '/home/pgavriel/ros_ws/src/nist_atb_eval/data/full'
os.chdir(srcpath)
img2 = cv2.imread('picam2-taskboard-14-22-33-464538.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

savepath = '/home/pgavriel/tb_data/misc'
os.chdir(savepath)

if img1.shape != img2.shape:
    print("Images are not the same shape")
    sys.exit()

roi = getROI('/home/pgavriel/ros_ws/src/nist_atb_eval/config/tb_roi.csv',img1.shape[0],img1.shape[1])

cv2.imshow('1',img1)
cv2.imshow('2',img2)
cv2.waitKey(0)

size = 50

stacks = []
for r in roi:
    filename = r[0] + '.png'
    x = int(r[1] * img1.shape[0])
    y = int(r[2] * img1.shape[1])
    s = int(size * r[3])
    print("Saving {} - x:{} y:{} s:{}".format(filename,x,y,s))
    crop1 = img1[y-s:y+s, x-s:x+s]
    crop1 = cv2.resize(crop1, (100,100), interpolation = cv2.INTER_AREA)
    crop2 = img2[y-s:y+s, x-s:x+s]
    crop2 = cv2.resize(crop2, (100,100), interpolation = cv2.INTER_AREA)

    stack = np.hstack((crop1, crop2))
    stacks.append(stack)
    image = Image.fromarray(stack)
    image.save(filename)

v1 = np.vstack((stacks[0],stacks[1],stacks[2],stacks[3],stacks[4]))
v2 = np.vstack((stacks[5],stacks[6],stacks[7],stacks[8],stacks[9]))
h1 = np.hstack((v1,v2))
image = Image.fromarray(h1)
image.save('h1.png')
v1 = np.vstack((stacks[10],stacks[11],stacks[12],stacks[13],stacks[14]))
v2 = np.vstack((stacks[15],stacks[16],stacks[17],stacks[18],stacks[19]))
h2 = np.hstack((v1,v2))
image = Image.fromarray(h2)
image.save('h2.png')
