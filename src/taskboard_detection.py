#!/usr/bin/env python

import os, sys
import rospy
import math
import numpy as np
import apriltag
import imutils
import cv2
from PIL import Image

def find_apriltag(gray_img):
    detector = apriltag.Detector()
    result = detector.detect(gray_img)
    return result

def isolate_board(image,thresh=120,area_lb=0,area_ub=100000):
    print("\nISOLATING TASKBOARD:")
    area = 0
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray,9,75,75)
    ret,th1 = cv2.threshold(smooth,thresh,255,cv2.THRESH_BINARY)
    stage1_img = cv2.cvtColor(th1,cv2.COLOR_GRAY2BGR)
    if imutils.is_cv2() or imutils.is_cv4():
        (contours, _) = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    elif imutils.is_cv3():
        (_, contours, _) = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(stage1_img, [box], 0,(0,255,0),2)
        if area < area_lb or area > area_ub:
            print("Found contour was not within specified bounds, not returning corners.")
            print("Contour Area:",area,"\tLBound:",area_lb," UBound:",area_ub)
            box = None

    else:
        box = None

    print("Bounding Box Found:\n",box)
    print("Region Area:",area)
    return stage1_img, smooth, box

def process_taskboard(image):
    path = '/home/ubuntu/catkin_ws/src/ros_picam/captures/debug/'
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    tag = find_apriltag(gray)
    if tag == None:
        print("No apriltag found, unable to process.")
    else:
        print("Apriltag found.")
        smooth = cv2.bilateralFilter(gray,9,75,75)
        ret,th1 = cv2.threshold(smooth,thresh,255,cv2.THRESH_BINARY)
        img = Image.fromarray(th1)
        img.save(os.path.join(path,"thresh1.png"))
