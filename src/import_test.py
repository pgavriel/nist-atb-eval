#!/usr/bin/env python
import os, sys
import numpy as np
import rospy
import apriltag
import pyrealsense2 as rs
from PIL import Image
import imutils
import cv2
#import matplotlib.pyplot as plt

def main():
    print "\nOS\t" + os.__file__
    print "\nROSPY\t" + rospy.__file__
    print "\nNUMPY\t" + np.__file__
    #print "\nPLT\t" + plt.__file__
    print "\nREALSENSE\t" + rs.__file__
    print "\nCV2\t" + cv2.__file__
    print "\nCV_VER\t" + cv2.__version__
    print "\nAPRILTAG\t" + apriltag.__file__
    print "\nDone."

if __name__ == '__main__':
    main()
