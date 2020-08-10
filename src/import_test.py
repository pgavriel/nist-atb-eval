#!/usr/bin/env python
import os, sys
import numpy as np
import rospy
import apriltag
import pyrealsense2 as rs
from PIL import Image
#import matplotlib.pyplot as plt

def main():
    print "\nOS\t" + os.__file__
    print "\nROSPY\t" + rospy.__file__
    print "\nNUMPY\t" + np.__file__
    #print "\nPLT\t" + plt.__file__
    print "\nREALSENSE\t" + rs.__file__
    print "\nAPRILTAG\t" + apriltag.__file__
    print "\nDone."

if __name__ == '__main__':
    main()
