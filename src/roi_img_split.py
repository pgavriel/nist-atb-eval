#!/usr/bin/env python
import os, sys
import rospy
import csv
import cv2
from PIL import Image
import numpy as np
from roi_config import getROI
from roi_config import cropRegion
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    source_path = '/home/pgavriel/ros_ws/src/nist_atb_eval/data/test2'
    save_path = '/home/pgavriel/tb_data/test2'
    roi_csv = '/home/pgavriel/ros_ws/src/nist_atb_eval/config/tb_roi.csv'
    # Setup ROS Node
    rospy.init_node('taskboard_image_splitter', log_level=rospy.INFO)
    # Load ROS Parameters
    source_path = rospy.get_param("~source_path")
    rospy.loginfo("Source Path: %s", source_path)
    save_path = rospy.get_param("~save_path")
    rospy.loginfo("Save Path: %s", save_path)
    roi_csv = rospy.get_param("~roi_csv")
    rospy.loginfo("ROI CSV: %s", roi_csv)
    out_size = rospy.get_param("~output_size")
    rospy.loginfo("Output Size : %dx%d", out_size,out_size)
    load_labels = rospy.get_param("~load_labels")
    rospy.loginfo("Load Labels: %s", "TRUE" if load_labels else "FALSE")
    board_state = rospy.get_param("~board_state")
    if load_labels:
        rospy.loginfo("Loading from CSV: %s",board_state)
        csv_file = board_state
    else:
        board_state = board_state.split(" ")
        rospy.loginfo("Manually Set: %s", board_state)
        scores = board_state


    # Gather list of .png files in source path
    file_list=os.listdir(source_path)
    for f in file_list:
        if not(f.endswith(".png")):
            file_list.remove(f)
    file_list.sort()
    print("\nImages found in source path: {}\n".format(len(file_list)))


    # Create and move to output/save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    os.chdir(save_path)
    print("MOVED TO SAVE DIR: {}".format(save_path))

    # Load ROI from .csv
    img = cv2.imread(join(source_path,file_list[0]))
    print("ROI LOADED:")
    roi = getROI(roi_csv,img.shape[0],img.shape[1])

    # Load labels into dict from csv
    if load_labels:
        scores_dict = dict()
        with open(csv_file) as label_csv:
            reader = csv.reader(label_csv)
            for row in reader:
                file = row[0]
                states = row[1:]
                if len(states) == 20:
                    scores_dict[file] = states
                else:
                    rospy.logwarn("State length incorrect for %s, skipping.",file)

    skips = []
    for f in file_list:
        if load_labels:
            if f in scores_dict.keys():
                scores = scores_dict[f]
            else:
                rospy.logwarn("File '%s' not found in csv. Skipping image.",f)
                skips.append(f)
                continue
        print("File: {}  Scores: {}".format(f,scores))
        file = join(source_path,f)
        counter = 1
        try:
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for r in roi:
                crop = cropRegion(r,img)
                crop = cv2.resize(crop, (out_size,out_size), interpolation = cv2.INTER_AREA)
                image = Image.fromarray(crop)
                folder = "{:02d}-{}".format(counter,r[0])
                if not os.path.exists(folder):
                    os.makedirs(folder)
                _, _, files = next(os.walk(folder))
                file_count = "{:04d}".format(len(files)+1)
                filename = "{}/{}-{}.png".format(folder,scores[counter-1],file_count)
                print("Saving {}".format(filename))
                image.save(filename)
                counter = counter + 1

        except BaseException as error:
            print('An exception occurred: {}'.format(error))

    print("\nImage splitting complete.")
    if len(skips) > 0:
        print("Split {}/{} images successfully.".format(len(file_list)-len(skips),len(file_list)))
        print("Images skipped: {}".format(skips))
    else:
        print("All {} images split successfully.".format(len(file_list)))
