#!/usr/bin/env python
import os, sys
import rospy
import csv
import cv2
from PIL import Image
import numpy as np
from roi_config import getROI
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    source_path = '/home/pgavriel/ros_ws/src/nist_atb_eval/data/test2'
    save_path = '/home/pgavriel/tb_data/test2'
    roi_csv = '/home/pgavriel/ros_ws/src/nist_atb_eval/config/tb_roi.csv'

    file_list=os.listdir(source_path)
    for f in file_list:
        if not(f.endswith(".png")):
            file_list.remove(f)
    file_list.sort()
    print("FILE LIST:")
    print(file_list)


    # Create and move to output/save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    os.chdir(save_path)
    print("MOVED TO SAVE DIR: {}".format(save_path))

    # Load ROI from .csv
    img = cv2.imread(join(source_path,file_list[0]))
    print("ROI LOADED:")
    roi = getROI(roi_csv,img.shape[0],img.shape[1])

    # Set labels manually (All loaded images have the same state)
    #scores = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    scores = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]

    # Load labels into dict from csv
    use_dict = False
    scores_dict = dict()
    csv_file = '/home/pgavriel/ros_ws/src/nist_atb_eval/data/test/labels.txt'
    with open(csv_file) as label_csv:
        reader = csv.reader(label_csv)
        for row in reader:
            file = row[0]
            states = row[1:]
            if len(states) == 20:
                scores_dict[file] = states
            else:
                print("State length incorrect for {}, skipping.".format(file))


    #print(scores)

    size = 50
    for f in file_list:
        if use_dict:
            scores = scores_dict[f]
            print("File: {}  Scores: {}".format(f,scores))
        file = join(source_path,f)
        counter = 1
        try:
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for r in roi:
                x = int(r[1] * img.shape[0])
                y = int(r[2] * img.shape[1])
                s = int(size * r[3])
                crop = img[y-s:y+s, x-s:x+s]
                crop = cv2.resize(crop, (96,96), interpolation = cv2.INTER_AREA)
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


    pass
