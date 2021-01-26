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

# Define naming scheme for extracted files
def naming_scheme(filename):
    return filename[:3]

if __name__ == '__main__':
    source_path = '/home/pgavriel/ros_ws/src/ros_picam/captures'
    save_path = '/home/pgavriel/Pictures/tb_data'
    roi_csv = '/home/pgavriel/ros_ws/src/nist_atb_eval/config/tb_roi.csv'

    file_list = [f for f in listdir(source_path) if isfile(join(source_path, f))]
    print(file_list)

    # Create and move to output/save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    os.chdir(save_path)

    # Load ROI from .csv
    img = cv2.imread(join(source_path,file_list[0]))
    roi = getROI(roi_csv,img.shape[0],img.shape[1])
    print(roi)

    size = 50
    for f in file_list:
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
                folder = "{:02d}".format(counter)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                _, _, files = next(os.walk(folder))
                file_count = "{:04d}".format(len(files)+1)
                filename = "{}/{}-{}.png".format(folder,naming_scheme(f),file_count)
                print("Saving {}".format(filename))
                image.save(filename)
                counter = counter + 1

        except BaseException as error:
            print('An exception occurred: {}'.format(error))


    pass