#!/usr/bin/env python
import os, sys
import csv
import numpy as np
import cv2
import rospy
from PIL import Image


def nothing(x):
    pass

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err



def processTB(image):
    tb = image.copy()
    tb = cv2.bilateralFilter(tb,9,75,75)
    tb = cv2.cvtColor(tb, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('pre',tb)
    tb = cv2.normalize(tb,  tb, 0, 255, cv2.NORM_MINMAX)
    #cv2.imshow('post',tb)
    return tb
    pass


def evalAllComponents(eval,empty,full,comp_list,original):
    debug = False
    print("\nEVALUATING COMPONENTS:")
    if debug: print(" # Component\tEmpty MSE\tFull MSE\tScore\t\tPresent?")
    comp_number = 1
    scores = []
    for comp in comp_list:
        name = comp[0]
        x = comp[1]
        y = comp[2]
        scale = int(25 * comp[3])
        eval_crop = eval[y-scale:y+scale,x-scale:x+scale]
        emp_crop = empty[y-scale:y+scale,x-scale:x+scale]
        full_crop = full[y-scale:y+scale,x-scale:x+scale]
        #stack = np.hstack( (emp_crop,eval_crop,full_crop) )
        #cv2.imshow(name,stack)
        e_error = mse(eval_crop,emp_crop)
        f_error = mse(eval_crop,full_crop)
        score = e_error - f_error
        scores.append(int(score))
        eerror_f = str('{:07.1f}'.format(e_error))
        ferror_f = str('{:07.1f}'.format(f_error))
        score_f = str('{:=+6d}'.format(int(score)))
        present = False if (score<0) else True
        if not present:
            cv2.rectangle(original,(x-scale,y-scale),(x+scale,y+scale),(0,0,255),2)
        else:
            cv2.rectangle(original,(x-scale,y-scale),(x+scale,y+scale),(0,255,0),2)

        if debug: print(str('{:02}'.format(comp_number)),name,"\t",eerror_f,"\t",ferror_f,"\t",score_f,"\t",present)
        comp_number = comp_number + 1
    cv2.imshow('Evaluation',original)
    #print("SCORES\n",scores)
    rospy.loginfo("SCORES:\n %s", scores)
    return scores,original

def drawScores(image,roi,scores):
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r, score in zip(roi,scores):
        x = r[1]
        y = r[2]
        scale = int(25 * r[3])
        if score < 0:
            color = (0,0,255)
        else:
            color = (0,255,0)
        cv2.rectangle(img,(x-scale,y-scale),(x+scale,y+scale),color,2)
        cv2.putText(img,str(int(score)),(x-scale,y-scale-2), font, 0.4,color,1,cv2.LINE_AA)
    return img

def getROI(csv_file,width,height):
    im_width = width
    im_height = height
    roi_list = []
    #with open(csv_file, newline='') as roi_csv:
    with open(csv_file) as roi_csv:
        reader = csv.DictReader(roi_csv)
        for row in reader:
            name = row['name']
            x = float(row['xval']) * im_width
            y = float(row['yval']) * im_height
            region_scale = float(row['scale'])
            roi_list.append((name,int(x),int(y),region_scale))
    return roi_list

def main():
    #ROS Setup & Parameters
    rospy.init_node('eval_node', log_level=rospy.INFO)
    step_through = rospy.get_param("~step")
    roi_file = rospy.get_param("~roi_file")
    data_dir = rospy.get_param("~data_dir")
    eval_dir = rospy.get_param("~eval_dir")
    eval_img = rospy.get_param("~eval_img")
    image = os.path.join(eval_dir,eval_img)
    rospy.loginfo("roi_file: %s", roi_file)
    rospy.loginfo("data_dir: %s", data_dir)
    rospy.loginfo("eval_img: %s", image)

    #OpenCV Window Setup
    #cv2.namedWindow('Testing', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Evaluation', cv2.WINDOW_AUTOSIZE)
    emptylist = os.listdir(os.path.join(data_dir,'empty'))
    fulllist = os.listdir(os.path.join(data_dir,'full'))
    rospy.loginfo("EMPTY BOARD REFERENCE IMAGES:\n %s", emptylist)
    rospy.loginfo("FULL BOARD REFERENCE IMAGES:\n %s", fulllist)
    #emptyboard = cv2.imread(os.path.join(data_dir,'testing/empty.png'))
    #fullboard = cv2.imread(os.path.join(data_dir,'testing/full.png'))
    queryboard = cv2.imread(image)
    roi = getROI(roi_file,queryboard.shape[0],queryboard.shape[1])
    original = queryboard.copy()
    cv2.imshow('orig',original)

    while True:
        q = processTB(queryboard)
        sum_score = np.zeros(20)
        for e, f in zip(emptylist,fulllist):
            e = 'empty/'+e
            f = 'full/'+f
            emptyboard = cv2.imread(os.path.join(data_dir,e))
            fullboard = cv2.imread(os.path.join(data_dir,f))
            empty = processTB(emptyboard)
            full = processTB(fullboard)
            scores, score_img = evalAllComponents(q,empty,full,roi,original)
            sum_score = [a + b for a, b in zip(scores, sum_score)]
            #cv2.imshow(e,score_img)
            #print("SUM SCORES\n",sum_score)
            rospy.loginfo("SUM SCORES:\n %s", sum_score)
        print("\n")
        rospy.loginfo("FINAL SCORES:\n %s", sum_score)
        final_img = drawScores(original,roi,sum_score)
        cv2.imshow('FINAL',final_img)
        #ref = cv2.absdiff(e,f)
        #eval = cv2.absdiff(e,q)
        #vs_emp = cv2.absdiff(e,q)
        #vs_full = cv2.absdiff(f,q)


        #evalAllComponents(ref,eval,roi,original,thresh)
        #evalAllComponents2(ref,vs_emp,vs_full,roi,original,thresh)
        #evalAllComponents3(q,e,f,roi,original,thresh)

        #stack1 = np.hstack( (e,f,q) )
        #cv2.imshow('Testing',stack1)

        #stack2 = np.hstack( (ref,eval) )
        #stack2 = np.hstack( (ref,vs_emp,vs_full) )
        #cv2.imshow('ref/eval',stack2)


        wait = 0 if step_through else 1000
        key = cv2.waitKey(wait) & 0xFF
        if key == ord('p'):
            step_through = not step_through
        elif key == ord('q'):
            break
    pass

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException: pass
