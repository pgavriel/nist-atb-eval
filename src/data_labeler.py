#!/usr/bin/env python
import os, sys
import csv
import cv2
import rospy
from roi_config import getROI
from roi_config import getPoints
import Tkinter as tk
import time

global roi
global board_state
global position
global file_list

board_state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def next_board():
    global position
    global file_list

    with open(save_file, 'a') as newcsv:
        writer = csv.writer(newcsv,delimiter=',')
        writelist = []
        writelist.append(file_list[position])
        for s in board_state:
            writelist.append(s)
        print("Appending CSV: {}\n".format(writelist))
        writer.writerow(writelist)
    position = position+1

    if (position >= len(file_list)):
        quit_labeling()
    else:
        #imageCanvas.update()
        tkimg = tk.PhotoImage(file=os.path.join(source_path,file_list[position]))
        w.tkimg = tkimg
        imageCanvas.create_image(0,0,image=tkimg,anchor="nw")
        #imageCanvas.update()
        draw_boardstate()
        w.title(getframetitle())
        w.update()


def empty_board():
    global board_state
    print("All components set to 0")
    board_state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    draw_boardstate()


def fill_board():
    global board_state
    print("All components set to 1")
    board_state = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    draw_boardstate()


def quit_labeling():
    global position
    position = len(file_list)
    print("Quitting...")
    w.quit()


def getframetitle():
    str = "Current File: {} ({}/{})".format(file_list[position],position+1,len(file_list))
    print(str)
    return str


def draw_boardstate():
    imageCanvas.update()
    size = (imageCanvas.winfo_width()-2,imageCanvas.winfo_height()-2)

    counter = 0
    for r in roi:
        if (board_state[counter] == 1):
            color = 'green'
        elif (board_state[counter] == 0):
            color = 'red'
        else:
            color = 'black' #Shouldn't happen?
        p = getPoints(r,size)
        imageCanvas.create_line(p[0][0],p[0][1],p[1][0],p[1][1],p[2][0],p[2][1],p[3][0],p[3][1],p[0][0],p[0][1],width=4,fill=color)
        counter = counter + 1
    statestr = "STATE: {}".format(board_state)
    stateLabel.config(text=statestr)
    w.update()


def mousecallback(event):
    #print "Clicked at", event.x, event.y
    imageCanvas.update()
    size = (imageCanvas.winfo_width()-2,imageCanvas.winfo_height()-2)
    counter = 0
    for r in roi:
        p = getPoints(r,size)
        if (event.x >= p[0][0] and event.x <= p[1][0]):
            if(event.y >= p[0][1] and event.y <= p[2][1]):
                board_state[counter] = 0 if board_state[counter]==1 else 1
                print("Component {} set to {}".format(counter+1,board_state[counter]))
                break
        counter = counter + 1
    draw_boardstate()


# Define Tkinter GUI
w = tk.Tk()
w.title("--")
#tkimg = tk.PhotoImage(file=os.path.join(source_path,file_list[position]))
imageCanvas = tk.Canvas(w, width=800, height=800)
imageCanvas.pack()
#imageCanvas.create_image(0,0,image=tkimg,anchor="nw")
imageCanvas.bind("<Button-1>",mousecallback)
statestr = "STATE: {}".format(board_state)
stateLabel = tk.Label(w,text=statestr,font=("Courier", 14))
stateLabel.pack()

buttonFrame = tk.Frame(w,width=800,height=50)
emptyButton = tk.Button(buttonFrame, text="Empty Board", command=empty_board)
emptyButton.grid(column=0,row=0,padx=15)
fillButton = tk.Button(buttonFrame, text="Fill Board", command=fill_board)
fillButton.grid(column=1,row=0,padx=25)
nextButton = tk.Button(buttonFrame, text="Next", command=next_board)
nextButton.grid(column=2,row=0,padx=25)
quitButton = tk.Button(buttonFrame, text="Quit", command=quit_labeling)
quitButton.grid(column=3,row=0,padx=15)
buttonFrame.pack()


if __name__ == '__main__':
    #ROS Setup & Parameters
    rospy.init_node('taskboard_data_labeler', log_level=rospy.INFO)

    source_path = rospy.get_param("~source_path")
    rospy.loginfo("Source Path: %s", source_path)
    save_path = rospy.get_param("~save_path")
    save_file = os.path.join(save_path,rospy.get_param("~save_file"))
    rospy.loginfo("Save Path: %s", save_file)
    roi_csv = rospy.get_param("~roi_csv")
    rospy.loginfo("ROI CSV: %s", roi_csv)
    position = rospy.get_param("~starting_position")
    rospy.loginfo("Starting Position: %d", position)


    # Gather list of .png files in source path
    file_list=os.listdir(source_path)
    for f in file_list:
        if not(f.endswith(".png")):
            file_list.remove(f)
    file_list.sort()
    print("\nImages found in source path: {}\n".format(len(file_list)))

    img = cv2.imread(os.path.join(source_path,file_list[position]))
    roi = getROI(roi_csv,img.shape[0],img.shape[1],output=False)


    tkimg = tk.PhotoImage(file=os.path.join(source_path,file_list[position]))
    imageCanvas.create_image(0,0,image=tkimg,anchor="nw")
    draw_boardstate()
    w.title(getframetitle())

    # Tkinter mainloop
    w.mainloop()
