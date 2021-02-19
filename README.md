# nist-atb-eval   

## Description  
This package contains tools for data collection, network training, and automatic evaluation of [NIST Assembly Taskboard 1](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly). First, OpenCV is used to extract a taskboard from an image, and warp it into a regular square image using Homography. These nearly identical taskboard images constitute the base for the dataset. Each component is then extracted from the original dataset, and these sub images are used to train an individual neural network for classifying each component. To then evaluate a taskboard, the same process is used to extract the taskboard, split it into component images, and those images are sent through their respective networks in order to get a score for each one. Those scores can then be summed to get an overall score for the board.  

PACKAGE STILL A WORK IN PROGRESS.    

## STAGE 1: Dataset Collection  
![Taskboard Image Transformation](data/misc/tb_transform.png)  
The universality of the pre-trained models contained in this package is unproven and unlikely considering they were trained on a relatively small dataset in fairly static conditions. Making the trained models more robust is a future goal. So in order to get things working in a new environment, a new dataset will need to be collected. However, it's worth noting that with a set of roughly 200 images, I was able to get a board evaluation accuracy of 98.6% on a testing set.   

To start collecting data you need to set up some kind of overhead camera that can take pictures of the taskboard in various states and positions. Initially we started using an Intel RealSense camera, but the resolution wasn't ideal, so instead we used a Raspberry Pi 4 with an Arducam Auto-Focus 5MP camera module. The Raspberry Pi ran a ROS node with a service let us extract and save taskboard images over wifi. The setup for the ROS Pi Camera is detailed [here](https://github.com/pgavriel/ros_picam).  

For an initial dataset, the easiest thing to do is collect a set of images with all components present, and another set of images for empty boards. This way very little data labeling needs to be done when splitting the taskboard images into training images.  

### Taskboard Extraction (using taskboard_detection.py)   
OpenCV is used to extract the precise locations of the four corners of a taskboard, which is then used to warp the taskboard into a square image. Approaching the problem this way provides a very consistent dataset even if the board is rotated or the camera is moved in some way. The way this is currently implemented works, but could be significantly improved or approached differently, because as long as the board corners can be accurately extracted, everything else should still work.  
**taskboard_detection.py** can be imported, and the **process_taskboard(image)** function will take an image and attempt to extract and return a taskboard image like the ones in the dataset. Note that an apriltag is used in our approach as a way of keeping the output orientation consistent. Debug images can be saved to see what's going wrong if things aren't working. The method used is outlined:  
1. Isolate the board   
  - Filter the image     
  - Threshold the image (may need to be adjusted, light surfaces may cause problems)  
  - Find the largest contour in the thresholded image and assume that it's the taskboard  
2. Fix the orientation of the contour based on the apriltag   
3. Refine the corner positions  
  - Masks the image around an inflated taskboard contour  
  - Perform Hough line detection that should detect the edges of the board  
  - Inspect square regions around the original corners, and look for edge line intersections to be the new refined corner positions  
4. Use the refined corner positions to warp the taskboard into a square image using Homography   


## STAGE 2: ROI Setup & Data Splitting  
Because all the collected taskboard images should be nearly identical, the position of any particular component within each image should be fixed, meaning we can explicitly define image locations to extract each component.   
### roi_config.py / roi_config.launch  
Provide this script with one of your collected taskboard images, and the default ROI .csv file, and it will allow you to visually edit the component regions as needed, and then save the new configuration to a .csv file. This .csv file is then used to extract component sub-images in order to train the individual networks, and for evaluating taskboard images.  
### roi_img_split.py    
Once you have your ROI csv set up, this script can be used to go through your collected dataset, and split all of the component sub-images into separate folders that will be used to train the networks. The labels of the images being split must be specified, so that the first character in the filename of each image can be the proper label (i.e. 0 or 1).

## STAGE 3: Neural Network Setup & Training
Now that you have 20 folders full of correctly labeled component images, it's time to train the models. The **network_trainer.py** script is used for this.

## STAGE 4: Taskboard Image Evaluation
With the networks now trained, we should be able to extract a novel taskboard image, and feed it through the networks to evaluate the board state.  
**network_evaluate.py** defines a TaskboardEvaluator class that should be instantiated with the following:  
*model_dir* - The directory to find the saved .pth files   
*eval_dir* - The directory to find images to be evaluated  
*roi_csv* - Path to .csv file that defines ROIs on the board
*transform* - Transform composition to be performed on the data going into the networks. Should be the same transforms used when training.  
*model_ver* - The model version is just the prefix to look for on the .pth files.  

### TaskboardEvaluator Functions
**load_board(image)** - The argument passed is the file name of the image to evaluate, which should be found in *eval_dir*. This opens the image, finds the ROIs, takes a cropped image for each component, and performs the necessary transforms. The result is a list of 20 tensors stored in TaskboardEvaluator.components that are ready to be fed into their respective networks.  
**eval_board()** - After the board is loaded, this function calls **eval_component(c)** for each component tensor. After this function, the TaskboardEvaluator.scores member should be a list of 20 values representing the presence of the board components.  
**eval_component(component)** - Takes the specified component tensor, runs it through it's respective model, and returns the predicted score.
**save_results()** - *In order to use this you must first set the TaskboardEvaluator.save_dir member for output.* This function will save an image of the original board with green boxes drawn around detected components, and red boxes around components that were not detected. It can be easier to quickly verify accuracy this way.  

### Controls  
With any of the opencv windows focused, keystrokes will be heard by the program.  
##### data_collection node:   
P - Pause/Play (Toggle step_through mode)  
S - Save current taskboard image  
Q - Quit program  
##### roi_config node:  
S - Save new ROI configuration to specified .csv file  
Q - Quit program without saving  

### Notes    
apriltag import resolved with pip install apriltag  
imutils import resolved with pip install imutils    

### Deprecated Files  
**data_collection.py** - Original data collection was done using a realsense camera, and the code for board detection and extraction was all contained in one file. The relevant functions from this are contained in taskboard_detection.py.   
**mse_evaluate.py** - The first method of board evaluation tried to figure out the score for each component based on a Mean Squared Error between a test image, an empty board reference, and a full board reference. While this kind of works, it's not robust to different capture angles.  
**roi_comparison.py** - This is just a script that will get the ROIs for 2 taskboard images and save a bunch of side by side reference images (seen above).  
**import_test.py** - Just a simple test script for checking imports.  
