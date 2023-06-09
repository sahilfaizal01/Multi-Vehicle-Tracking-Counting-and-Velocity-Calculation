# Multi-Vehicle-Tracking-Counting-and-Velocity-Calculation
This computer vision project utilizes YOLOv7 and DeepSORT Algorithm for multi-object tracking.

# Future Work
* Implementing YOLOv8 version
* Incorporating Vehicle Intensity Heatmap
* Making a webapp

# Introduction
* Object Tracking is a method to track detected objects throughout the frames using their spatial and temporal features.
* What we do in tracking is, we get the initial set of detections, in the next step we assign them unique id's and then tracking them throughout the frames of the video feed while maintaining the assigned id's.
* So tracking is a two-step process
* In the first step, we do the detection and localization of the object using any time of object detector it can be YOLOv7, YOLOv8, YOLO-NAS or YOLOR.
* In the second step, using a motion predictor we predict the future motion of the object using its past information.

# Need for Object Tracker
* In a video feed, we are detection a car, whenever the car gets occluded or overlapped by something for example a truck, the detector will fail. But if we have a tracker with it, we will be able to predict the future motion as well as track the car by assigning it a unique id.
* In object tracking we assign a unique id to each of the object we want to track and maintain that id till the object is in that frame.
* In the single object tracker, we only track a single object, no mateer how many objects are present in the frame.
* In multiple object tracker, we can track multiple objects present in a frame at the same time even of different classes while maintaining high speed.

# DeepSORT Algorithm
* It is a computer vision algorithm used to track the objects while assigning each of the tracked object to a unique ID.
* It is an extension of the SORT algorithm.
* DeepSORT introduces deep learning into SORT algorithm by adding appearance descriptor to reduce the identify switches and hence making the tracking more efficient.
* DeepSORT introduces deep learning into SORT algorithm by adding appearance descriptor to reduce the identify switches and hence making the tracking more efficient.

![image](https://github.com/sahilfaizal01/Multi-Vehicle-Tracking-Counting-and-Velocity-Calculation/assets/106440078/79a6bfb3-bde8-4c85-891c-e80aef1be750)

## Simple Online Real Time Tracking (SORT)
SORT is an approach to object tracking where Kalman Filters and Hungarian Algorithm are used to track objects. <br>
SORT consists of four components which are as follows:-
#### 1) Detection
In the first step, detection of all the objects which are needed to be tracked is done using YOLOv7 or YOLOv8.
#### 2) Estimation
In this step, the detections are passed from the current frame to the next frame to estimate the position of the target in the next frame using Gaussian Distribution and constant velocity model. The estimation is done using the Kalman Filter.
#### 3) Data Association
We now have the target bounding box and the detected bounding box. So, a cost matrix is computed as the intersection-over-union (IoU) distance between each detection and all predicted bounding boxes from the existing targets.
#### 4) Creation and Delection of Track Identities
When object enters or leave the unique object id's are created and destroyed accordingly.

## Issues of SORT Algorithm
1) Deficiency in tracking to occlusion/fails in case of occlusion and different viewpoints.
2) Despite the effectiveness of Kalman Filter, it returns a relatively higher number of ID switches.

## How DeepSORT solves these issues?
* The authors of the SORT algorithm proposed DeepSORT to address the issues in the SORT algorithm.
* These issues are because of the association metric used.
* In DeepSORT another distance metric is introduced based on the appearance of the object. 
* The appearance feature vector is Deep Appearance Descriptor.
* DeepSORT uses a better association metrics which combines both motion and appearance descriptors.
* DeepSORT can be defined as a tracking algorithm which track object not only based on the velocity and motion of the object but also based on the appearance of the object.

## Implementation of Traffic Counter
* An empty dictionary is intialized which contains detected object name with the total count
* Intersection between an imaginary marker line and trail line is identified for traffic counting

## Implementation of Velocity Calculation
* Distance is estimated using the euclidean formula
* Pixel per meter (ppm) parameter can be set static (ppm = 8) or dynamic (ppm = 20 when vehicle is closer and ppm = 1 when away from the camera)
* Distance in meters (d_meters) will be distance in pixels / ppm
* Since FPS=15 here, the time constant = FPS * 3.6 (an adjusted constant for calibrated results)
* Speed will be d_meters * time constant

## Implementation of Vehicle Counting (Entry and Leaving)
* Vehicles entrying are moving North while the ones leaving are travelling South
* Based on the intersection and direction, two diff dictionaries are maintained to find entry and exit

# DETECTION RESULTS
## test1.mp4
<img width="1274" alt="image" src="https://github.com/sahilfaizal01/Multi-Vehicle-Tracking-Counting-and-Velocity-Calculation/assets/106440078/e702b7e3-1b12-4a1f-a707-adec34cd6ca5">

## test4.mp4
<img width="1274" alt="image" src="https://github.com/sahilfaizal01/Multi-Vehicle-Tracking-Counting-and-Velocity-Calculation/assets/106440078/7bcb339d-4191-4da9-bd65-f74b3954c54c">

## test5.mp4
<img width="955" alt="image" src="https://github.com/sahilfaizal01/Multi-Vehicle-Tracking-Counting-and-Velocity-Calculation/assets/106440078/00e2aae0-cd34-4ba6-8ef0-6ca3c9d3d718">

## test6.mp4
<img width="955" alt="image" src="https://github.com/sahilfaizal01/Multi-Vehicle-Tracking-Counting-and-Velocity-Calculation/assets/106440078/05df862b-9314-4bff-b1f3-ce227f723ccf">

## test7.mp4
<img width="955" alt="image" src="https://github.com/sahilfaizal01/Multi-Vehicle-Tracking-Counting-and-Velocity-Calculation/assets/106440078/24f5acb2-8365-4467-8510-6bac3abc50fc">




