#The argparse module makes it easy to write user-friendly command-line interfaces.
import argparse
import time
from pathlib import Path

# To implement object detection and tracking, we need opencv-python library, which is imported as cv2
import cv2
#As YOLOv7 is built using PyTorch, To perform object detection/tracking using YOLOv7 we need to improt the PyTorch module.
#To use the PyTorch library we do, import torch
# To check the version of the PyTorch library we do print(torch. __version__)
import torch
#torch.backends controls the behavior of various backends that PyTorch supports.
#These backends include:
#torch.backends.cuda
#torch.backends.cudnn
import torch.backends.cudnn as cudnn

#The random is a module present in the NumPy library.
# This module contains the functions which are used for generating random numbers
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

#Deque (Doubly Ended Queue) in Python is implemented using the module “collections“.
#Deque is preferred over a list in the cases where we need quicker append and pop operations from both the ends of the container
from collections import deque
# To convert a list into an array we use the numpy library
import numpy as np
import math
# Defining the color palette, because we will have a wide variety of objects, we want to establish a color palette,
#So the color of the bounding box and the tracking trails will be same for each object
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# Initializing a Dictionary by the name data_deque
data_deque = {}

#Created a Object Counter Dictionary, which will store each of the detected object name with the totalcount, like 4 cars have been
#detected, 6 trucks in the complete video
# In the Object Counter Dictionary we will store the count of vehicles Leaving, and we will have something like this
#objct_counter={'car':4, truck:'5', motorcycle:'10'}
object_counter = {}
# In the Object Counter 1 Dictionary we will store the count of vehicles Entering, and we will have something like this
#objct_counter1={'car':14, truck:'10', motorcycle:'13'}
object_counter1 = {}


#Creating a Line Manually in the Video, so when vehicle trail intersect this line, we will do an increment in the count
#of the vehicle that has passed
line = [(244,440), (1050, 456)]

#Creating a dictionary, by the name speed_line_queue, using object tracking we will assign a unique id to each of the
#detected object and in the speed_line_queue dictionary we will store the unique id of each of the detected object as the key
# with its estimated speed as the value

speed_line_queue = {}
#{'67':[100, 121, 90], '2',[190,121,210]}
#In the def xyxy_to_xywh function, we are converting the bounding box output received from YOLOv7
#to a format that is competible with DeepSORT, Using this function we convert our x any y coordinattes to
#center coordinates which is x_c and y_c and return the width and height of the bounding boxes
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

#Converting the xy coordinates into another format but in this case we are only looking for Top, Left and width and height
#This function return the Top Left Coordinates along with the width and heght of the bounding boxes
def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs
#To compute colors for our labels we select a color from our color palette and return
#a tuple of colors
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
#Using the draw_border function i will create a rounded rectangle over the bounding box in which i will then
#put the text
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img
# In UI_box function we are passing cv2.rectangle to create a rectangle around the detected object
#Plus also we are using cv2.text, to add the label for example what object it is, i.e car, bus, in the rounded rectangle
#cv2.text will add label in the rounded rectangle which we have created using draw_border
def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def estimateSpeed(location1, location2): ###
    #Euclidean Distance Formula
    #d = √[(x2 – x1)2 + (y2 – y1)2].
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	#We can make it dynamic closer to camera 20 pixel per meter and away from camera one pixel per meter
	ppm = 8 #Pixels per Meter
    #d_meters stands for distance in meters while d_pixels stands for distance in pixels,
    # we have calculated the distance in pixels in code using the Euclidean Distance Formula.
	d_meters = d_pixels / ppm
    # 15 refers to 15 frame per second, we can play with this constant to get
    # more calibrated results, while 3.6 is the constant which we can adjust.
	time_constant = 15 * 3.6
    #speed = distance/time
    #time = 1/frequency
    #time_constant here refers to frequency
	speed = d_meters * time_constant
	return int(speed)
def intersect(A, B, C,D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0]-A[0])
def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""
    return direction_str
# In draw_boxes function, i am calling the above UI_box function
# to draw the bounding boxes around the detected object, assign unique id's to each detected object,
# adding labels and draw the trails  for tracking

def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    cv2.line(img, line[0], line[1], (46,162,112), 3)
    # As we are doing detection frame by frame, so here i am checking the height and width of the current frame
    height, width, _ = img.shape
    # We will store the unique id of the detected object, until the object is in the frame, if the object
    # disappears from the frame, we will remove  the unique id of the object from the list and will also
    #save the unique id of the new object appearing in the frame
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)
    # Using .pop key we remove the ID of the object from the data_deque list, if the object is no more in the frame


    #Here we will loop through the bounding boxes one by one
    #So here we have all the four coordinates x1, x2, y1, y2.
    #x1y1 represents the top left corner of the bounding box and x2y2 represents the bottom right corner of the bounding
    #box
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge of the bounding box
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # # get a unique ID of each object
        id = int(identities[i]) if identities is not None else 0

        #if the new object appears in the frame, so after assigning it a unique id in the above step,
        #in the next step, we will create a new buffer/ a new list deque which has maximum length of 64,
        #so as we try to add the 65th element, the first element will be out
        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= opt.trailslen)
          speed_line_queue[id] = []
        # Setting a unique color for each object bounding box and rounded rectangle which contains the label
        color = compute_color_for_labels(object_id[i])
        #Object ID, contains the ID of the object, for example for the Person Class the ID will be zero as per the COCO format
        #Using the Object ID we find the object name.
        obj_name = names[object_id[i]]
        # Setting the Label in the Required Format
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # add center to buffer
        #Using  center position of the bounding box of the tracked object and every time we see the object in the other
        #frame we will connect centers to form a series of line. Then we will understand path the object took.
        data_deque[id].appendleft(center)

        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            obj_speed = estimateSpeed(data_deque[id][1], data_deque[id][0])
            speed_line_queue[id].append(obj_speed)
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                if "South" in direction:
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
                if "North" in direction:
                    if obj_name not in object_counter1:
                        object_counter1[obj_name] = 1
                    else:
                        object_counter1[obj_name] += 1
        try:
            label = label + " " + str(sum(speed_line_queue[id])//len(speed_line_queue[id])) + "km/h"  ##
        except :
            pass
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(opt.trailslen / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
        # 4. Display Count in top right corner
        for idx, (key, value) in enumerate(object_counter1.items()):
            #objct_counter1 = {'car': 14, truck: '10', motorcycle: '13'}
            cnt_str = str(key) + ":" + str(value)
            cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
            cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.line(img, (width - 150, 65 + (idx * 40)), (width, 65 + (idx * 40)), [85, 45, 255], 30)
            cv2.putText(img, cnt_str, (width - 150, 75 + (idx * 40)), 0, 1, [255, 255, 255], thickness=2,
                        lineType=cv2.LINE_AA)

        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str1 = str(key) + ":" + str(value)
            cv2.line(img, (20, 25), (500, 25), [85, 45, 255], 40)
            cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.line(img, (20, 65 + (idx * 40)), (127, 65 + (idx * 40)), [85, 45, 255], 30)
            cv2.putText(img, cnt_str1, (11, 75 + (idx * 40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    return img
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names)) 
def detect(save_img=False):
    names, source, weights, view_img, save_txt, imgsz, trace = opt.names, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace 
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # initialize deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = load_classes(names)
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                # Initializing List, which will contain our center coordinates with width and height of the bounding boxes
                # As we process our predictions, we will append these bounding boxes to these list
                xywh_bboxs = []
                # Same as above we will apend confidence value in this list
                confs = []
                # Same for the oids (object id), we will append the Object ID's into it
                oids = []
                # Write results
                # Here we are looping over all the detections because  YOLOv7 predictions give us the corner coordinates like
                # x1, y1, x2, y2 but DeepSORT requires the center coordinates and the height and the width
                for *xyxy, conf, cls in reversed(det):
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    #if save_img or view_img:  # Add bbox to image
                        #label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                # Now here we are using the deepsort update function, the reason this function is called the update function
                # because it takes all the detections and just add the identities
                # DeepSORT just assign unique id, the ids are not incremental
                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    # DrawBoxes function to draw the bounding boxes, label them and show the ID of the tracker
                    draw_boxes(im0, bbox_xyxy, names, object_id,identities)
                    # Extract tracked object's bounding box coordinates
                    for i, box in enumerate(bbox_xyxy):
                      x1, y1, x2, y2 = [int(i) for i in box]
                      # increment frequency counter for whole bounding box
                      global_img_np_array[y1:y2,x1:x2] += 1
                    # heatmap array pre-processing
                    global_img_np_array_norm = (global_img_np_array - global_img_np_array.min())/(global_img_np_array - global_img_np_array.max())
                    global_img_np_array_norm = global_img_np_array.astype('uint8')
                    # apply gaussian blur and draw heatmap
                    global_img_np_array_norm = cv2.GaussianBlur(global_img_np_array_norm,(9,9),0)
                    heatmap_img = cv2.applyColorMap(global_img_np_array_norm,cv2.COLORMAP_JET)
                    # super_imposed_img = None
                    # Overlap heatmaps on video frames
                    super_imposed_img = cv2.addWeighted(heatmap_img,0.5,im0,0.5,0)
                    # visualize the heatmap
                    cv2.imshow('Heatmap',super_imposed_img)
                    cv2.waitKey(1)
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), super_imposed_img)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, super_imposed_img)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            global_img_np_array = np.ones([int(h),int(w)], dtype=np.uint32)
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(super_imposed_img)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
