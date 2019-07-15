from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import imutils
from imutils.video import VideoStream
import argparse
import pickle as pkl
import copy
#from boxmaster import BoxMaster


def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_


# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create,
    "goturn": cv2.TrackerGOTURN_create
}

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    parser.add_argument("-v", "--video", type=str,
                    help="path to input video file")
    parser.add_argument("-t", "--tracker", type=str, default="kcf",
                    help="OpenCV object tracker type")
    return parser.parse_args()



def common_interval(a1, a2, b1, b2):
    if a1 >= b1 and a2 <= b2:
        intersection = a2 - a1
    elif a1 < b1 and a2 > b2:
        intersection = b2 - b1
    elif a1 < b1 and a2 > b1:
        intersection = a2 - b1
    elif a2 > b2 and a1 < b2:
        intersection = b2 - a1
    else:
        intersection = 0

    return intersection

def common_area(box1, box2):
#    print(box1, box2)
    width = common_interval(box1[0], box1[0] + box1[2], box2[0], box2[0] + box2[2])
    height = common_interval(box1[1], box1[1] + box1[3], box2[1], box2[1] + box2[3])
    return width * height

def area(box):
    return box[2]*box[3]


if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    # if a video path was not supplied, grab the reference to the web cam
    if not args.video:
        print("[INFO] starting video stream...")
        cap = cv2.VideoCapture(0)

    # otherwise, grab a reference to the video file
    else:
        cap = cv2.VideoCapture(args.video)

    time.sleep(1.0)

    frames = 0
    trackers = []
    idle_begin = {}
    last_frame = {}
    count = 0
    tracked_people = 0
    wait_time = 2
    last_detect_time = time.time() - 1
    max_tracked_people = 10
    accuracy = 0.2
    start = time.time()    
    while cap.isOpened():

        ret, frame = cap.read()
        # resize the frame (so we can process it faster)
        frame = imutils.resize(frame, width=640)

        # check to see if we have reached the end of the stream
        if ret:

            if len(trackers) < max_tracked_people and time.time() - last_detect_time >= wait_time:
                img, orig_im, dim = prep_image(frame, inp_dim)

                im_dim = torch.FloatTensor(dim).repeat(1,2)

                if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()

                output = model(Variable(img), CUDA)
                output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

                # if type(output) == int:
                #     frames += 1
                #     print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                #     cv2.imshow("frame", orig_im)
                #     key = cv2.waitKey(1)
                #     if key & 0xFF == ord('q'):
                #         break
                #     continue


                output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim

    #            im_dim = im_dim.repeat(output.size(0), 1)
                output[:,[1,3]] *= frame.shape[1]
                output[:,[2,4]] *= frame.shape[0]
                print(output[:, 1:5])


                print(frame.shape[1], frame.shape[0])
                classes = load_classes('data/coco.names')
                colors = pkl.load(open("pallete", "rb"))


                list(map(lambda x: write(x, orig_im), output))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1500)
                if key & 0xFF == ord('q'):
                     break

                human_boxes = []

                for object in output:
                    if classes[int(object[-1])] == "person":
                        obj_data = tuple(object[1:5].cpu().numpy())
                        box = (obj_data[0], obj_data[1], obj_data[2] - obj_data[0], obj_data[3] - obj_data[1])
                        human_boxes.append(box)

                number_of_persons = 0

                for tracker in list(trackers):
                    max_match = 0
                    max_human = None
                    (success, box) = tracker.update(orig_im)
                    for human_box in list(human_boxes):
                        com_area = common_area(box, human_box)
                        if com_area == 0:
                            continue
                        area_box = area(box)
                        area_human_box = area(human_box)
                        match = 0
                        if area_box > 0 and area_human_box > 0:
                            match = (com_area/area_box + com_area/area_human_box)/2
                        if match > max_match:
                            max_match = match
                            max_human = human_box
                    print("Matching: ", max_match)
                    if max_match == 0:
                        trackers.remove(tracker)
                    elif max_match >= accuracy:
                        trackers.remove(tracker)
                        new_tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
                        new_tracker.init(orig_im, max_human)
                        trackers.append(new_tracker)
                        number_of_persons += 1
                        idle_begin[new_tracker] = 0
                        human_boxes.remove(max_human)
                    else:
                        trackers.remove(tracker)

                index = 0
                for box in human_boxes:
                    number_of_persons += 1
                    if len(trackers) < max_tracked_people:
                        if box[2] <= 1 or box[3] <= 1:
                            continue

                        tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
                        tracker.init(frame, box)
                        trackers.append(tracker)
                        idle_begin[tracker] = 0
                        tracked_people += 1

                print("People count: ", number_of_persons)
                print(frame.shape[1], frame.shape[0])
                #update last time people were detected
                last_detect_time = time.time()
                print(count)

            else:

                #print(" TRACKERS :", len(trackers))
                frame_copy = copy.copy(frame)
                for tracker in list(trackers):
                    (success, box) = tracker.update(frame_copy)
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if not success:
                        if idle_begin[tracker] < 3:
                            idle_begin[tracker] += 1
                        elif idle_begin[tracker] == 3:
                            idle_begin[tracker] = time.time()
                        elif abs(time.time() - idle_begin[tracker]) > 3:
                            trackers.remove(tracker)
                            tracked_people -= 1
                    elif common_area((0, 0, frame.shape[1], frame.shape[0]), (x, y, w, h))/(w*h) < 0.5:
                        trackers.remove(tracker)
                        count += 1
                        tracked_people -= 1
                    else:
                        idle_begin[tracker] = 0
                        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

                # list(map(lambda x: write(x, orig_im), output))
                # objs = [classes[int(x[-1])] for x in output]
                # number_of_persons = 0
                # total_objs = 0
                # for detected_obj in objs:
                #     if detected_obj == "person":
                #         number_of_persons += 1
                #     total_objs += 1
                # print("Number of people: " + str(number_of_persons) + ". Total number of objects: " + str(total_objs))
                #
                # cv2.imshow("frame", orig_im)
                # key = cv2.waitKey(1)
                # if key & 0xFF == ord('q'):
                #     break
                # frames += 1
                # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            
        else:
            cap.release()
            break
    

    
    

