#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
import imutils
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def common_interval(a1, a2, b1, b2):
    if b1 <= a1 and a2 <= b2:
        intersection = a2 - a1
    elif a1 < b1 and b2 < a2:
        intersection = b2 - b1
    elif a1 < b1 and b1 < a2:
        intersection = a2 - b1
    elif a1 < b2 and b2 < a2:
        intersection = b2 - a1
    else:
        intersection = 0

    return intersection


def common_area(box1, box2):
    width = common_interval(box1[0], box1[0] + box1[2], box2[0], box2[0] + box2[2])
    height = common_interval(box1[1], box1[1] + box1[3], box2[1], box2[1] + box2[3])
    return width * height


def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, line=1100, max_age=30)

    writeVideo_flag = True
    output_filee = open("rezultate", "w")

    video_capture = cv2.VideoCapture("video.mp4")
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 

    fps = 0.0
    detection_rate = 1
    timee = 0
    line = 1100
    frame_no = -1
    res = []
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3

        if ret != True:
            break
        t1 = time.time()
        frame_no += 1
       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb


        if frame_no % detection_rate == 0 or (frame_no + 1) % (detection_rate) == 0 or (frame_no + 2) % (detection_rate) == 0:
            print("Rest:", time.time() - timee)
            timee = time.time()
            boxs = yolo.detect_image(image)
            features = encoder(frame, boxs)
            print("Yolo", time.time() - timee)
            timee = time.time()

            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            #detections = [detection for detection in detections if  detection.tlwh[0] +  detection.tlwh[2] < line]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

        else:
            tracker.predict()

        print(tracker.get_next_id())
           # print("box_num",len(boxs))
        print(res)
        for track in list(tracker.tracks):
            #if not track.is_confirmed(): #or track.time_since_update > 1:
            #    continue
            color = (255,255,255)

            box = track.to_tlwh()
            if not track.exited and track.age > 60 and box[2] and box[3] and \
                    track.time_since_update < 5 and box[0] + box[2] > line:
                starting_box = track.start_to_tlwh()
                real_time = track.age/video_fps
                distance = abs(starting_box[0] - box[0])
                distance += abs(starting_box[1] - box[1])
                res.append(str(distance) + " " + str(real_time))
                output_filee.write(str(distance) + " " + str(real_time))
                output_filee.write("\n")
                track.exited = True
                track.exited_since = track.age

            if track.exited:
                color = (0, 255, 0)
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        if frame_no % detection_rate == 0 or (frame_no + 1) % (detection_rate) == 0 or (frame_no + 2) % (detection_rate) == 0:
            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        cv2.line(frame, (int(line), 0), (int(line), frame.shape[1]), (0, 255, 255), 2)
        cv2.imshow('', frame)


        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
