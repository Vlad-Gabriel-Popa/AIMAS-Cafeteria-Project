import imutils
import cv2
import argparse
import time

from yolo import Yolo
from boxmaster import BoxMaster


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    parser.add_argument("--track", dest='track', help=
    "Maximum number of tracked people",
                        default=10, type=int)
    parser.add_argument("--video", type=str,
                        help="path to input video file")
    parser.add_argument("--tracker", type=str, default="kcf",
                        help="OpenCV object tracker type")
    return parser.parse_args()



if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    #bbox_attrs = 5 + num_classes

    inp_dim = int(args.reso)
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    yolo = Yolo(confidence, nms_thesh, args.reso, cfgfile, weightsfile, num_classes)


    # if a video path was not supplied, grab the reference to the web cam
    if not args.video:
        cap = cv2.VideoCapture(0)

    # otherwise, grab a reference to the video file
    else:
        cap = cv2.VideoCapture(args.video)

    time.sleep(1.0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps > 0

    frames = 0
    wait_time = 3
    last_yolo_time = -2
    max_tracked_people = args.track
    box_master = BoxMaster(0.4, args.tracker, max_tracked_people, yolo)

    while cap.isOpened():
        ret, frame = cap.read()
        # resize the frame (so we can process it faster)
        # check to see if we have reached the end of the stream

        frames += 1
        timestamp = frames/fps

        if ret:
            frame = imutils.resize(frame, width=640)
            box_master.update(frame, timestamp)

            if timestamp - last_yolo_time >= wait_time:
                human_boxes = yolo.get_human_boxes(frame, inp_dim)
                box_master.yolo_detect(frame, human_boxes, timestamp)
                last_yolo_time = timestamp

        else:
            cap.release()
            break