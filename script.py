import argparse
from datetime import datetime
import os
import cv2
from motion_detector import MotionDetector


# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path/URL to the video file")
ap.add_argument("-p", "--path_to_save", type=str,
                default="images/", help="path where images will be saved")
ap.add_argument("-time", "--wait-time", type=int,
                default=10, help="time to wait before saving another photo on the disk")
ap.add_argument("-a", "--min-area", type=int,
                default=100, help="minimum area size for an object to be detected")
# Threshold value for movment detection
# (the smaller, the more sensitive the detection will be)
ap.add_argument("-thr", "--threshold_value", type=int,
                default=12, help="threshold value for movement detection")
args = vars(ap.parse_args())

# Add '/' at the end of the path
if not args["path_to_save"].endswith("/"):
    args["path_to_save"] += "/"

# Motion detecting object, the parameter is the
md = MotionDetector(args["min_area"], args["threshold_value"])

# Date and time when the last photo was saved
timestamp = datetime.now()

# Create path if it doesn't already exist
if not os.path.exists(args["path_to_save"]):
    os.makedirs(args["path_to_save"])

# Capture the video using the path/URL/webcam
if args.get("video", None) is None:
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture(args["video"])

while 1:
    # Get a frame and calculate current time
    result, frame = capture.read()
    if result:
        current_time = datetime.now()
        delta_time = current_time - timestamp
        # If WAIT_TIME seconds have passed and motion is detected, save the image
        if delta_time.total_seconds() >= args["wait_time"] and md.is_motion_detected(frame):
            cv2.imwrite(
                args["path_to_save"] + current_time.strftime("%Y-%m-%d_%H-%M-%S.jpg"), frame)
            timestamp = current_time

        # Display image
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        print("Video ended.")
        break

capture.release()
cv2.destroyAllWindows()
