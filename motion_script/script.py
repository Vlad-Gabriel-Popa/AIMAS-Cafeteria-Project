import argparse
from datetime import datetime
import os
import os.path
import time
import cv2
from imutils.video import VideoStream
from motion_detector import MotionDetector


# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path/URL to the video file")
ap.add_argument("-u", "--username", type=str,
                default="", help="enter username (if URL needs BASIC authorization)")
ap.add_argument("-pass", "--password", type=str,
                default="", help="enter password (if URL needs BASIC authorization)")
ap.add_argument("-path", "--path_to_save", type=str,
                default="images", help="path where images will be saved")
ap.add_argument("-l", "--logs_folder", type=str,
                default="logs", help="path where logs will be saved")
ap.add_argument("-pid", "--pid_file", type=str,
                default="pid", help="file where pid of the process will be saved")
ap.add_argument("-time", "--wait-time", type=int,
                default=10, help="time to wait before saving another photo on the disk")
ap.add_argument("-a", "--min-area", type=int,
                default=100, help="minimum area size for an object to be detected")
# Threshold value for movment detection
# (the smaller, the more sensitive the detection will be)
ap.add_argument("-thr", "--threshold_value", type=int,
                default=12, help="threshold value for movement detection")
args = vars(ap.parse_args())


# Add '/' at the end of the paths
if not args["path_to_save"].endswith("/"):
    args["path_to_save"] += "/"
if not args["logs_folder"].endswith("/"):
    args["logs_folder"] += "/"

# Create paths if they don't already exist
if not os.path.exists(args["path_to_save"]):
    os.makedirs(args["path_to_save"])
if not os.path.exists(args["logs_folder"]):
    os.makedirs(args["logs_folder"])

# Motion detecting object
md = MotionDetector(args["min_area"], args["threshold_value"])

# Date and time when the last photo was saved
start_time = datetime.now()
log_time = datetime.now()
timestamp = datetime.now()


log_file = open(args["logs_folder"] +
                timestamp.strftime("%Y-%m-%d"), "a+", buffering=1)
log_file.write("\nNew session started at " +
               timestamp.strftime("Day:%d-%m-%Y Hour: %H:%M:%S") + "\n\n")


pid = str(os.getpid())
pidfile = open(args["pid_file"], "w+")
pidfile.write(os.path.basename(__file__) + '\n')
pidfile.write(pid)
pidfile.close()

# Capture the video using the path/URL/webcam
if args.get("video", None) is None:
    capture = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    # If username and password were provided then insert them in the URL
    if args["username"] and args["password"] and "//" in args["video"]:
        index = args["video"].index("//")
        args["video"] = args["video"][:(
            index+2)] + args["username"] + ':' + args["password"] + '@' + args["video"][(index+2):]
    capture = cv2.VideoCapture(args["video"])

while 1:
    # Get a frame and calculate current time
    frame = capture.read()
    if frame is not None:
        current_time = datetime.now()
        delta_time = current_time - timestamp
        # If WAIT_TIME seconds have passed and motion is detected, save the image
        if md.is_motion_detected(frame) and delta_time.total_seconds() >= args["wait_time"]:
            cv2.imwrite(
                args["path_to_save"] + current_time.strftime("%Y-%m-%d_%H-%M-%S.jpg"), frame)
            # Write to log file
            log_file.write(
                "Saved: " + current_time.strftime("%Y-%m-%d_%H-%M-%S.jpg\n"))
            # Update timestamp
            timestamp = current_time

        log_delta_time = (current_time - log_time).total_seconds()
        if log_delta_time >= 5.0:
            log_file.write("\nStill alive after " + str((current_time - start_time).total_seconds()) +
                           " seconds. Current time: " + current_time.strftime("%H:%M:%S (%d-%m-%Y)\n"))
            log_time = current_time

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        print("Video ended.")
        break
log_file.close()
if args.get("video", None) is None:
    capture.stop()
else:
    capture.release()
cv2.destroyAllWindows()
