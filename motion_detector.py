import imutils
import cv2


class MotionDetector:

    def __init__(self, min_area, tresh_value):
        # remeber 2 frames before the current frame (in grayscale)
        self.last_last_frame = None
        self.last_frame = None
        # minimum area for an object to be detected
        self.min_area = min_area
        # threshold value for motion detection
        self.tresh_value = tresh_value

    def is_motion_detected(self, frame):
        """
        returns true if motion is detected
        """

        if frame is None:
            return False

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if last_frame is None, initialize it
        if self.last_frame is None:
            self.last_frame = gray
            return False

        # if last_last_frame is None, initialize it
        if self.last_last_frame is None:
            self.last_last_frame = self.last_frame
            self.last_frame = gray
            return False
        # compute the absolute difference between the current frame and
        # the frame that came 2 frames before
        frame_delta = cv2.absdiff(self.last_last_frame, gray)
        thresh = cv2.threshold(
            frame_delta, self.tresh_value, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        contour = None
        for contour in contours:
                # if the contour is big enough, return true
            if cv2.contourArea(contour) >= self.min_area:
                self.last_last_frame = self.last_frame
                self.last_frame = gray
                return True
        self.last_last_frame = self.last_frame
        self.last_frame = gray
        return False
