import time
import cv2
import operator
from enum import Enum


class Direction(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class Box:
    def __init__(self, tracker, box, timestamp, yolo):
        self.time_created = time.time()
        self.directions = (Direction.NONE, Direction.NONE)
        self.movements = {}
        self.position = box
        self.last_time = timestamp
        self.speed = {}
        self.is_standing_in_queue = False
        self.tracker = tracker
        self.is_tracker_on = True
        self.last_time_alive = timestamp
        self.yolo = yolo

        self.movements[Direction.NONE] = 0
        self.movements[Direction.LEFT] = 0
        self.movements[Direction.RIGHT] = 0
        self.movements[Direction.UP] = 0
        self.movements[Direction.DOWN] = 0

        self.speed[Direction.NONE] = 0
        self.speed[Direction.LEFT] = 0
        self.speed[Direction.RIGHT] = 0
        self.speed[Direction.UP] = 0
        self.speed[Direction.DOWN] = 0

    def update(self, frame_in, timestamp):

        frame_time = timestamp
        if frame_time - self.last_time > 10:
            self.is_standing_in_queue = True
        (success, box) = self.tracker.update(frame_in)
        frame_out = frame_in.copy()
        if success:
            self.is_tracker_on = True
            (x, y, w, h) = [float(v) for v in box]
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if not self.position:
                self.position = box
                self.last_time = timestamp
                self.speed = dict.fromkeys(self.speed, 0)
            else:
                distanceOX = (x + x + w)/2 - (self.position[0] + self.position[0] + self.position[2])/2
                distanceOY = (y + y + h)/2 - (self.position[1] + self.position[1] + self.position[3])/2
                directionOX = Direction.RIGHT if distanceOX >= 0 else Direction.LEFT
                directionOY = Direction.DOWN if distanceOY >= 0 else Direction.UP
                directions = sorted(self.movements, key=self.movements.get)
                self.directions = (directionOX, directionOY)

                self.movements[directionOX] += abs(distanceOX)
                self.movements[directionOY] += abs(distanceOY)
                self.speed = dict.fromkeys(self.speed, 0)
                self.speed[directionOX] = abs(distanceOX) / (frame_time - self.last_time)
                self.speed[directionOY] = abs(distanceOY) / (frame_time - self.last_time)

                self.position = box
                self.last_time = timestamp
            self.last_time_alive = timestamp
            return frame_out
        else:
            self.is_tracker_on = False
            time_elapsed = timestamp - self.last_time_alive
            if time_elapsed > 1:
                (x, y, w, h) = [float(v) for v in self.position]
                eps = 10

                (directionOX, directionOY) = self.directions
                distanceOX = self.speed[directionOX] * time_elapsed
                distanceOY = self.speed[directionOY] * time_elapsed

                if directionOX == Direction.LEFT: distanceOX *= (-1)
                if directionOY == Direction.UP: distanceOY *= (-1)

                if self.is_standing_in_queue:
                    distanceOX = 0
                    distanceOY = 0

                leftY = y + distanceOY - eps if y + distanceOY - eps >= 0 else 0
                rigthY = y + h + distanceOY + eps if y + h + distanceOY + eps < frame_in.shape[0] else frame_in.shape[0]
                leftX = x + distanceOX - eps if x + distanceOX - eps >= 0 else 0
                rigthX = x + w + distanceOX + eps if x + w + distanceOX + eps < frame_in.shape[1] else frame_in.shape[1]
                img = frame_in[leftY:rigthY, leftX:rigthX].copy()
                human_boxes = self.yolo.get_human_boxes(frame_in, )
                if not len(human_boxes):


            return None

    def area(self):
        return self.position[2] * self.position[3]
