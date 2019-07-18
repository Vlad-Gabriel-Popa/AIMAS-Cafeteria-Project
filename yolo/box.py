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
    def __init__(self, tracker, box, timestamp):
        self.time_created = timestamp
        self.directions = (Direction.NONE, Direction.NONE)
        self.movements = {}
        self.position = box
        self.last_time = timestamp
        self.speed = {}
        self.is_standing_in_queue = False
        self.tracker = tracker
        self.is_tracker_on = True
        self.last_time_alive = timestamp
        self.times = 0
        self.start_positionOx = (2*box[0] + box[2])/2

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
        if frame_time - self.time_created > 10:
            self.is_standing_in_queue = True
        (success, box) = self.tracker.update(frame_in)
        if success:
            self.is_tracker_on = True
            if not self.position:
                self.position = box
                self.last_time = timestamp
                self.speed = dict.fromkeys(self.speed, 0)
            else:
                (x, y, w, h) = [float(v) for v in box]
                distanceOX = (x + x + w)/2 - (self.position[0] + self.position[0] + self.position[2])/2
                distanceOY = (y + y + h)/2 - (self.position[1] + self.position[1] + self.position[3])/2
                directionOX = Direction.RIGHT if distanceOX >= 0 else Direction.LEFT
                directionOY = Direction.DOWN if distanceOY >= 0 else Direction.UP
                directions = sorted(self.movements, key=self.movements.get)

                self.movements[directionOX] += abs(distanceOX)
                self.movements[directionOY] += abs(distanceOY)
                self.speed = dict.fromkeys(self.speed, 0)

                self.speed[Direction.LEFT] = self.movements[Direction.LEFT] / (timestamp - self.time_created)
                self.speed[Direction.RIGHT] = self.movements[Direction.RIGHT] / (timestamp - self.time_created)
                self.speed[Direction.UP] = self.movements[Direction.UP] / (timestamp - self.time_created)
                self.speed[Direction.DOWN] = self.movements[Direction.DOWN] / (timestamp - self.time_created)

                speedsOx = [self.speed[Direction.LEFT], self.speed[Direction.RIGHT]]
                speedsOy = [self.speed[Direction.UP], self.speed[Direction.DOWN]]
                max_directionOx = Direction(speedsOx.index(max(speedsOx)) + 1)
                max_directionOy = Direction(speedsOy.index(max(speedsOy)) + 3)
                self.directions = (max_directionOx, max_directionOy)
                #self.directions = (directionOX, directionOY)
               # self.speed[directionOX] = abs(distanceOX) / (frame_time - self.last_time)
                #self.speed[directionOY] = abs(distanceOY) / (frame_time - self.last_time)

                self.position = box
                self.last_time = timestamp
            self.last_time_alive = timestamp
            return box
        else:
            self.is_tracker_on = False
            return None

    def area(self):
        return self.position[2] * self.position[3]

    def set_tracker(self, new_tracker, closest_box, timestamp):

        self.tracker = new_tracker
        self.position = closest_box
        #box.speed = dict.fromkeys(box.speed, 0)
        self.last_time_alive = timestamp

        (x, y, w, h) = [float(v) for v in closest_box]
        distanceOX = (x + x + w) / 2 - (self.position[0] + self.position[0] + self.position[2]) / 2
        distanceOY = (y + y + h) / 2 - (self.position[1] + self.position[1] + self.position[3]) / 2
        directionOX = Direction.RIGHT if distanceOX >= 0 else Direction.LEFT
        directionOY = Direction.DOWN if distanceOY >= 0 else Direction.UP

        self.movements[directionOX] += abs(distanceOX)
        self.movements[directionOY] += abs(distanceOY)
