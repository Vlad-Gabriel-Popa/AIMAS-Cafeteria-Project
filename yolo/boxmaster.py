import cv2
from box import Box
from box import Direction
from yolo import Yolo

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


def area(box):
    return box[2]*box[3]


def find_closest(box, boxes):
    x = (2 * box[0] + box[2])/2
    y = (2 * box[1] + box[3])/2

    return max(boxes, key=lambda a_box: (x - (2 * a_box[0] + a_box[2])/2) + (y - (2 * a_box[1] + a_box[3])/2))


class BoxMaster:
    def __init__(self, accuracy, tracker_type, max_tracked_people, yolo):
        self.boxes = []
        self.accuracy = accuracy
        self.tracker_type = tracker_type
        self.max_tracked_people = max_tracked_people
        self.people_count = 0
        self.yolo = yolo
        self.count = 0
        self.info = []

    def yolo_detect(self, frame, yolo_boxes, timestamp):

        self.people_count = 0
        for box in list(self.boxes):
            max_match = 0
            max_human = None
            for human_box in list(yolo_boxes):
                com_area = common_area(box.position, human_box)
                if com_area == 0:
                    continue
                area_box = area(box.position)
                area_human_box = area(human_box)
                match = 0
                if area_box > 0 and area_human_box > 0:
                    match = (com_area / area_box + com_area / area_human_box) / 2
                if match > max_match:
                    max_match = match
                    max_human = human_box
            print("Matching: ", max_match)
            if max_match == 0:
                self.boxes.remove(box)
            elif max_match >= self.accuracy:
                new_tracker = OPENCV_OBJECT_TRACKERS[self.tracker_type]()
                new_tracker.init(frame, max_human)
                box.tracker = new_tracker
                self.people_count += 1
                yolo_boxes.remove(max_human)
            else:
                self.boxes.remove(box)

        for box in yolo_boxes:
            self.people_count += 1
            if len(self.boxes) < self.max_tracked_people:
                if box[2] < 1 or box[3] < 1:
                    continue

                tracker = OPENCV_OBJECT_TRACKERS[self.tracker_type]()
                tracker.init(frame, box)
                self.boxes.append(Box(tracker, box, timestamp))
                self.people_count += 1

        print("People count: ", self.people_count)

    def update(self, frame, timestamp):
        result_frame = frame.copy()
        track_boxes = []
        for box in list(self.boxes):
        #    print("AAAA:   ", common_area((0, 0, frame.shape[1], frame.shape[0]), box.position) / (box.area()))
            if common_area((0, 0, frame.shape[1], frame.shape[0]), box.position) / (box.area()) < 0.95:
                distance_traveled = max(box.movements[Direction.LEFT], box.movements[Direction.RIGHT])
                if box.position[0] <= 0:
                    distance_traveled = box.start_positionOx

                if box.position[0] + box.position[2] >= frame.shape[1]:
                    distance_traveled = frame.shape[1] - box.start_positionOx

                time_traveled = timestamp - box.time_created
                target_distance = frame.shape[1]
                if time_traveled > 2:
                    self.info.append((target_distance * time_traveled / distance_traveled, time_traveled, distance_traveled))
                self.boxes.remove(box)
                self.count += 1
                continue

            box_aux = box.update(frame, timestamp)
            if box_aux:
                track_boxes.append(box_aux)

            else:
                time_elapsed = timestamp - box.last_time_alive
                if time_elapsed > 0.1:
                    if box.times >= 3:
                        self.boxes.remove(box)
                        continue
                    box.times += 1
                    (x, y, w, h) = [float(v) for v in box.position]
                    eps = 100

                    (directionOX, directionOY) = box.directions
                    distanceOX = box.speed[directionOX] * time_elapsed
                    distanceOY = box.speed[directionOY] * time_elapsed

                    if directionOX == Direction.LEFT:
                        distanceOX *= (-1)
                    if directionOY == Direction.UP:
                        distanceOY *= (-1)

                    if box.is_standing_in_queue:
                        distanceOX = 0
                        distanceOY = 0

                   # distanceOX = 0
                   # distanceOY = 0

                    leftY = y + distanceOY - eps
                    rigthY = y + h + distanceOY + eps
                    leftY = 0 if leftY < 0 else frame.shape[0] - 1 if leftY > frame.shape[0] - 1 else leftY
                    rigthY = 0 if rigthY < 0 else frame.shape[0] - 1 if rigthY > frame.shape[0] - 1 else rigthY

                    leftX = x + distanceOX - eps
                    rigthX = x + w + distanceOX + eps
                    leftX = 0 if leftX < 0 else frame.shape[1] - 1 if leftX > frame.shape[1] - 1 else leftX
                    rigthX = 0 if rigthX < 0 else frame.shape[1] - 1 if rigthX > frame.shape[1] - 1 else rigthX


                    leftY = int(leftY)
                    leftX = int(leftX)
                    rigthY = int(rigthY)
                    rigthX = int(rigthX)

                    if leftX == frame.shape[1] - 1 or rigthX <= 0 or leftY == frame.shape[0] - 1 or rigthY <= 0:
                        continue

                    print(leftX, rigthX, leftY, rigthY, distanceOX, distanceOY)

                    img = frame[leftY:rigthY, leftX:rigthX].copy()
                    human_boxes = self.yolo.get_human_boxes(img, 640)
                    if not len(human_boxes):
                        continue
                    if len(human_boxes) == 1 and common_area(human_boxes[0], (0,0,w+2*eps, h+2*eps))/(human_boxes[0][2]*human_boxes[0][3]) > 0.92:
                        self.boxes.remove(box)
                        continue

                    closest_box = find_closest((eps, eps, w, h), human_boxes)
                    while not closest_box[2] or not closest_box[3]:
                        human_boxes.remove(closest_box)
                        if not len(human_boxes):
                            break
                        closest_box = find_closest((eps, eps, w, h), human_boxes)
                    if not closest_box[2] or not closest_box[3]:
                        continue

                    (xA, yA, wA, hA) = [int(v) for v in closest_box]
                    if not wA or not hA:
                        continue
                    closest_box = (xA + leftX, yA + leftY, wA, hA)

                    new_tracker = OPENCV_OBJECT_TRACKERS[self.tracker_type]()
                    new_tracker.init(frame, closest_box)
                    box.set_tracker(new_tracker, closest_box, timestamp)
                    track_boxes.append(closest_box)

        for track_box in track_boxes:
            (x, y, w, h) = [int(v) for v in track_box]
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Frame", result_frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            exit(0)
        print("COUNT: ", self.count)
        for info in self.info:
            print("Time: ", info)