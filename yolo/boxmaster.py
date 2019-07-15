import cv2
from box import Box
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


class BoxMaster:
    def __init__(self, accuracy, tracker_type, max_tracked_people, yolo):
        self.boxes = []
        self.accuracy = accuracy
        self.tracker_type = tracker_type
        self.max_tracked_people = max_tracked_people
        self.people_count = 0
        self.yolo = yolo

    def yolo(self, frame, yolo_boxes, timestamp):

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
                #self.boxes.remove(box)
                new_tracker = OPENCV_OBJECT_TRACKERS[self.tracker_type]()
                new_tracker.init(frame, max_human)
                box.set_tracker(new_tracker)
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
                self.boxes.append(Box(tracker, box, timestamp, self.yolo))
                self.people_count += 1

        print("People count: ", self.people_count)

        return

    def update(self, frame, timestamp):
        frame_copy = frame.copy()
        for box in list(self.boxes):
            frame_aux = box.update(frame_copy, timestamp)
            if frame_aux:
                frame = frame_aux

            if common_area((0, 0, frame.shape[1], frame.shape[0]), box.position) / (box.area()) < 0.6:
                self.boxes.remove(box)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            exit(0)
