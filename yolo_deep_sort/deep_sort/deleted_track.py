import numpy as np
from .track import Track

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

class DeletedTrack:
    def __init__(self, track):
        self.track = track
        self.age = 0

    def search(self, track_boxes, tracks):
        self.age += 1
        max_match = 0
        idx_max = -1
        for idx, box in enumerate(track_boxes):
            if box is None or tracks[idx].direction != self.track.direction:
                continue
            match = common_area(self.track.to_tlwh(), box)
            if match > max_match:
                max_match = match
                idx_max = idx

        if max_match < 0.1:
            return -1
        return idx_max

