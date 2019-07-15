from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import imutils
from imutils.video import VideoStream
import argparse
import pickle as pkl
import copy


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


class Yolo:
    def __init__(self, confidence, nms_thesh, reso, cfgfile, weightsfile, num_classes, classes):
        self.confidence = confidence
        self.nms_thesh = nms_thesh
        self.CUDA = torch.cuda.is_available()
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)
        self.model.net_info["height"] = reso
        self.inp_dim = int(self.model.net_info["height"])
        self.num_classes = num_classes
        self.classes = load_classes('data/coco.names')

        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32
        if self.CUDA:
            self.model.cuda()

        self.model.eval()

    def get_human_boxes(self, frame, inp_dim):
        frame_copy = frame.copy()
        img, orig_im, dim = prep_image(frame_copy, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)
        if self.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        output = self.model(Variable(img), self.CUDA)
        output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf= self.nms_thesh)
        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

        output[:, [1, 3]] *= frame.shape[1]
        output[:, [2, 4]] *= frame.shape[0]

        human_boxes = []
        for detected_object in output:
            if self.classes[int(detected_object[-1])] == "person":
                obj_data = tuple(detected_object[1:5].cpu().numpy())
                box = (obj_data[0], obj_data[1], obj_data[2] - obj_data[0], obj_data[3] - obj_data[1])
                human_boxes.append(box)
        return human_boxes


