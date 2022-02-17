"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np

from preprocessing import crop_image_from_craft
from text_detection_utils import craft_utils
from text_detection_utils import imgproc
from text_detection_utils import file_utils
from text_detection_utils.refinenet import RefineNet
import json
import zipfile
from text_detection_utils.craft import CRAFT
from collections import OrderedDict


## Cropping Dependencies
import os
import math
import cv2


class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def get_tuple(self):
        return self.x, self.y

    def get_rotated(self, a, o):
        new_x = ((self.x - o.x) * math.cos(-math.radians(a))) - \
                ((self.y - o.y) * math.sin(-math.radians(a))) + o.x
        new_y = ((self.x - o.x) * math.sin(-math.radians(a))) + \
                ((self.y - o.y) * math.cos(-math.radians(a))) + o.y
        return Point(new_x, new_y)

    def __str__(self):
        return f"({self.x}, {self.y})"


class BBox:
    def __init__(self, data):
        coordinates = data.split(',')
        self.tl = Point(coordinates[0], coordinates[1])
        self.tr = Point(coordinates[2], coordinates[3])
        self.bl = Point(coordinates[4], coordinates[5])
        self.br = Point(coordinates[6], coordinates[7])

    @classmethod
    def of(cls, tl, tr, bl, br):
        data = [str(tl.x), str(tl.y), str(tr.x), str(tr.y), str(bl.x), str(bl.y), str(br.x), str(br.y)]
        return BBox(','.join(data))

    def get_center_y(self):
        return (self.tl.y + self.br.y + self.tr.y + self.bl.y) / 4

    def get_rotate_angle(self):
        return math.degrees(math.atan((self.tr.y - self.tl.y) / (self.tr.x - self.tl.x)))

    def get_rotate_box(self, o):
        a = self.get_rotate_angle()
        return BBox.of(self.tl.get_rotated(a, o), self.tr.get_rotated(a, o),
                       self.bl.get_rotated(a, o), self.br.get_rotated(a, o))

    def __str__(self):
        return f"[tl:{self.tl}, tr:{self.tr}, bl:{self.bl}, br:{self.br}]"


def merge_boxes(box_to_merge):
    tl = box_to_merge[0].tl
    tr = box_to_merge[0].tr
    bl = box_to_merge[0].bl
    br = box_to_merge[0].br
    for box in box_to_merge:
        if box.bl.x < bl.x:
            bl = box.bl
        if box.tl.x < tl.x:
            tl = box.tl
        if box.tr.x > tr.x:
            tr = box.tr
        if box.br.x > br.x:
            br = box.br
    return BBox.of(tl, tr, bl, br)


def merge_nearest_bbox(boxes):
    dist = dict()
    points = dict()
    x = []
    for idx, bbox in enumerate(boxes):
        x.append(bbox.get_center_y())
        for i in range(0, idx):
            points[(i, idx)] = abs(x[i] - x[idx])
            distance = abs(x[i] - x[idx])
            if dist.get(distance) is None:
                dist[distance] = []
            dist[distance].append(i)
            dist[distance].append(idx)

    length = len(x)
    box_to_merge = set()
    while length > 3:
        mini = min(dist.keys())
        box_to_merge.update(list(dist.pop(mini)))
        length = len(x) - len(box_to_merge) + 1

    box_to_merge = [boxes[i] for i in box_to_merge]
    [boxes.remove(i) for i in box_to_merge]
    box = merge_boxes(box_to_merge)
    boxes.append(box)

    return boxes


def get_points_from_file(datas):
    boxes = []
    for data in datas:
        if not data.isspace():
            boxes.append(BBox(data.replace("\n", "")))

    if len(boxes) > 3:
        boxes = merge_nearest_bbox(boxes)
    return boxes


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


text_threshold = 0.7
low_text = 0.4
link_threshold = 0.4
canvas_size = 1280
mag_ratio = 1.5
poly = False
show_time = False
refine = False
test_folder = "/data/"


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    # if cuda:
    #     x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def text_detection_module(image_path, use_cuda = False):

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    image = imgproc.loadImage(image_path)

    # load net
    net = CRAFT()
    weight_path = "craft_mlt_25k.pth"
    if use_cuda:
        net.load_state_dict(copyStateDict(torch.load(weight_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(weight_path, map_location='cpu')))

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()

    refine_net = RefineNet()
    # print('Loading weights of refiner from checkpoint (craft_refiner_CTW1500.pth)')
    weight_path = "craft_refiner_CTW1500.pth"
    if use_cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(weight_path)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(weight_path, map_location='cpu')))

    refine_net.eval()
    poly = True

    if image.shape[0] == 2:
        image = image[0]
    if len(image.shape) == 2 :
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        image = image[:,:,:3]

    bboxes, polys, score_text = test_net(net, image, 0.7, 0.4, 0.4, False,
                                         False, refine_net)

    file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

    img_list_to_return = crop_image_from_craft(image, bboxes)


    return img_list_to_return, bboxes

    # # save score text
    # filename, file_ext = str(index), ".jpg"
    # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    # cv2.imwrite(mask_file, score_text)
    #
    # file_utils.saveResult('{}{}'.format(filename, file_ext), image[:, :, ::-1], polys, dirname=result_folder)



if __name__ == '__main__':
    cuda = True

    # load net
    net = CRAFT()     # initialize

    weight_path = "craft_mlt_25k.pth"
    print('Loading weights from checkpoint (' + weight_path + ')')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(weight_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(weight_path, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    # if refine:
    #     from text_detection_utils.refinenet import RefineNet
    #     refine_net = RefineNet()
    #     print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
    #     if cuda:
    #         refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
    #         refine_net = refine_net.cuda()
    #         refine_net = torch.nn.DataParallel(refine_net)
    #     else:
    #         refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))
    #
    #     refine_net.eval()
    #     poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))
