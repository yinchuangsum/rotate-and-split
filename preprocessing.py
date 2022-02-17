from random import seed, random

import cv2
import numpy as np

from angle_detection_correction import four_point_transform_custom


def add_padding(cv2_array, pad = 50):
    return cv2.copyMakeBorder(cv2_array, pad, pad, pad, pad, cv2.BORDER_CONSTANT, (1, 1, 1))


def scale_up_image(cv2_array, scale_percentage):
    scale_percent = scale_percentage + 100
    width = int(cv2_array.shape[1] * scale_percent / 100)
    height = int(cv2_array.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(cv2_array, dim, interpolation=cv2.INTER_AREA)


def crop_image_from_craft(cv2_array, boxes):
    list_to_return = []
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)

        pts = poly

        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        igg = four_point_transform_custom(cv2_array, pts)
        a = random()
        # cv2.imwrite("test_{}.jpg".format(a), igg)
        croped = cv2_array[y:y + h, x:x + w].copy()
        # cv2.imwrite("test_crop_{}.jpg".format(a), croped)

        ## (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)

        ## (4) add the white background
        bg = np.ones_like(croped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst2 = bg + dst

        list_to_return.append(dst2)
    return list_to_return

        # cv2.imwrite("croped.png", croped)
        # cv2.imwrite("mask.png", mask)
        # cv2.imwrite("dst.png", dst)
        # cv2.imwrite("dst2.png", dst2)

        # poly = poly.reshape(-1, 2)
        # cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)