# coding: utf-8

from __future__ import division, print_function

import numpy as np
import cv2
import sys
from utils.data_aug import *
import random

PY_VERSION = sys.version_info[0]
iter_cnt = 0


def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed info.
    line format: line_index, img_path, img_width, img_height, [eye_point_1 (3 number)], ...
    return:
        line_idx: int64
        pic_path: string.
        points: shape [N, 2], N is the ground truth count, elements in the second
            dimension are [point_x, point_y]
        labels: shape [N]. class index.
        img_width: int.
        img_height: int
    '''
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    assert len(s) > 8, 'Annotation error! Please check your annotation file. Make sure there is at least one target object in each image.'
    line_idx = int(s[0])
    pic_path = s[1]
    img_width = int(s[2])
    img_height = int(s[3])
    s = s[4:]
    assert len(s) % 3 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
    point_cnt = len(s) // 3
    points = []
    labels = []
    for i in range(point_cnt):
        label, point_x, point_y = int(s[i * 3]), float(s[i * 3 + 1]), float(s[i * 3 + 2])
        points.append([point_x, point_y])
        labels.append(label)
    points = np.asarray(points, np.float32)
    labels = np.asarray(labels, np.int64)
    return line_idx, pic_path, points, labels, img_width, img_height


def process_box(points, labels, img_size, class_num):
    '''
    Generate the y_true label
    params:
        points: [N, 2] shape, float32 dtype. `point_x, point_y`.
        labels: [N] shape, int64 dtype. Here N=2
        class_num: int64 num. Here class_num.Because we only have two states:open eye or close eye.
    '''

    points = points[:, 0:2]

    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 4 + class_num), np.float32)

    # mix up weight default to 1.
    y_true_52[..., -1] = 1.

    y_true = [y_true_52]

    # N
    for i in range(labels.shape[0]):
        feature_map_group = 0  # Here we only use the 52*52 feature map
        # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        ratio = 8
        x = int(np.floor(points[i, 0] / ratio))
        y = int(np.floor(points[i, 1] / ratio))
        c = labels[i]

        y_true[feature_map_group][y, x, :2] = points[i]
        y_true[feature_map_group][y, x, 2] = 1.
        y_true[feature_map_group][y, x, 3 + c] = 1.

    return y_true_52


def parse_data(line, class_num, img_size, mode, letterbox_resize):
    '''
    param:
        line: a line from the training/test txt file
        class_num: totol class nums.Here is two classes, because there are two eyes one person
        img_size: the size of image to be resized to. [width, height] format.
        mode: 'train' or 'val'. When set to 'train', data_augmentation will be applied.
        letterbox_resize: whether to use the letterbox resize, i.e., keep the original aspect ratio in the resized image.
    '''
    if not isinstance(line, list):
        img_idx, pic_path, points, labels, _, _ = parse_line(line)
        img = cv2.imread(pic_path)


        # expand the 2nd dimension, mix up weight default to 1.
        points = np.concatenate((points, np.full(shape=(points.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)
    else:
        # the mix up case
        _, pic_path1, points1, labels1, _, _ = parse_line(line[0])
        img1 = cv2.imread(pic_path1)
        img_idx, pic_path2, points2, labels2, _, _ = parse_line(line[1])
        img2 = cv2.imread(pic_path2)

        img, points = mix_up(img1, img2, points1, points2)
        labels = np.concatenate((labels1, labels2))

    if mode == 'train':
        # random color jittering
        # NOTE: applying color distort may lead to bad performance sometimes
        img = random_color_distort(img)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, points = random_expand(img, points, 4)

        # # random cropping
        # h, w, _ = img.shape
        # boxes, crop = random_crop_with_constraints(boxes, (w, h))
        # x0, y0, w, h = crop
        # img = img[y0: y0+h, x0: x0+w]

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img, points = resize_with_point(img, points, img_size[0], img_size[1], interp=interp, letterbox=letterbox_resize)

        # random horizontal flip
        h, w, _ = img.shape
        img, points = random_flip(img, points, px=0.5)
    else:
        img, points = resize_with_point(img, points, img_size[0], img_size[1], interp=1, letterbox=letterbox_resize)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # the input of yolo_v3 should be in range 0~1
    img = img / 255.


    y_true_52 = process_box(points, labels, img_size, class_num)

    return img_idx, img, y_true_52


def get_batch_data(batch_line, class_num, img_size, mode, multi_scale=False, mix_up=False, letterbox_resize=True, interval=10):
    '''
    generate a batch of imgs and labels
    param:
        batch_line: a batch of lines from train/val.txt files
        class_num: num of total classes.
        img_size: the image size to be resized to. format: [width, height].
        mode: 'train' or 'val'. if set to 'train', data augmentation will be applied.
        multi_scale: whether to use multi_scale training, img_size varies from [320, 320] to [640, 640] by default. Note that it will take effect only when mode is set to 'train'.
        letterbox_resize: whether to use the letterbox resize, i.e., keep the original aspect ratio in the resized image.
        interval: change the scale of image every interval batches. Note that it's indeterministic because of the multi threading.
    '''
    global iter_cnt
    # multi_scale training
    if multi_scale and mode == 'train':
        random.seed(iter_cnt // interval)
        random_img_size = [[x * 32, x * 32] for x in range(10, 20)]
        img_size = random.sample(random_img_size, 1)[0]
    iter_cnt += 1

    img_idx_batch, img_batch, y_true_52_batch = [], [], []

    # mix up strategy
    if mix_up and mode == 'train':
        mix_lines = []
        batch_line = batch_line.tolist()
        for idx, line in enumerate(batch_line):
            if np.random.uniform(0, 1) < 0.5:
                mix_lines.append([line, random.sample(batch_line[:idx] + batch_line[idx+1:], 1)[0]])
            else:
                mix_lines.append(line)
        batch_line = mix_lines

    for line in batch_line:
        img_idx, img, y_true_52 = parse_data(line, class_num, img_size, mode, letterbox_resize)

        img_idx_batch.append(img_idx)
        img_batch.append(img)

        y_true_52_batch.append(y_true_52)

    img_idx_batch, img_batch, y_true_52_batch = np.asarray(img_idx_batch, np.int64), np.asarray(img_batch), np.asarray(y_true_52_batch)

    return img_idx_batch, img_batch, y_true_52_batch
