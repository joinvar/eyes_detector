# coding: utf-8

from __future__ import division, print_function

import numpy as np
import cv2
from collections import Counter

from utils.nms_utils import cpu_nms, score_filter
from utils.data_utils import parse_line


def calc_iou(pred_boxes, true_boxes):
    '''
    Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
    shape_info: pred_boxes: [N, 4]
                true_boxes: [V, 4]
    return: IoU matrix: shape: [N, V]
    '''

    # [N, 1, 4]
    pred_boxes = np.expand_dims(pred_boxes, -2)
    # [1, V, 4]
    true_boxes = np.expand_dims(true_boxes, 0)

    # [N, 1, 2] & [1, V, 2] ==> [N, V, 2]
    intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
    intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [N, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [N, 1, 2]
    pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    # shape: [N, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # [1, V, 2]
    true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
    # [1, V]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]

    # shape: [N, V]
    iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)

    return iou


def evaluate_on_cpu(y_pred, y_true, num_classes, calc_now=True, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    '''
    Given y_pred and y_true of a batch of data, get the recall and precision of the current batch.
    '''

    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps
            # shape: [13, 13, 3, 80]
            true_probs_temp = y_true[j][i][..., 5:-1]
            # shape: [13, 13, 3, 4] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:4]

            # [13, 13, 3]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 3] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]
        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes

        # [1, xxx, 4]
        pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[1][i:i + 1]
        pred_probs = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels = cpu_nms(pred_boxes, pred_confs * pred_probs, num_classes,
                                                      max_boxes=max_boxes, score_thresh=score_thresh, iou_thresh=iou_thresh)

        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        # calc iou
        # [N, V]
        iou_matrix = calc_iou(pred_boxes, true_boxes)
        # [N]
        max_iou_idx = np.argmax(iou_matrix, axis=-1)

        correct_idx = []
        correct_conf = []
        for k in range(max_iou_idx.shape[0]):
            pred_labels_dict[pred_labels_list[k]] += 1
            match_idx = max_iou_idx[k]  # V level
            if iou_matrix[k, match_idx] > iou_thresh and true_labels_list[match_idx] == pred_labels_list[k]:
                if match_idx not in correct_idx:
                    correct_idx.append(match_idx)
                    correct_conf.append(pred_confs[k])
                else:
                    same_idx = correct_idx.index(match_idx)
                    if pred_confs[k] > correct_conf[same_idx]:
                        correct_idx.pop(same_idx)
                        correct_conf.pop(same_idx)
                        correct_idx.append(match_idx)
                        correct_conf.append(pred_confs[k])

        for t in correct_idx:
            true_positive_dict[true_labels_list[t]] += 1

    if calc_now:
        # avoid divided by 0
        recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
        precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict


def func_evaluate(sess, score_filter_op, pred_boxes_flag, pred_scores_flag, y_pred, y_true, num_classes, iou_thresh=0.5, calc_now=True):
    num_images = y_true[0].shape[0]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(num_images):
        true_labels_list, true_boxes_list = [], []
        for j in range(1):
            # shape: [52, 52, 2]
            true_probs_temp = y_true[j][i][..., 3:-1]
            true_probs = np.reshape(true_probs_temp, (-1, 2))
            # shape: [52, 52, 2] (x_center, y_center, w, h)
            true_boxes_temp = y_true[j][i][..., 0:2]

            # [52, 52]
            object_mask = true_probs_temp.sum(axis=-1) > 0

            # [V, 2] V: Ground truth number of the current image
            true_probs_temp = true_probs_temp[object_mask]
            # [V, 4]
            true_boxes_temp = true_boxes_temp[object_mask]

            # [V], labels
            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            # [V, 4] (x_center, y_center, w, h)
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items():
                true_labels_dict[cls] += count

        # [V, 4] (xmin, ymin, xmax, ymax)
        true_boxes = np.array(true_boxes_list)
        box_centers = true_boxes[:, 0:2]

        # [1, xxx, 4] from 136th line of file model.py
        pred_boxes = y_pred[0][i:i + 1]
        # print('pred_boxes  1 shape :', pred_boxes.shape)
        # [1, xxx, 1]
        pred_confs = y_pred[1][i:i + 1]
        # [1, xxx, class_num]
        pred_probs = y_pred[2][i:i + 1]

        # pred_boxes: [N, 4]
        # pred_confs: [N]
        # pred_labels: [N]
        # N: Detected box number of the current image
        pred_boxes, pred_confs, pred_labels, pre_mask = sess.run(score_filter_op,
                                                       feed_dict={pred_boxes_flag: pred_boxes,
                                                                  pred_scores_flag: pred_confs * pred_probs})
        # len: N
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()
        if pred_labels_list == []:
            continue

        for k in range(len(pred_labels_list)):
            pred_labels_dict[pred_labels_list[k]] += 1

        prob_matrix = true_probs * pre_mask

        # [13, 13]
        prob_mask = prob_matrix.sum(axis=-1) > 0
        # [V, 2] V: Ground truth number of the current image
        prob_matrix = prob_matrix[prob_mask]

        # print('*' * 30)
        # print('prob_matrix')
        # print(prob_matrix)
        # print('*' * 30)
        for k in range(prob_matrix.shape[0]):
            for t in range(2):
                true_positive_dict[t] += 1 if prob_matrix[k][t] == 1 else 0


    if calc_now:
        # avoid divided by 0
        recall = true_positive_dict[0] / (sum(true_labels_dict.values()) + 1e-6)
        precision = true_positive_dict[0] / (sum(pred_labels_dict.values()) + 1e-6)

        return recall, precision
    else:
        return true_positive_dict, true_labels_dict, pred_labels_dict


def get_preds_gpu(sess, score_filter_op, pred_boxes_flag, pred_scores_flag, y_true, y_pred):
    '''
    Given the y_pred of an input image, get the predicted bbox and label info.
    return:
        pred_content: 2d list.
    '''
    # image_id = image_ids[0]

    # keep the first dimension 1
    pred_boxes = y_pred[0][0:1]
    pred_confs = y_pred[1][0:1]
    pred_probs = y_pred[2][0:1]

    boxes, scores, labels, pre_mask = sess.run(score_filter_op,
                                     feed_dict={pred_boxes_flag: pred_boxes,
                                                pred_scores_flag: pred_confs * pred_probs})

    pred_content = []
    pred_label = []
    for i in range(len(labels)):
        x_min, y_min = boxes[i]
        score = scores[i]
        label = labels[i]
        pred_label.append([score, label])
    pred_content.append([pre_mask, y_true, pred_label])
    return pred_content


gt_dict = {}  # key: img_id, value: gt object list
def parse_gt_rec(gt_filename, target_img_size, letterbox_resize=True):
    '''
    parse and re-organize the gt info.
    return:
        gt_dict: dict. Each key is a img_id, the value is the gt bboxes in the corresponding img.
    '''

    global gt_dict

    if not gt_dict:
        new_width, new_height = target_img_size
        with open(gt_filename, 'r') as f:
            for line in f:
                img_id, pic_path, boxes, labels, ori_width, ori_height = parse_line(line)

                objects = []
                for i in range(len(labels)):
                    x_min, y_min = boxes[i]
                    label = labels[i]

                    if letterbox_resize:
                        resize_ratio = min(new_width / ori_width, new_height / ori_height)

                        resize_w = int(resize_ratio * ori_width)
                        resize_h = int(resize_ratio * ori_height)

                        dw = int((new_width - resize_w) / 2)
                        dh = int((new_height - resize_h) / 2)

                        objects.append([x_min * resize_ratio + dw,
                                        y_min * resize_ratio + dh,
                                        label])
                    else:
                        objects.append([x_min * new_width / ori_width,
                                        y_min * new_height / ori_height,
                                        label])
                gt_dict[img_id] = objects
    return gt_dict


# The following two functions are modified from FAIR's Detectron repo to calculate mAP:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/voc_eval.py
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(val_preds, classidx,use_07_metric=False):
    '''
    Top level function that does the PASCAL VOC evaluation.

    gt_dict
        gt_dict[img_id] = objects
        objects
            [[x_min, y_min, label], [x_min, y_min, label]]

    val_preds
        [[image_id, x_min, y_min, score, label],[image_id, x_min, y_min, score, label]]
    '''
    # 1.obtain gt: extract all gt objects for this class
    y_trues = [val_pred[1] for val_pred in val_preds]
    y_mask = [val_pred[0] for val_pred in val_preds]
    y_true_mask = [y_trues, y_mask]
    num_images = 0
    nd = 0
    class_recs = {}
    npos = 0
    tp = []
    fp = []
    num_img = 0
    for y_true_i, y_mask_i in zip(y_trues, y_mask):
        num_img = y_true_i[0].shape[0]  # y_true_i[0] this 0 is y_true_52
        num_images += num_img
        for img_id in range(num_img):
            image = y_true_i[0][img_id][..., 3:-1]
            image1 = image.reshape((-1, 2))
            image = np.sum(image1, axis=0)
            npos += image[classidx]

            pred_mask = image1 * y_mask_i
            pred_mask = np.sum(pred_mask, axis=0)
            tp_tmp = pred_mask[classidx]
            tp.append(tp_tmp)

            y_mask = np.sum(y_mask_i, axis=0)
            fp_tmp = y_mask[classidx] - tp_tmp
            fp.append(fp_tmp)


    tp = np.array(tp)
    fp = np.array(fp)
    nd = num_images
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (float(npos) + np.finfo(np.float64).eps)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    tfp = tp + fp
    # return rec, prec, ap
    return npos, tfp[-1], tp[-1] / (float(npos) + np.finfo(np.float64).eps), tp[-1] / (float(tfp[-1]) + np.finfo(np.float64).eps), ap
