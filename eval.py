# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
from tqdm import trange

from utils.data_utils import get_batch_data
from utils.misc_utils import parse_anchors, read_class_names, AverageMeter
from utils.eval_utils import evaluate_on_cpu, func_evaluate, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import score_filter

from model import yolov3

#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="Eyes_detector eval procedure.")
# some paths
parser.add_argument("--eval_file", type=str, default="./data/my_data/val.txt",
                    help="The path of the validation or test txt file.")

parser.add_argument("--restore_path", type=str, default="./checkpoint/best_model_Epoch_33_step_2346_mAP_0.3094_loss_7.2790_lr_0.0001",
                    help="The path of the weights to restore.")

parser.add_argument("--class_name_path", type=str, default="./data/my_data/data.names",
                    help="The path of the class names.")

# some numbers
parser.add_argument("--img_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image to `img_size`, size format: [width, height]")

parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize, i.e., keep the original image aspect ratio.")

parser.add_argument("--num_threads", type=int, default=10,
                    help="Number of threads for image processing used in tf.data pipeline.")

parser.add_argument("--prefetech_buffer", type=int, default=5,
                    help="Prefetech_buffer used in tf.data pipeline.")

parser.add_argument("--score_threshold", type=float, default=0.01,
                    help="Threshold of the probability of the classes in nms operation.")


args = parser.parse_args()

# args params
args.classes = read_class_names(args.class_name_path)
args.class_num = len(args.classes)
args.img_cnt = len(open(args.eval_file, 'r').readlines())

# setting placeholders
is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
score_filter_op = score_filter(pred_boxes_flag, pred_scores_flag, args.class_num, args.score_threshold)

##################
# tf.data pipeline
##################
val_dataset = tf.data.TextLineDataset(args.eval_file)
val_dataset = val_dataset.batch(1)
val_dataset = val_dataset.map(
    lambda x: tf.py_func(get_batch_data, [x, args.class_num, args.img_size, 'val', False, False, args.letterbox_resize],
                         [tf.int64, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
val_dataset.prefetch(args.prefetech_buffer)
iterator = val_dataset.make_one_shot_iterator()

image_ids, image, y_true_52 = iterator.get_next()
image_ids.set_shape([None])
y_true = [y_true_52]
image.set_shape([None, args.img_size[1], args.img_size[0], 3])
for y in y_true:
    y.set_shape([None, None, None, None])

##################
# Model definition
##################
yolo_model = yolov3(args.class_num)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
loss = yolo_model.compute_loss(pred_feature_maps, y_true)
y_pred = yolo_model.predict(pred_feature_maps)

saver_to_restore = tf.train.Saver()

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    saver_to_restore.restore(sess, args.restore_path)

    print('\n----------- start to eval -----------\n')

    val_loss_total, val_loss_xy,  val_loss_conf, val_loss_class = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    val_preds = []

    for j in trange(args.img_cnt):
        __y_true, __y_pred, __loss = sess.run([y_true, y_pred, loss], feed_dict={is_training: False})
        pred_content = get_preds_gpu(sess, score_filter_op, pred_boxes_flag, pred_scores_flag, __y_true, __y_pred)

        val_preds.extend(pred_content)
        val_loss_total.update(__loss[0])
        val_loss_xy.update(__loss[1])
        val_loss_conf.update(__loss[2])
        val_loss_class.update(__loss[3])

    rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
    for ii in range(args.class_num):
        npos, nd, rec, prec, ap = voc_eval(val_preds, ii)
        rec_total.update(rec, npos)
        prec_total.update(prec, nd)
        ap_total.update(ap, 1)
        print('Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}'.format(ii, rec, prec, ap))

    mAP = ap_total.average
    print('final mAP: {:.4f}'.format(mAP))
    print("recall: {:.3f}, precision: {:.3f}".format(rec_total.average, prec_total.average))
    print("total_loss: {:.3f}, loss_xy: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
        val_loss_total.average, val_loss_xy.average, val_loss_conf.average, val_loss_class.average
    ))
