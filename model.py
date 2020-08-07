# coding=utf-8
# for better understanding about yolov3 architecture, refer to this website (in Chinese):
# https://blog.csdn.net/leviopku/article/details/82660381

from __future__ import division, print_function

import tensorflow as tf
slim = tf.contrib.slim

from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer

class yolov3(object):

    def __init__(self, class_num, use_label_smooth=False, use_focal_loss=False, batch_norm_decay=0.999, weight_decay=5e-4, use_static_shape=True):
        self.class_num = class_num
        self.batch_norm_decay = batch_norm_decay
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.weight_decay = weight_decay
        # inference speed optimization
        # if `use_static_shape` is True, use tensor.get_shape(), otherwise use tf.shape(tensor)
        # static_shape is slightly faster
        self.use_static_shape = use_static_shape

    def forward(self, inputs, is_training=False, reuse=False):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d], 
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
                                # weights_regularizer=slim.l2_regularizer(self.weight_decay)
            ):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, route_2.get_shape().as_list() if self.use_static_shape else tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, route_1.get_shape().as_list() if self.use_static_shape else tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, (3 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')
            return [feature_map_3]

    def reorg_layer(self, feature_map):
        '''
        feature_map: a feature_map from [feature_map_52] returned
            from `forward` function
        '''
        # NOTE: size in [h, w] format! don't get messed up!   -----> note: 31st line in this file
        grid_size = feature_map.get_shape().as_list()[1:3] if self.use_static_shape else tf.shape(feature_map)[1:3]  # [52, 52]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)


        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image for example and the feature_map's shape is 52*52:
        # points: [N, 52, 52, 2] last_dimension: [point_x, point_y]
        # conf_logits: [N, 52, 52, 1]
        # prob_logits: [N, 52, 52, class_num]
        points, conf_logits, prob_logits = tf.split(feature_map, [2, 1, self.class_num], axis=-1)
        points = tf.nn.sigmoid(points)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 2]), tf.float32)

        # get the absolute box coordinates on the feature_map 
        points = points + x_y_offset
        # rescale to the original image scale
        points = points * ratio[::-1]

        # shape: [N, 52, 52, 2]
        # last dimension: (point_x, point_y)
        points = tf.concat([points], axis=-1)

        # shape:
        # x_y_offset: [52, 52, 2]
        # boxes: [N, 52, 52, 2], rescaled to the original image scale
        # conf_logits: [N, 52, 52, 1]
        # prob_logits: [N, 52, 52, class_num]
        return x_y_offset, points, conf_logits, prob_logits


    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        # we only use 52*52 feature_map
        feature_map_3 = feature_maps[0]

        result = self.reorg_layer(feature_map_3)
        def _reshape(result):
            x_y_offset, points, conf_logits, prob_logits = result
            grid_size = x_y_offset.get_shape().as_list()[:2] if self.use_static_shape else tf.shape(x_y_offset)[:2]
            points = tf.reshape(points, [-1, grid_size[0] * grid_size[1], 2]) # yooooooooooooolo is 4, here is 2
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1], 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1], self.class_num])
            # shape: (take 416*416 input image for example and feature_map_1's shape is 52*52)
            # points: [N, 52*52, 2]
            # conf_logits: [N, 52*52, 1]
            # prob_logits: [N, 52*52, class_num]
            return points, conf_logits, prob_logits

        points_list, confs_list, probs_list = [], [], []
        # for result in reorg_results:
        points, conf_logits, prob_logits = _reshape(result)
        confs = tf.sigmoid(conf_logits)
        probs = tf.sigmoid(prob_logits)
        points_list.append(points)
        confs_list.append(confs)
        probs_list.append(probs)
        
        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, 52*52, 2]
        points = tf.concat(points_list, axis=1)
        # shape: [N, 52*52, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, 52*52, class_num]
        probs = tf.concat(probs_list, axis=1)

        return points, confs, probs
    
    def loss_layer(self, feature_map_i, y_true):
        '''
        calc loss function from a certain scale
        input:
            feature_map_i: shape: [N, 52, 52, (3 + num_class)] etc.
            y_true: y_ture from a certain scale. shape: [N, 52, 52,  3 + num_class + 1] etc.
        '''
        
        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pre_points, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i)

        ###########
        # get mask
        ###########

        # shape: take 416x416 input image for example and feature_map's shape is 52*52:
        # [N, 52, 52 1]
        object_mask = y_true[..., 2:3]


        # shape: [N, 52, 52 2]
        pred_xy = pre_points[..., 0:2]
        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 52, 52, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_xy / ratio[::-1] - x_y_offset


        ############
        # loss_part
        ############
        # mix_up weight
        # [N, 52, 52, 1]
        mix_w = y_true[..., -1:]
        # shape: [N, 52, 52, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * mix_w) / N

        # shape: [N, 52, 52, 1]
        conf_loss_pos = tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        # TODO: may need to balance the pos-neg by multiplying some weights
        conf_loss = conf_loss_pos
        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            # TODO: alpha should be a mask array if needed
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # shape: [N, 52, 52, 1]
        # whether to use label smooth
        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 3:-1] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 3:-1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, conf_loss, class_loss
    

    def box_iou(self, pred_boxes, valid_true_boxes):
        '''
        param:
            pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
            valid_true: [V, 4]
        '''

        # [13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # shape: [13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # [V, 2]
        true_box_xy = valid_true_boxes[:, 0:2]
        true_box_wh = valid_true_boxes[:, 2:4]

        # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = tf.expand_dims(true_box_area, axis=0)

        # [13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou

    
    def compute_loss(self, y_pred, y_true):
        '''
        param:
            y_pred: [feature_map_52]
            y_true: input y_true by the tf.data pipeline
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.


        # calc loss in 3 scales, but here we only have 52*52 scale
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i])
            loss_xy += result[0]
            loss_conf += result[1]
            loss_class += result[2]
        total_loss = loss_xy + loss_conf + loss_class
        return [total_loss,  loss_xy, loss_conf, loss_class]
