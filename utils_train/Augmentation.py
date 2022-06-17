import tensorflow as tf
import random
from utils_train.temp_f import *

def randomResize(image, boxes, targetH, targetW, p = 1.0):
    def _keep_aspect_ratio(img, boxes, h, w):
        image_shape = tf.cast(tf.shape(img), tf.float32)
        image_height, image_width = image_shape[0], image_shape[1]

        img = tf.image.resize_with_pad(img, h, w)

        h, w = tf.cast(h, dtype=tf.float32), tf.cast(w, dtype=tf.float32)
        resize_coef = tf.math.minimum(h / image_height, w / image_width)
        resized_height, resized_width = image_height * resize_coef, image_width * resize_coef
        pad_y, pad_x = (h - resized_height) / 2, (w - resized_width) / 2
        boxes = boxes * tf.stack([resized_height, resized_width, resized_height, resized_width]) + \
                        tf.stack([pad_y, pad_x, pad_y, pad_x,]
        )
        boxes /= tf.stack([h, w, h, w])
        return img, boxes

    def _dont_keep_aspect_ration(img, boxes, h, w):
        img = tf.image.resize(img, (h, w))
        return img, boxes

    if tf.random.uniform([], minval=0, maxval=1) < p:
        keep_aspect_ratio = True
    else:
        keep_aspect_ratio = False
    image, boxes = tf.cond(keep_aspect_ratio,
                            lambda: _keep_aspect_ratio(image, boxes, targetH, targetW),
                            lambda: _dont_keep_aspect_ration(image, boxes, targetH, targetW))
    return image, boxes


def flipHorizontal(image, boxes, p = 1.0):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return image, boxes

    image = tf.image.flip_left_right(image)
    boxes = tf.stack([boxes[:, 0], 1.0 - boxes[:, 3], boxes[:, 2], 1.0 - boxes[:, 1]], axis=-1)

    return image, boxes


def flipVertical(image, boxes, p = 1.0):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return image, boxes
    image = tf.image.flip_up_down(image)
    boxes = tf.stack([1.0 - boxes[:, 2], boxes[:, 1], 1.0 - boxes[:, 0], boxes[:, 3]], axis=-1)

    return image, boxes

def randomCrop(image, bbox, class_id, p = 1.0):
    ###
    # This crop code from TFOD API
    #
    ###

    if tf.random.uniform([], minval=0, maxval=1) < p:
        return image, bbox, class_id

    def _prune_completely_outside_window(bbox, window):
        y_min, x_min, y_max, x_max = tf.split(value=bbox, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        coordinate_violations = tf.concat([
            tf.greater_equal(y_min, win_y_max), tf.greater_equal(x_min, win_x_max),
            tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
        ], 1)
        valid_indices = tf.reshape(
            tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
        return tf.gather(bbox, valid_indices), valid_indices

    def _prune_outside_window(bbox, window):
        y_min, x_min, y_max, x_max = tf.split(value=bbox, num_or_size_splits=4, axis=1)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        coordinate_violations = tf.concat([
            tf.less(y_min, win_y_min), tf.less(x_min, win_x_min),
            tf.greater(y_max, win_y_max), tf.greater(x_max, win_x_max)
        ], 1)
        valid_indices = tf.reshape(
            tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
        return tf.gather(bbox, valid_indices), valid_indices

    def _prune_non_overlapping_boxes(boxlist1, boxlist2, min_overlap=0.0):
        ioa_ = ioa(boxlist2, boxlist1)  # [M, N] tensor
        ioa_ = tf.reduce_max(ioa_, axis = [0])  # [N] tensor
        keep_bool = tf.greater_equal(ioa_, tf.constant(min_overlap))
        keep_inds = tf.squeeze(tf.where(keep_bool), axis=[1])
        new_boxlist1 = tf.gather(boxlist1, keep_inds)
        return new_boxlist1, keep_inds

    def _change_coordinate_frame(boxlist, window):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        boxlist_new = scale(boxlist - [window[0], window[1], window[0], window[1]],
                            1.0 / win_height, 
                            1.0 / win_width)
        return boxlist_new
    image_shape = tf.shape(image)

    boxes_expanded = tf.expand_dims(bbox, 1) # boxes are [N, 4]. Lets first make them [N, 1, 4].
    
    im_box_begin, im_box_size, im_box = tf.image.sample_distorted_bounding_box(image_shape,
                                                            bounding_boxes=boxes_expanded,
                                                            min_object_covered=random.choice([0.2, 0.3, 0.5, 0.7, 0.9]), #[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
                                                            aspect_ratio_range=[0.5, 2.0], #rand
                                                            area_range=[0.1, 1], #rand
                                                            max_attempts=100,
                                                            use_image_if_no_bounding_boxes=True)


    new_image = tf.slice(image, im_box_begin, im_box_size)

    im_box_rank2 = tf.squeeze(im_box, axis=[0]) #[1,4]
    im_box_rank1 = tf.squeeze(im_box) #[4]

    boxlist, inside_window_ids = _prune_completely_outside_window(bbox, im_box_rank1)
    overlapping_boxlist, keep_ids = _prune_non_overlapping_boxes(boxlist, im_box_rank2, 0.3) #0.3 = overlap_thresh

    new_bbox = _change_coordinate_frame(overlapping_boxlist, im_box_rank1)
    new_bbox = tf.clip_by_value(new_bbox, clip_value_min=0.0, clip_value_max=1.0)
    
    kpt_vis_of_boxes_inside_window = tf.gather(class_id, inside_window_ids)
    kpt_vis_of_boxes_completely_inside_window = tf.gather(kpt_vis_of_boxes_inside_window, keep_ids)

    return new_image, new_bbox, kpt_vis_of_boxes_completely_inside_window

def colorJitter(image, p = 1.0):
    if tf.random.uniform([], minval=0, maxval=1) > p:
        return image

    if tf.random.uniform([], minval=0, maxval=1) < p:
        image = tf.image.random_brightness(image, 0.3)
    if tf.random.uniform([], minval=0, maxval=1) < p:
        image = tf.image.random_contrast(image, 0.2, 0.4)
    if tf.random.uniform([], minval=0, maxval=1) < p:
        image = tf.image.random_hue(image, 0.5)
    if tf.random.uniform([], minval=0, maxval=1) < p:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1,1,3])
    return image


def mixUp(images_one, images_two, bboxes_one, bboxes_two, classes_one, classes_two):
    def _sample_beta_distribution(size, concentration_0=0.5, concentration_1=0.5):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    images = images_one * 0.5 + images_two * (1 - 0.5)
    return images, tf.concat([bboxes_one, bboxes_two], 1), tf.concat([classes_one, classes_two], 1)


def randomExpand(image, bbox, expandMax=1.5, p = 1.0):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return image, bbox
        
    original_w = tf.cast(tf.shape(image)[1], tf.float32)
    original_h = tf.cast(tf.shape(image)[0], tf.float32)

    expandYU = tf.random.uniform(shape = [], minval=0.0, maxval=expandMax, dtype=tf.float32)
    expandYD = tf.random.uniform(shape = [], minval=0.0, maxval=expandMax, dtype=tf.float32)

    expandXL = tf.random.uniform(shape = [], minval=0.0, maxval=expandMax, dtype=tf.float32)
    expandXR = tf.random.uniform(shape = [], minval=0.0, maxval=expandMax, dtype=tf.float32)

    HUPad = tf.zeros([tf.cast(expandYU*original_h, tf.int32), tf.cast(original_w, tf.int32), 3], image.dtype)
    HDPad = tf.zeros([tf.cast(expandYD*original_h, tf.int32), tf.cast(original_w, tf.int32), 3], image.dtype)
    newImage = tf.concat([HUPad, image, HDPad], 0)

    hpadded_w = tf.cast(tf.shape(newImage)[1], tf.float32)
    hpadded_h = tf.cast(tf.shape(newImage)[0], tf.float32)

    WLPad = tf.zeros([tf.cast(hpadded_h, tf.int32), tf.cast(expandXL*original_w, tf.int32), 3], image.dtype)
    WRPad = tf.zeros([tf.cast(hpadded_h, tf.int32), tf.cast(expandXR*original_w, tf.int32), 3], image.dtype)
    newImage = tf.concat([WLPad, newImage, WRPad], 1)

    new_bbox = tf.stack([
            ((bbox[..., 0] + expandYU)/(expandYU + expandYD + 1.0)),
            ((bbox[..., 1] + expandXL)/(expandXL + expandXR + 1.0)),
            ((bbox[..., 2] + expandYU)/(expandYU + expandYD + 1.0)),
            ((bbox[..., 3] + expandXL)/(expandXL + expandXR + 1.0))
        ], axis= -1)
    return newImage, new_bbox

def mixUp(ds1, ds2):
    def _sample_beta_distribution(size, concentration_0=0.5, concentration_1=0.5):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)
    
    images_one, bboxes_one, classes_one = ds1
    images_two, bboxes_two, classes_two = ds2

    images = images_one * 0.5 + images_two * (1 - 0.5)
    return images, tf.concat([bboxes_one, bboxes_two], 0), tf.concat([classes_one, classes_two], 0)

def mosaic(ds1, ds2, ds3, ds4):
    images1, bboxes1, classes1 = ds1
    images2, bboxes2, classes2 = ds2
    images3, bboxes3, classes3 = ds3
    images4, bboxes4, classes4 = ds4
    
    h, w, _ = tf.unstack(tf.shape(images1))

    images1, bboxes1, classes1  = randomCrop(images1, bboxes1, classes1)
    images2, bboxes2, classes2  = randomCrop(images2, bboxes2, classes2)
    images3, bboxes3, classes3  = randomCrop(images3, bboxes3, classes3)
    images4, bboxes4, classes4  = randomCrop(images4, bboxes4, classes4)
    
    border = tf.random.uniform([2], h//5*2, h//5*3, tf.int32)

    images1 = tf.image.resize(images1, [border[0], border[1]])
    images2 = tf.image.resize(images2, [border[0], w-border[1]])
    images3 = tf.image.resize(images3, [h-border[0], border[1]])
    images4 = tf.image.resize(images4, [h-border[0], w-border[1]])
    output_image = tf.concat([tf.concat([images1, images2], 1), tf.concat([images3, images4], 1)], 0)

    border = tf.cast(border/h, tf.float32)
    bboxes1 = tf.concat([border,border], -1)*bboxes1
    bboxes2 = tf.stack([border[0], 1-border[1], border[0], 1-border[1]], -1)*bboxes2+tf.stack([0.0,border[1],0.0,0.0], -1)
    bboxes3 = tf.stack([1-border[0], border[1], 1-border[0], border[1]], -1)*bboxes3+tf.stack([border[0],0.0,0.0,0.0], -1)
    bboxes4 = tf.stack([1-border[0], 1-border[1], 1-border[0], 1-border[1]], -1)*bboxes4+tf.stack([border[0],border[1],0.0,0.0], -1)

    output_boxes = tf.concat([bboxes1, bboxes2, bboxes3, bboxes4], 0)

    output_classes = tf.concat([classes1, classes2, classes3, classes4], 0)
    return output_image, output_boxes, output_classes