import tensorflow as tf
import numpy as np

def intersection(boxlist1, boxlist2):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1, num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2, num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def area(boxlist):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist, num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

def ioa(boxlist1, boxlist2):
    """Computes pairwise intersection-over-area between box collections.
    intersection-over-area (IOA) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, ioa(box1, box2) != ioa(box2, box1).
    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes
      scope: name scope.
    Returns:
      a tensor with shape [N, M] representing pairwise ioa scores.
    """
    intersections = intersection(boxlist1, boxlist2)
    areas = tf.expand_dims(area(boxlist2), 0)
    return tf.truediv(intersections, areas)

def scale(boxlist, y_scale, x_scale, scope=None):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(value=boxlist, num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = tf.concat([y_min, x_min, y_max, x_max], 1)
    return scaled_boxlist
