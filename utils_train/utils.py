import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np

_policy = tf.keras.mixed_precision.global_policy()

def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


'''def convert_to_xywh(boxes):
    return tf.concat([
      (boxes[..., :2] + boxes[..., 2:]) / 2.0, 
      boxes[..., 2:] - boxes[..., :2]
      ], axis=-1)'''

def convert_to_xywh(boxes):
    ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(boxes))
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return tf.transpose(tf.stack([ycenter, xcenter, height, width]))

def convert_to_corners(boxes):
    return tf.concat([
      boxes[..., :2] - boxes[..., 2:] / 2.0, 
      boxes[..., :2] + boxes[..., 2:] / 2.0],axis=-1)

def CalculateIOU(boxes1, boxes2):
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)

    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2]) #[BOX1 BOX2 2]
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:]) #[BOX1 BOX2 2]

    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]

    union_area = boxes1_area[:, None] + boxes2_area - intersection_area
    return tf.math.divide_no_nan(intersection_area, union_area)

def CalculateCIOU(boxes_gt, boxes_pred):
    '''
        input_format x1 y1 x2 y2
    '''
    inter_lu = tf.maximum(boxes_gt[:, :, :2], boxes_pred[:, :, :2])
    inter_rd = tf.minimum(boxes_gt[:, :, 2:], boxes_pred[:, :, 2:])
    
    outer_lu = tf.minimum(boxes_gt[:, :, :2], boxes_pred[:, :, :2])
    outer_rd = tf.maximum(boxes_gt[:, :, 2:], boxes_pred[:, :, 2:])

    w_gt = boxes_gt[:, :, 2] - boxes_gt[:, :, 0]
    h_gt = boxes_gt[:, :, 3] - boxes_gt[:, :, 1]
    w_pred = boxes_pred[:, :, 2] - boxes_pred[:, :, 0]
    h_pred = boxes_pred[:, :, 3] - boxes_pred[:, :, 1]

    center_x_gt = (boxes_gt[:, :, 2] + boxes_gt[:, :, 0]) / 2
    center_y_gt = (boxes_gt[:, :, 3] + boxes_gt[:, :, 1]) / 2
    center_x_pred = (boxes_pred[:, :, 2] + boxes_pred[:, :, 0]) / 2
    center_y_pred = (boxes_pred[:, :, 3] + boxes_pred[:, :, 1]) / 2

    boxes_gt_area = (boxes_gt[:, :, 2] - boxes_gt[:, :, 0]) * (boxes_gt[:, :, 3] - boxes_gt[:, :, 1])
    boxes_pred_area = (boxes_pred[:, :, 2] - boxes_pred[:, :, 0]) * (boxes_pred[:, :, 3] - boxes_pred[:, :, 1])

    inter_intersection = tf.maximum(0.0, inter_rd - inter_lu)
    inter_intersection_area = inter_intersection[:, :, 0] * inter_intersection[:, :, 1]
    union_area = tf.maximum(boxes_gt_area + boxes_pred_area - inter_intersection_area, 1e-8)
    
    iou = tf.clip_by_value(inter_intersection_area/union_area, 0.0, 1.0)

    outer_intersection = tf.maximum(0.0, outer_rd - outer_lu)
    c = (outer_intersection[:, :, 0]**2) + (outer_intersection[:, :, 1]**2)
    d = (center_x_gt - center_x_pred)**2 + (center_y_gt - center_y_pred)**2
    u = d / c

    arctanTerm = tf.math.atan(w_gt / (h_gt+1e-8)) - tf.math.atan(w_pred / (h_pred+1e-8))
    v = 4 / (np.pi ** 2) * tf.pow(arctanTerm, 2)
    ar = 8 / (np.pi ** 2) * arctanTerm * ((w_pred - 2 * w_pred) * h_pred)

    S = 1 - iou
    alpha = v/(S + v + 1e-8)

    cious = iou - (u + alpha * ar)
    return tf.clip_by_value(cious, -1.0, 1.0)

def CalculateDIOU(boxes_gt, boxes_pred):
    '''
        input_format x1 y1 x2 y2
    '''
    inter_lu = tf.maximum(boxes_gt[..., :2], boxes_pred[..., :2])
    inter_rd = tf.minimum(boxes_gt[..., 2:], boxes_pred[..., 2:])
    
    outer_lu = tf.minimum(boxes_gt[..., :2], boxes_pred[..., :2])
    outer_rd = tf.maximum(boxes_gt[..., 2:], boxes_pred[..., 2:])

    center_x_gt = (boxes_gt[..., 2] + boxes_gt[..., 0]) / 2
    center_y_gt = (boxes_gt[..., 3] + boxes_gt[..., 1]) / 2
    center_x_pred = (boxes_pred[..., 2] + boxes_pred[..., 0]) / 2
    center_y_pred = (boxes_pred[..., 3] + boxes_pred[..., 1]) / 2

    boxes_gt_area = (boxes_gt[..., 2] - boxes_gt[..., 0]) * (boxes_gt[..., 3] - boxes_gt[..., 1])
    boxes_pred_area = (boxes_pred[..., 2] - boxes_pred[..., 0]) * (boxes_pred[..., 3] - boxes_pred[..., 1])

    inter_intersection = tf.maximum(0.0, inter_rd - inter_lu)
    inter_intersection_area = inter_intersection[..., 0] * inter_intersection[..., 1]
    union_area = tf.maximum(boxes_gt_area + boxes_pred_area - inter_intersection_area, 1e-8)
    
    iou = tf.clip_by_value(inter_intersection_area/union_area, 0.0, 1.0)

    outer_intersection = tf.maximum(0.0, outer_rd - outer_lu)
    c = (outer_intersection[..., 0]**2) + (outer_intersection[..., 1]**2)
    d = (center_x_gt - center_x_pred)**2 + (center_y_gt - center_y_pred)**2
    u = d / c

    Dious = iou - u
    return tf.clip_by_value(Dious, -1.0, 1.0)
    #return -tf.math.log(tf.clip_by_value(iou,1e-8,1.0))+u
################################## from TFOD
def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape

def matmul_gather_on_zeroth_axis(params, indices): #tf.gayher
    params_shape = combined_static_and_dynamic_shape(params)
    indices_shape = combined_static_and_dynamic_shape(indices)
    params2d = tf.reshape(params, [params_shape[0], -1])
    indicator_matrix = tf.one_hot(indices, params_shape[0])
    gathered_result_flattened = tf.matmul(indicator_matrix, params2d)

    return tf.reshape(gathered_result_flattened, tf.stack(indices_shape + params_shape[1:]))