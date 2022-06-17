import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np

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