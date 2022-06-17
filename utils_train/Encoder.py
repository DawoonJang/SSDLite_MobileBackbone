import tensorflow as tf
from utils_train.utils import CalculateIOU

class AnchorBox():
    def __init__(self, config):
        self.target_size = config["model_config"]["target_size"]
        self._num_anchors = config["model_config"]["numAnchors"]
        self.FeatureMapResolution = config["model_config"]["feature_map_shapes"]
        self._anchorSpecs = config["model_config"]["AnchorSpecs"]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        anchor_lists = []
        for AnchorSpec in self._anchorSpecs:
            anchor_list = []
            isDual, next_scale = AnchorSpec[-1]
            for Anch in AnchorSpec[:-1]:
                scale, ratio = Anch
                box_width = scale * self.target_size * tf.sqrt(ratio)
                box_height = scale * self.target_size / tf.sqrt(ratio)
                anchor_list.append((box_height, box_width))

            if isDual:
                box_height = box_width = tf.sqrt(scale * next_scale) * self.target_size
                anchor_list.append((box_height, box_width))

            anchor_list = tf.stack(anchor_list)[tf.newaxis, tf.newaxis, ...]
            anchor_lists.append(anchor_list)
            
        return anchor_lists

    def _get_anchors(self, feature_size, level):
        gridSize = tf.math.ceil(self.target_size / feature_size)
        cx = tf.linspace(0.5*gridSize, self.target_size - 0.5*gridSize, feature_size)
        cy = tf.linspace(0.5*gridSize, self.target_size - 0.5*gridSize, feature_size)

        x_centers, y_centers = tf.meshgrid(cx, cy)
        widths_grid, x_centers_grid  = tf.meshgrid(gridSize, x_centers)
        heights_grid, y_centers_grid = tf.meshgrid(gridSize, y_centers)
        centers = tf.stack([y_centers_grid, x_centers_grid], axis=-1)
        centers = tf.reshape(centers, [feature_size,feature_size, -1, 2])
        centers = tf.tile(centers, [1, 1, self._num_anchors[level], 1])
        dims = tf.tile(self._anchor_dims[level], [feature_size, feature_size, 1, 1])

        anchors = tf.concat([centers, dims], axis=-1)

        #clip
        #anchors = convert_to_corners(anchors)
        #anchors = tf.clip_by_value(anchors, 0.0, self.target_size)
        #anchors = convert_to_xywh(anchors)

        #normalize
        anchors = anchors/self.target_size
        anchors = tf.reshape(anchors, [feature_size * feature_size * self._num_anchors[level], 4])
        return anchors
        
    def get_anchors(self):
        anchors = [
            self._get_anchors(
                FeatureRes,
                i,
            )
            for i, FeatureRes in enumerate(self.FeatureMapResolution)
        ]
        return tf.concat(anchors, axis=0)


class LabelEncoder():
    def __init__(self, config):
        self._anchor_box = AnchorBox(config).get_anchors()
        self._box_variance = tf.convert_to_tensor(config['model_config']['box_variances'], dtype=tf.float32)
        self._mode = config['training_config']['BoxLoss']['LossFunction'].lower()
        self._num_classes = config['training_config']["num_classes"]

    def _match_anchor_boxes(self, gt_boxes, match_iou=0.5, ignore_iou=0.5):
        cost_matrix = CalculateIOU(self._anchor_box, gt_boxes)
        max_iou = tf.reduce_max(cost_matrix, axis=1)
        matched_gt_idx = tf.argmax(cost_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))

        return (
            matched_gt_idx,
            positive_mask,
            ignore_mask, 
            max_iou
        )

    def _compute_box_target(self, matched_gt_boxes):
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - self._anchor_box[:, :2]) / self._anchor_box[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / self._anchor_box[:, 2:])
            ],
            axis=-1,
        )
        return box_target / self._box_variance

    def _encode_sample(self, gt_boxes, cls_ids):
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask, max_iou = self._match_anchor_boxes(gt_boxes)

        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx) 
        
        box_target = self._compute_box_target(matched_gt_boxes)
        box_target = tf.where(tf.expand_dims(positive_mask, -1), box_target, 1e-8) ## why no nan by this

        cls_target = tf.where(positive_mask, matched_gt_cls_ids, -1.0)
        cls_target = tf.where(ignore_mask, -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)

        label = tf.concat([box_target, cls_target, tf.expand_dims(max_iou, -1)], axis=-1)
        return label

    def _encode_batch(self, gt_boxes, cls_ids):
        batch_size = tf.shape(gt_boxes)[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        return labels.stack()