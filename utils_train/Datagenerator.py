import tensorflow as tf
import tensorflow_datasets as tfds

from utils_train.utils import *
from utils_train.Augmentation import *

_policy = tf.keras.mixed_precision.global_policy()

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
    
class DatasetBuilder():
    def __init__(self, config, mode='train'):
        assert(mode in ['train', 'validation', 'bboxtest'])
        self._dataset = None

        self._label_encoder = LabelEncoder(config)
        self._target_size = config["model_config"]["target_size"]
        self._batch_size = config["training_config"]["batch_size"]

        self.mode = mode
        
        if mode == 'train' or mode == 'bboxtest':
             [self._tfrecords], dataset_info = tfds.load(name="coco/2017", split=["train"], with_info=True, shuffle_files=True)
             self.labelMapFunc = dataset_info.features["objects"]["label"].int2str
        else:
             [self._tfrecords] = tfds.load(name="coco/2017", split=["validation"], with_info=False, shuffle_files=False)

        self._build_dataset()
    
    def _preprocess(self, samples):
        '''
            in_bbox_format: [ymin xmin ymax xmax]
            out_bbox_format: [cy cx h w]
        '''

        image = samples["image"]
        originalShape = tf.shape(image)[:2]
        classes = tf.cast(samples["objects"]["label"], dtype=tf.int32)
        bbox = samples["objects"]["bbox"]
        ####################################
        noCrowMask =  tf.logical_not(samples["objects"]["is_crowd"])
        classes = tf.boolean_mask(classes, noCrowMask)
        bbox = tf.boolean_mask(bbox, noCrowMask)
        ####################################
        validboxMask = tf.reduce_all(bbox[..., 2:] > bbox[..., :2], -1)
        classes = tf.boolean_mask(classes, validboxMask)
        bbox = tf.boolean_mask(bbox, validboxMask)
        ####################################
        #humanMask = tf.equal(classes, 0)
        #classes = tf.boolean_mask(classes, humanMask)
        #bbox = tf.boolean_mask(bbox, humanMask)
        ####################################

        if self.mode == 'train' or self.mode == 'bboxtest':
            image, bbox, classes  = randomCrop(image, bbox, classes, p = 0.9)
            image, bbox           = randomExpand(image, bbox, expandMax = 0.3, p = 0.2)
            image, bbox           = randomResize(image, bbox, self._target_size, self._target_size, p = 0.0)
            image, bbox           = flipHorizontal(image, bbox, p = 0.5)
            #image                 = colorJitter(image, p = 0.3)
        else:
            image, bbox           = randomResize(image, bbox, self._target_size, self._target_size, p = 0.0)

        if self.mode == 'train' or self.mode == 'validation':
            if len(bbox) > 0:
                bbox = tf.concat([
                    (bbox[..., :2]+bbox[..., 2:])/2.0,
                    (bbox[..., 2:]-bbox[..., :2])],axis=-1)
            else:
                bbox = tf.zeros([1, 4], tf.float32)
                classes = -1*tf.ones([1], tf.int32)
        
        cocoLabel = {"original_shape": originalShape, "image_id": samples['image/id']}
        if self.mode == 'bboxtest':
            return image, bbox, classes
        elif self.mode == 'train':
            return (image/127.5) -1.0, self._label_encoder._encode_sample(bbox, classes)
        else:
            return (image/127.5) -1.0, self._label_encoder._encode_sample(bbox, classes), cocoLabel

    def _build_dataset(self):
        self._tfrecords = self._tfrecords.filter(lambda samples: len(samples["objects"]["label"]) >= 1) #117266 #4952  and tf.reduce_any(samples["objects"]["label"] == 0)
        
        if self.mode == 'train':
            self._dataset = (
                self._tfrecords#.with_options(options)
                .shuffle(16*self._batch_size, reshuffle_each_iteration=False)
                .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size=self._batch_size, drop_remainder = True)
                .prefetch(tf.data.AUTOTUNE)
            )

        elif self.mode == 'validation':
            self._dataset = (
                self._tfrecords
                .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size=self._batch_size, drop_remainder = False)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            self._dataset = (
                self._tfrecords
                .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            )
            
    @property
    def dataset(self):
        return self._dataset



class DatasetBuilder_custom():
    def __init__(self,  config, mode='train'):
        assert(mode in ['train', 'valid', 'bboxtest']) #'invalid'
        self._dataset = None

        self._label_encoder = LabelEncoder(config)
        self._target_size = config["model_config"]["target_size"]
        self._batch_size = config["training_config"]["batch_size"]

        self.mode = mode

        self._tfrecords = tf.data.Dataset.list_files("data/P.tfrecord")
        self._build_dataset()
    
    def _preprocess(self, samples):
        '''
            in_bbox_format: [ymin xmin ymax xmax]
            out_bbox_format: [cy cx h w]
        '''
        ###################################################
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }

        parsed_example = tf.io.parse_single_example(samples,
                                                    feature_description)
        classes = tf.sparse.to_dense(parsed_example['image/object/class/label'])
        classes = tf.cast(classes, tf.int32)-1

        image = tf.io.decode_image(parsed_example['image/encoded'], channels=3)
        image = tf.cast(image, dtype=tf.uint8)
        image.set_shape([None, None, 3])

        bbox = tf.stack([
            tf.sparse.to_dense(parsed_example['image/object/bbox/ymin']),
            tf.sparse.to_dense(parsed_example['image/object/bbox/xmin']),
            tf.sparse.to_dense(parsed_example['image/object/bbox/ymax']),
            tf.sparse.to_dense(parsed_example['image/object/bbox/xmax']),
        ], axis=-1)
        ###################################################

        if self.mode == 'train' or self.mode == 'bboxtest':
            image, bbox, classes  = randomCrop(image, bbox, classes, p = 0.9)
            #image, bbox           = randomExpand(image, bbox, expandMax = 0.3, p = 0.4)
            image, bbox           = randomResize(image, bbox, self._target_size, self._target_size, p = 0.0)
            image, bbox           = flipHorizontal(image, bbox, p = 0.5)
            image, bbox           = flipVertical(image, bbox, p = 0.5)
            #image, bbox           = cutOut(image, bbox, p = 1.0)
            image                 = colorJitter(image, p = 0.5)
        else:
            image, bbox           = randomResize(image, bbox, self._target_size, self._target_size, p = 0.0)

        if self.mode == 'train' or self.mode == 'valid':
            if len(bbox) > 0:
                bbox = tf.concat([
                    (bbox[..., :2]+bbox[..., 2:])/2.0,
                    (bbox[..., 2:]-bbox[..., :2])],axis=-1)
            else:
                bbox = tf.zeros([1, 4], tf.float32)
                classes = -3*tf.ones([1], tf.int32)
        
        if self.mode == 'bboxtest':
            return image, bbox, classes
        else:
            return image, self._label_encoder._encode_sample(bbox, classes), {"Dummy":1}

    def _build_dataset(self):        
        if self.mode == 'train':
            options = tf.data.Options()
            options.deterministic = False
            self._dataset = (
                self._tfrecords.interleave(tf.data.TFRecordDataset,
                                            cycle_length=256,
                                            block_length=16,
                                            num_parallel_calls=tf.data.AUTOTUNE)
                .with_options(options)
                .shuffle(8*self._batch_size)
                .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size=self._batch_size, drop_remainder = False)
                .prefetch(tf.data.AUTOTUNE)
            )

        else:
            self._dataset = (
                self._tfrecords.interleave(tf.data.TFRecordDataset,
                                            cycle_length=256,
                                            block_length=16,
                                            num_parallel_calls=tf.data.AUTOTUNE)
                .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            )
    @property
    def dataset(self):
        return self._dataset