import tensorflow as tf
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model.ModelBuilder import ModelBuilder
from utils_train.Datagenerator import DatasetBuilder

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tqdm.notebook import trange, tqdm

modelName = "MobileDet_PFH_SSD"
#modelName = "MobileDet_PFH_SSD"

model_dir = "checkpoints/"
modelPart = modelName.split("_")

with open(os.path.join("model", "0_Config", modelName+".json"), "r") as config_file:
    config = json.load(config_file)

config['modelName'] = modelName
config['training_config']['num_classes'] = 80

model = ModelBuilder(config)
model.load_weights("logs/MB3Det_SSDLite_AllClass_21_0_HClass_289").expect_partial()
#model.load_weights("logs/MB3Small_SSDLite_FuncH1_23_9_ac_6").expect_partial()

test_dataset = DatasetBuilder(config, mode = 'validation')
_processed_detections = []

_labelMapList = [1, 2, 3, 4, 5, 6, 7, 8, 9, \
            10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
            23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, \
            38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, \
            51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, \
            63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, \
            79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

if __name__=="__main__":
    model.summary()

    _coco_eval_obj = COCO("data/coco_annotation/annotations/instances_val2017.json")
    for sample in tqdm(test_dataset.dataset):
        images = sample[0]
        encoded_label = sample[1]
        cocoLabel = sample[2]

        final_bboxes, final_labels, final_scores, final_num = model.predict(images)
        image_ids = cocoLabel["image_id"]
        original_shape = cocoLabel["original_shape"]

        coco_eval_dict = {
            'image_id': None,
            'category_id': None,
            'bbox': [],
            'score': None
        }

        for idx, image_id in enumerate(image_ids):
            valid_detections = final_num[idx]
            boxes = final_bboxes[idx][:valid_detections]
            classes = final_labels[idx][:valid_detections]
            scores = final_scores[idx][:valid_detections]
            originalSize = tf.cast(original_shape[idx], tf.float32)
            
            #####
                #input bbox format y1 x1 y2 x2
                #output bbox format x1 y1 w h
                
            boxes = np.stack([
                boxes[..., 1]*originalSize[1], #x1 * w
                boxes[..., 0]*originalSize[0], #y1 * h
                (boxes[..., 3] - boxes[..., 1])*originalSize[1],
                (boxes[..., 2] - boxes[..., 0])*originalSize[0]
                ], axis=-1)
            #####
            
            for box, int_id, score in zip(boxes, classes, scores):
                temp_dict = coco_eval_dict.copy()
                temp_dict['image_id'] = int(image_id)
                temp_dict['category_id'] = _labelMapList[int(int_id)] if int_id < 80 else int(int_id)
                temp_dict['bbox'] = box.tolist()
                temp_dict['score'] = float(score)
                
                _processed_detections.append(temp_dict)

    with open("data/coco_annotation/annotations/inference.json", 'w') as f:
        json.dump(_processed_detections, f, indent=4)


    predictions = _coco_eval_obj.loadRes("data/coco_annotation/annotations/inference.json")

    cocoEval = COCOeval(_coco_eval_obj, predictions, 'bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


    cocoEval.params.catIds = [1]
        
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()