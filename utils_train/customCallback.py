import tensorflow as tf
import numpy as np
import json
import os

from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils_train.customLrScheduler import CosineDecayWithLinearWarmup

class Logger(tf.keras.callbacks.Callback):
    def __init__(self,
                config,
                val_data,
                annotation_file_path,
                prediction_file_path):
        super().__init__()
        self.val_data = val_data

        self.weightsave_path =os.path.join('logs', config['modelName'])

        self.prediction_file_path = os.path.normpath(prediction_file_path)
        self._coco_eval_obj = COCO(annotation_file_path)

        self._train_summary = tf.summary.create_file_writer("logs/" + config['modelName'] +"/tensorboard/train")
        self._valid_summary = tf.summary.create_file_writer("logs/" + config['modelName'] +"/tensorboard/valid")

        self._eval_interval = 10
        self._min_loss = 1000.0

        self.patience = 0

        self._labelMapList = [1, 2, 3, 4, 5, 6, 7, 8, 9, \
            10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
            23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, \
            38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, \
            51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, \
            63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, \
            79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    def on_epoch_end(self, epoch, logs=None):
        self._write_summaries(epoch, logs)
        
        if logs["val_TotalL"] + 0.01 < self._min_loss and epoch > 10:
            self._min_loss = logs["val_TotalL"]
            isImprove = True
            self.patience = 0
        else:
            isImprove = False
            self.patience += 1

        if self.patience >= 5:
            #tf.keras.backend.set_value(self.model.optimizer.lr, self.model.optimizer.lr*0.9)
            self.patience = 0

        if (isImprove or ((epoch + 1) % self._eval_interval == 0)) and epoch > 3:
            self._cocoeval(epoch)

    def accumulate_results(self, sample):
        images = sample[0]
        encoded_label = sample[1]
        cocoLabel = sample[2]

        final_bboxes, final_labels, final_scores, final_num = self.model.predict(images)
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
                try:
                    temp_dict['image_id'] = int(image_id)
                except:
                    temp_dict['image_id'] = int(image_id.numpy()[:-4])
                temp_dict['category_id'] = self._labelMapList[int(int_id)] if int_id < 80 else int(int_id)
                temp_dict['bbox'] = box.tolist()
                temp_dict['score'] = float(score)
                
                self._processed_detections.append(temp_dict)

    def _write_summaries(self, epoch, logs):
        with self._train_summary.as_default():
            with tf.name_scope('Loss'):
                tf.summary.scalar(name="ClassL", data=logs["ClassL"], step=epoch)
                tf.summary.scalar(name="BoxL", data=logs["BoxL"], step=epoch)
                tf.summary.scalar(name="TotalL", data=logs["TotalL"], step=epoch)
                tf.summary.scalar(name="RegL", data=logs["RegL"], step=epoch)

            with tf.name_scope('LearningRate'):
                tf.summary.scalar(name="LearngingRate", data=self.model.optimizer.lr, step=epoch)

            with tf.summary.record_if(epoch % self._eval_interval == 0):
                for layer in self.model.layers:
                    for weight in layer.weights:
                        weight_name = weight.name.replace(':', '_')
                        tf.summary.histogram(name=weight_name, data=weight, step=epoch)

        with self._valid_summary.as_default():
            with tf.name_scope('Loss'):
                tf.summary.scalar(name="ClassL", data=logs["val_ClassL"], step=epoch)
                tf.summary.scalar(name="BoxL", data=logs["val_BoxL"], step=epoch)
                tf.summary.scalar(name="TotalL", data=logs["val_TotalL"], step=epoch)

        self._train_summary.flush()
        self._valid_summary.flush()
            
    def _write_mAP(self, scores, epoch):
        with self._valid_summary.as_default():
            with tf.name_scope('mAP'):
                for key, value in scores.items():
                    tf.summary.scalar(name=key, data=value, step=epoch)

        self._valid_summary.flush()

    def _cocoeval(self, epoch, catIds = None):
        self._processed_detections = []
        for sample in self.val_data:
            self.accumulate_results(sample)

        if len(self._processed_detections) > 0:
            with open(self.prediction_file_path, 'w') as f:
                json.dump(self._processed_detections, f, indent=4)

            predictions = self._coco_eval_obj.loadRes(self.prediction_file_path)

            cocoEval = COCOeval(self._coco_eval_obj, predictions, 'bbox')

            if catIds is not None: 
                cocoEval.params.catIds = catIds #[1]
                
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            self.model.save_weights(os.path.join(self.weightsave_path, "weights", "_epoch"+str(epoch+1)+"_mAP"+'%.3f'%cocoEval.stats[0]))

            scores = {
                'AP-IoU=0.50:0.95': cocoEval.stats[0],
                'AP-IoU=0.50': cocoEval.stats[1],
                'AP-IoU=0.75': cocoEval.stats[2],
                'AP-Small': cocoEval.stats[3],
                'AP-Medium': cocoEval.stats[4],
                'AP-Large': cocoEval.stats[5],
                'AR-(all)-IoU=0.50:0.95': cocoEval.stats[8],
            }
            
            self._write_mAP(scores, epoch)

class CallbackBuilder():
    def __init__(self, config, dataForEval, val_file="data/pascal_test2007.json"):
        self.Logger = Logger(
            val_data = dataForEval,
            config = config,
            annotation_file_path = val_file,
            prediction_file_path = "data/inference.json"
        )
            
        self.LrScheduler = tf.keras.callbacks.LearningRateScheduler(
            CosineDecayWithLinearWarmup(initial_learning_rate = config["training_config"]["initial_learning_rate"],
                                        warmup_learning_rate = config["training_config"]["initial_learning_rate"]/3,
                                        warmup_steps = 4,
                                        total_steps = config["training_config"]["epochs"]))

        self.TB = tf.keras.callbacks.TensorBoard(log_dir='logs/', profile_batch='500, 540')
    def get_callbacks(self):
        callbacks_list = [self.Logger, self.LrScheduler] 

        return callbacks_list