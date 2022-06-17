import tensorflow as tf

from model.BackBone.builder import BackBoneBuild
from model.Neck.builder import NeckBuild
from model.Head.builder import HeadBuild

from utils_train.Encoder import AnchorBox
from utils_train.utils import convert_to_corners

_policy=tf.keras.mixed_precision.global_policy()

def get_scaled_losses(loss, regularization_losses=None):
    loss = tf.reduce_mean(loss)
    if regularization_losses:
        loss = loss + tf.math.add_n(regularization_losses)
    return loss

def reduce_losses(losses_dict):
    for key, value in losses_dict.items():
        losses_dict[key] = tf.reduce_mean(value)
    return losses_dict

class DecodePredictions(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(DecodePredictions, self).__init__(**kwargs)
        self._Anchors= AnchorBox(config).get_anchors()
        self._loc_variance=tf.constant(config['model_config']['box_variances'], dtype=_policy.compute_dtype)
        self.iou_threshold=0.6
        self.score_threshold=0.1
        self.max_detections=100
        self._num_classes=config['training_config']["num_classes"]
        self._mode = config['training_config']['BoxLoss']['LossFunction'].lower()

    def _decode_box_predictions(self, loc_predictions):
        boxes=loc_predictions * self._loc_variance
        boxes=tf.concat([
            boxes[..., :2] * self._Anchors[..., 2:]+self._Anchors[..., :2],
            tf.math.exp(boxes[..., 2:]) * self._Anchors[..., 2:],
            ],axis=-1)

        return convert_to_corners(boxes)

    def _combined_nms(self, predictions):
        scores=tf.nn.sigmoid(predictions[..., 4:])

        if self._mode=='smoothl1':
            boxes_decoded=self._decode_box_predictions(predictions[..., :4])
        else:
            boxes_decoded=self._decode_box_predictions(predictions[..., :4])

        if len(boxes_decoded.get_shape().as_list()) == 3:
            boxes_decoded=tf.expand_dims(boxes_decoded, axis=2)
            
        detections=tf.image.combined_non_max_suppression(
            boxes=boxes_decoded,
            scores=scores,
            max_output_size_per_class=self.max_detections,
            max_total_size=self.max_detections,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            clip_boxes=True,
            name='combined_nms')

        return detections.nmsed_boxes, detections.nmsed_classes, detections.nmsed_scores, detections.valid_detections

    def call(self, predictions):
        return self._combined_nms(tf.cast(predictions, tf.float32))

    def get_config(self):
        config=super().get_config()
        config.update({
            "config": config,
        })
        return config

class ModelBuilder(tf.keras.Model):
    def __init__(self, config, **kwargs):
        backbone=BackBoneBuild(config)
        neck=NeckBuild(config)
        head=HeadBuild(config)

        inputs=tf.keras.Input((config["model_config"]["target_size"], config["model_config"]["target_size"], 3), name="inputlayer")
        features=backbone(inputs, config)
        features=neck(features, config)
        outputs=head(features, config)

        super().__init__(inputs=[inputs],
                        outputs=outputs,
                        name='Detector')

        self.config=config
         

    def compile(self, loss, optimizer, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, metrics=metrics, **kwargs)
        self.loss_fn=loss
        self.optimizer=optimizer

    def train_step(self, data):
        images, y_true=data

        with tf.GradientTape() as tape:
            y_pred=self(images, training=True)
            loss_values=self.loss_fn(y_true, y_pred)

            cls_loss=loss_values[0]    #[batch]
            loc_loss=loss_values[1]    #[batch]

            loss=cls_loss+loc_loss
            _scaled_losses=get_scaled_losses(loss, self.losses)
            #_scaled_losses=self.optimizer.get_scaled_loss(_scaled_losses)#
        
        scaled_gradients = tape.gradient(_scaled_losses, self.trainable_variables)
        #scaled_gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)#
        self.optimizer.apply_gradients(zip(scaled_gradients, self.trainable_variables))

        loss_dict={
                    'ClassL': cls_loss,
                    'BoxL': loc_loss,
                    'RegL': self.losses,
                    'TotalL': loss
                }
        
        return reduce_losses(loss_dict)

    def test_step(self, data):
        images, y_true, _=data
        
        y_pred=self(images, training=False)
        loss_values=self.loss_fn(y_true, y_pred)
        cls_loss=loss_values[0]
        loc_loss=loss_values[1]

        loss=cls_loss+loc_loss

        loss_dict={
                    'ClassL': cls_loss,
                    'BoxL': loc_loss,
                    'TotalL': loss
                }
                    
        return reduce_losses(loss_dict)

    def predict_step(self, images):
        self._decode_predictions=DecodePredictions(self.config)  
        predictions=self(images, training=False)
        return self._decode_predictions(predictions)