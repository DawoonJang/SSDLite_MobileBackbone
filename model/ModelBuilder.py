import tensorflow as tf
import numpy as np

from model.BackBone.builder import BackBoneBuild
from model.Neck.builder import NeckBuild
from model.Head.builder import HeadBuild

from utils_train.Encoder import AnchorBox
from utils_train.utils import convert_to_corners

_policy=tf.keras.mixed_precision.global_policy()
strategy = tf.distribute.get_strategy()

def get_scaled_losses(loss, regularization_losses=None):
    loss = tf.reduce_mean(loss)
    if regularization_losses:
        loss += tf.math.add_n(regularization_losses)
    return loss/strategy.num_replicas_in_sync

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
        self.total_loss_tracker = tf.keras.metrics.Mean(name="TotalL")
        self.class_loss_tracker = tf.keras.metrics.Mean(name="ClassL")
        self.box_loss_tracker = tf.keras.metrics.Mean(name="BoxL")

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

            total_loss=cls_loss+loc_loss
            _scaled_losses=get_scaled_losses(total_loss, self.losses)
            _scaled_losses=self.optimizer.get_scaled_loss(_scaled_losses)
        
        scaled_gradients = tape.gradient(_scaled_losses, self.trainable_variables)
        scaled_gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(scaled_gradients, self.trainable_variables))

        self.class_loss_tracker.update_state(cls_loss)
        self.box_loss_tracker.update_state(loc_loss)
        self.total_loss_tracker.update_state(total_loss)

        loss_dict={
                    'HeatL': self.class_loss_tracker.result(),
                    'BoxL': self.box_loss_tracker.result(),
                    'RegL': tf.reduce_mean(self.losses),
                    'TotalL': self.total_loss_tracker.result()
                }
        
        return loss_dict

    def test_step(self, data):
        images, y_true, _=data
        
        y_pred=self(images, training=False)
        loss_values=self.loss_fn(y_true, y_pred)
        cls_loss=loss_values[0]
        loc_loss=loss_values[1]

        self.class_loss_tracker.update_state(cls_loss)
        self.box_loss_tracker.update_state(loc_loss)
        self.total_loss_tracker.update_state(cls_loss+loc_loss)

        loss_dict={
                    'HeatL': self.class_loss_tracker.result(),
                    'BoxL': self.box_loss_tracker.result(),
                    'TotalL': self.total_loss_tracker.result()
                }
                    
        return loss_dict

    def predict_step(self, images):
        self._decode_predictions=DecodePredictions(self.config)  
        predictions=self(images, training=False)
        return self._decode_predictions(predictions)

    def __repr__(self, table=True):
        print_str=''
        if table:
            print_str += '%25s | %16s | %20s | %10s | %10s | %6s | %6s\n'%( 'Layer Name', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPs')
            print_str += '-'*170+'\n'
        t_flops = 0

        for l in self.layers:
            _layername = str(l).lower()
            o_shape, i_shape, strides, ks, filters = ['', '', ''], ['', '', ''], '', '', ''
            flops = 0
            name = l.name

            if 'inputlayer' in _layername:
                continue

            elif 'reshape' in _layername:
                i_shape = l.input.get_shape()[1:4].as_list()
                o_shape = l.output.get_shape()[1:4].as_list()

            elif 'add' in _layername or 'multiply' in _layername or 'maximum' in _layername or 'concatenate' in _layername:
                i_shape = [l_input.get_shape()[1:].as_list() for l_input in l.input]
                #i_shape = l.input.get_shape()[1:].as_list()
                o_shape = l.output.get_shape()[1:].as_list()                
                flops = (len(i_shape) - 1)
                for i in i_shape[0]:
                    flops *= i

            elif 'average' in _layername and 'pool' not in _layername:
                i_shape = l.input[0].get_shape()[1:].as_list() + [2]
                o_shape = l.output.get_shape()[1:].as_list()
                flops = len(l.input)*i_shape[0]*i_shape[1]*i_shape[2]

            elif 'pool' in _layername and 'global' not in _layername:
                i_shape = l.input.get_shape()[1:].as_list()
                o_shape = l.output.get_shape()[1:].as_list()
                strides = l.strides
                ks = l.pool_size
                flops = (o_shape[3]*o_shape[0]*o_shape[1])*(ks[0]*ks[1])

            elif 'global' in _layername:
                i_shape = l.input.get_shape()[1:].as_list()
                o_shape = l.output.get_shape()[1:].as_list()
                flops = (i_shape[0])*(i_shape[1])*(i_shape[2])

            elif 'batchnormalization' in _layername:
                i_shape = l.input.get_shape()[1:].as_list()
                o_shape = l.output.get_shape()[1:].as_list()

                flops = 1
                for i in range(len(i_shape)):
                    flops *= i_shape[i]

            elif 'relu' in _layername:
                i_shape = l.input.get_shape()[1:].as_list()
                o_shape = l.output.get_shape()[1:].as_list()
                
                flops = 1
                for i in range(len(i_shape)):
                    flops *= i_shape[i]

            elif 'hswish' in _layername or  'sigmoid' in _layername:
                i_shape = l.input.get_shape()[1:].as_list()
                o_shape = l.output.get_shape()[1:].as_list()
                
                flops = 4
                for i in range(len(i_shape)):
                    flops *= i_shape[i]

            elif 'flatten' in _layername:
                i_shape = l.input.shape[1:4].as_list()
                flops = 1
                out_vec = 1
                for i in range(len(i_shape)):
                    flops *= i_shape[i]
                    out_vec *= i_shape[i]
                o_shape = flops
                flops = 0

            elif 'padding' in _layername:
                flops = 0

            elif 'dense' in _layername:
                i_shape = l.input.shape[1:4].as_list()[0]
                if (i_shape == None):
                    i_shape = out_vec

                o_shape = l.output.shape[1:4].as_list()
                flops = 2*(o_shape[0]*i_shape)
            elif 'conv2d ' in _layername:
                strides = l.strides
                ks = l.kernel_size
                filters = l.filters if l.filters else 1
                i_shape = l.input.get_shape()[1:].as_list()
                o_shape = l.output.get_shape()[1:].as_list()

                if 'separableconv2d' in _layername:
                    flops = 2*(filters+ks[0]*ks[1])*(i_shape[2]*(o_shape[0])*(o_shape[1]))
                else:
                    flops = 2*(filters*ks[0]*ks[1])*(i_shape[2]*(o_shape[0])*(o_shape[1]))
            else:
                print("TP: ", l)

            t_flops += flops
            if table:
                if isinstance(i_shape[0], list):
                    print_str += '%25s | %16s | %20s | %10s | %10s | %6s | %5.2f[M]\n'%(name, str(i_shape[0]), str(o_shape), str(ks), str(filters), str(strides), flops/1e6)
                    for idx in range(len(i_shape)-1):
                        print_str += '%44s | \n'%(str(i_shape[idx+1]))
                else:
                    print_str += '%25s | %16s | %20s | %10s | %10s | %6s | %5.2f[M]\n'%(name, str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops/1e6)

        trainable_params = sum([np.prod(w.get_shape().as_list()) for w in self.trainable_weights])
        none_trainable_params = sum([np.prod(w.get_shape().as_list()) for w in self.non_trainable_weights])
        total_params = trainable_params+none_trainable_params

        print_str += '-'*170+'\n'
        print_str += '         Total Params: %6.3f[M]  ' % (total_params/1e6)+'Trainable Params: %6.3f[M]  ' % (trainable_params/1e6)+ \
        'Total FLOPS: %6.3f[G]  ' % (t_flops/1e9)
        return print_str