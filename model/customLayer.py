import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

from tensorflow.keras import backend
from tensorflow.keras.layers import Conv2D, BatchNormalization, Multiply, GlobalAveragePooling2D, \
    DepthwiseConv2D, Add, GlobalMaxPooling2D, Concatenate, ReLU, Dropout, SeparableConv2D

def _depth(filters, multiplier=1.0, base=8):
    round_half_up=int(int(filters) * multiplier / base+0.5)
    result=int(round_half_up * base)
    return max(result, base)

class ReLU6(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.nn.relu6(inputs)
    def get_prunable_weights(self):
        return []

class HSigmoid6(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.nn.relu6(inputs+np.float32(3)) * np.float32(1. / 6.)
    def get_prunable_weights(self):
        return []

class HSwish6(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return inputs * tf.nn.relu6(inputs+np.float32(3)) * np.float32(1. / 6.)
    def get_prunable_weights(self):
        return []

class Sigmoid(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.math.sigmoid(inputs)
    def get_prunable_weights(self):
        return []

class WeightedSum(tf.keras.layers.Layer):
    """
        A custom keras layer to learn a weighted sum of tensors
    """

    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape=1):
        self.a = self.add_weight(
            name='alpha',
            shape=(4,),
            initializer='ones',
            dtype='float32',
            trainable=True,
        )
        super(WeightedSum, self).build(input_shape)

    def call(self, model_outputs):
        return self.a[0] * model_outputs[0] + self.a[1] * model_outputs[1] + self.a[2] * model_outputs[2] + self.a[3] * model_outputs[3]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def _Conv(inputs, filters, kernel_size=3, strides=2, padding='same', 
        use_bias=False, normalization=BatchNormalization, activation=ReLU6,
        prefix=None, **conf_dict):

    x=Conv2D(filters=filters,
              kernel_size=kernel_size,
              strides=strides,
              padding=padding,
              use_bias=use_bias,
              kernel_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
              kernel_regularizer=tf.keras.regularizers.l2(conf_dict['reg']),
              bias_initializer=conf_dict['bias_initializer'] if 'bias_initializer'in conf_dict.keys() else 'zeros',
              trainable=conf_dict['trainable'],
              name=prefix+'Conv')(inputs)

    if normalization is not None:
        x=normalization(trainable=conf_dict['trainable'], name=prefix+'BN')(x)

    if activation is not None:
        x=activation(name=prefix+'AC')(x)

    return x

def _DeptwiseConv(inputs, kernel_size=3, strides=2, padding='same', use_bias=False, dilation_rate=1,
                normalization=BatchNormalization, activation=ReLU6,
                prefix=None, **conf_dict):

    x=DepthwiseConv2D(kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    dilation_rate=dilation_rate,
                    use_bias=use_bias,
                    kernel_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
                    kernel_regularizer=tf.keras.regularizers.l2(conf_dict['reg']),
                    bias_initializer=conf_dict['bias_initializer'] if 'bias_initializer'in conf_dict.keys() else 'zeros',
                    trainable=conf_dict['trainable'],
                    name=prefix+'DepwiseConv')(inputs)

    if normalization is not None:
        x=normalization(trainable=conf_dict['trainable'], name=prefix+'DepwiseBN')(x)

    if activation is not None:
        x=activation(name=prefix+'DepwiseAC')(x)

    return x

def _SeparableConv(inputs, filters, kernel_size=3, strides=2, padding='same', use_bias=False, normalization=BatchNormalization, activation=ReLU, prefix=None, **conf_dict):
    x=SeparableConv2D(filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            depthwise_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
            #depthwise_regularizer=tf.keras.regularizers.l2(conf_dict['reg']),
            pointwise_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
            pointwise_regularizer=tf.keras.regularizers.l2(conf_dict['reg']),
            trainable=conf_dict['trainable'],
            bias_initializer=conf_dict['bias_initializer'] if 'bias_initializer'in conf_dict.keys() else 'zeros',
            name=prefix+'Conv')(inputs)

    if normalization is not None:
        x=normalization(trainable=conf_dict['trainable'], name=prefix+'BN')(x)

    if activation is not None:
        x=activation(name=prefix+'AC')(x)

    return x

def _SeparableDepthwiseConv(inputs, filters, kernel_size=3, strides=2, padding='same', use_bias=False, normalization=BatchNormalization, activation=ReLU6, prefix=None, **conf_dict):
    x=DepthwiseConv2D(kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    use_bias=False,
                    kernel_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
                    #kernel_regularizer=tf.keras.regularizers.l2(conf_dict['reg']),
                    bias_initializer=conf_dict['bias_initializer'] if 'bias_initializer'in conf_dict.keys() else 'zeros',
                    trainable=conf_dict['trainable'],
                    name=prefix+'DepthwiseConv')(inputs)

    x=BatchNormalization(trainable=conf_dict['trainable'], name=prefix+'DepthwiseBN')(x)
    x=ReLU6(name=prefix+'DepthwiseAC')(x)

    x=Conv2D(filters=filters,
            kernel_size=1,
            strides=1,
            use_bias=use_bias,
            padding='valid',
            kernel_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
            kernel_regularizer=tf.keras.regularizers.l2(conf_dict['reg']),
            trainable=conf_dict['trainable'],
            bias_initializer=conf_dict['bias_initializer'] if 'bias_initializer'in conf_dict.keys() else 'zeros',
            name=prefix+'PointwiseConv')(x)

    if normalization is not None:
        x=normalization(trainable=conf_dict['trainable'], name=prefix+'PointwiseBN')(x)

    if activation is not None:
        x=activation(name=prefix+'PointwiseAC')(x)

    return x

def _SEBlock(inputs, se_ratio, prefix, activation, **conf_dict):
    '''
        Reference:
                "Squeeze-and-Excitation Networks (CVPR 2018)"
                "Searching for MobileNetV3 (ICCV 2019)"
    '''
    infilters=backend.int_shape(inputs)[-1]

    x=GlobalAveragePooling2D(keepdims=True, name=prefix+'SEAvgPool')(inputs)

    x=_Conv(x, filters=_depth(infilters*se_ratio), kernel_size=1, padding='valid', use_bias=True,
        normalization=None, activation=activation, prefix=prefix+'SE_1', **conf_dict)
    x=_Conv(x, filters=infilters, kernel_size=1, padding='valid', use_bias=True,
        normalization=None, activation=HSigmoid6, prefix=prefix+'SE_2', **conf_dict)

    return Multiply(name=prefix+'SEMul')([inputs, x])

def _LSCBlock(inputs, prefix, **conf_dict):
    '''
        Reference:
                "MAOD: An Efficient Anchor-Free Object Detector Based on MobileDet (IEEE 2020)"
    '''
    x=Concatenate(name=prefix+'Spacial_Concat')([tf.reduce_max(inputs, axis=-1, keepdims=True), tf.reduce_mean(inputs, axis=-1, keepdims=True)])
    x=tf.keras.layers.Conv2D(filters=1, 
                            kernel_size=(7, 1), 
                            padding="same",
                            kernel_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
                            kernel_regularizer=tf.keras.regularizers.l2(conf_dict['reg']),
                            trainable=conf_dict['trainable'],
                            name=prefix+'Spacial_Attention_Conv1')(x)
    x=HSigmoid6(name=prefix+'Spacial_Attention_Ac1')(x)
    x=tf.keras.layers.Conv2D(filters=1, 
                            kernel_size=(1, 7), 
                            padding="same",
                            kernel_initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
                            kernel_regularizer=tf.keras.regularizers.l2(conf_dict['reg']),
                            trainable=conf_dict['trainable'],
                            name=prefix+'Spacial_Attention_Conv2')(x)
    x=HSigmoid6(name=prefix+'Spacial_Attention_Ac2')(x)

    inputs2=Multiply(name=prefix+'Spacial_Attention_Mul')([inputs, x])

    x2=Add(name=prefix+'Chennel_Attention_Add')([
        HSigmoid6(name=prefix+'Chennel_Attention_Ac1')(GlobalAveragePooling2D(keepdims=True, name=prefix+'Chennel_Attention_Pool1')(inputs2)) \
       ,HSigmoid6(name=prefix+'Chennel_Attention_Ac2')(GlobalMaxPooling2D(keepdims=True, name=prefix+'Chennel_Attention_Pool2')(inputs2))])
    
    return Multiply(name=prefix+'Chennel_Attention_Mul')([inputs2, x2])

def _IBN(x, expansion, filters, kernel_size=3, strides=1, dilation_rate=1, activation=ReLU6, attentionMode="_SEBlock", block_id=0, Residual=True, Detour=False, **conf_dict):
    shortcut=x
    infilters=backend.int_shape(x)[-1]

    prefix='IBN{}/'.format(block_id)

    if expansion > 1:
        x=_Conv(x, filters=_depth(infilters*expansion), kernel_size=1, 
                strides=1, activation=activation, prefix=prefix+'Expand', **conf_dict)
        out=x
    
    x=_DeptwiseConv(x, kernel_size=kernel_size, dilation_rate=dilation_rate, activation=activation,
                      strides=strides, prefix=prefix, **conf_dict)

    if attentionMode == '_SEBlock':
        x=_SEBlock(x, 0.25, prefix, activation, **conf_dict)
    elif attentionMode == '_LSCBlock':
        x=_LSCBlock(x, prefix, **conf_dict)
    elif attentionMode == None:
        pass
    else:
        raise ValueError("Not implemented yet")

    x=_Conv(x, filters=filters, kernel_size=1, strides=1, activation=None, prefix=prefix+'Project', **conf_dict)

    
    if tf.math.equal(infilters, filters) and strides==1 and Residual:
        if conf_dict['dropout']:
            x=Dropout(conf_dict['dropout'], noise_shape=(None, 1, 1, 1), name=prefix+'dropout')(x)
        return Add(name=prefix+'Add')([shortcut, x])
    else:
        if Detour:
            return x, out
        else:
            return x

def _Fused(x, expansion, filters, kernel_size=3, strides=1, activation=ReLU6, attentionMode="_SEBlock", block_id=0, Residual=True, **conf_dict):
    """Fused convolution layer."""
    shortcut=x
    infilters=backend.int_shape(x)[-1]
    prefix='FUC{}/'.format(block_id)

    x = _Conv(x,
            filters=_depth(infilters*expansion),
            kernel_size=kernel_size,
            strides=strides,
            activation_fn=activation,
            prefix=prefix+'Conv1',
            **conf_dict)
    out=x

    if attentionMode == '_SEBlock':
        x=_SEBlock(x, 0.25, prefix, activation, **conf_dict)

    x = _Conv(x,
              filters=filters,
              kernel_size=1,
              strides=1,
              activation_fn=tf.identity,
              prefix=prefix+'Conv2',
              **conf_dict)
    if tf.math.equal(infilters, filters) and strides==1 and Residual:
        return x + shortcut
    else:
        return x

def _Tucker(x,
            input_rank_ratio=0.25,
            output_rank_ratio=0.25,
            filters=3,
            kernel_size=3,
            strides=1,
            activation=ReLU6,
            block_id=0,
            Residual=True,
            **conf_dict):
    """Tucker convolution layer (generalized bottleneck)."""

    shortcut=x
    infilters=backend.int_shape(x)[-1]
    prefix='TUC{}/'.format(block_id)


    x = _Conv(x,
            filters=_depth(infilters, input_rank_ratio),
            kernel_size=1,
            strides=1,
            activation_fn=activation,
            prefix=prefix+'Conv1',
            **conf_dict)
    x = _Conv(x,
            _depth(filters, output_rank_ratio),
            kernel_size=kernel_size,
            strides=strides,
            activation_fn=activation,
            prefix=prefix+'Conv2',
            **conf_dict)
    x = _Conv(x,
            filters=filters,
            kernel_size=1,
            strides=1,
            activation_fn=tf.identity,
            prefix=prefix+'Conv3',
            **conf_dict)

    if tf.math.equal(infilters, filters) and strides==1 and Residual:
        x = x + shortcut
    return x

