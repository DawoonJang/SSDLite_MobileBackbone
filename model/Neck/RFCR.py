import tensorflow as tf
from model.customLayer import ReLU6, _Conv, _SeparableDepthwiseConv, WeightedSum, _IBN

def _Fblock(inputs, filters, kernel_size, strides, block_id=0, **config_dict):
    prefix = 'Fblock{}/'.format(block_id)
    x=_Conv(inputs, filters = filters//2, kernel_size = 1, strides = 1,
            prefix=prefix+"Expand", **config_dict)
    x=_SeparableDepthwiseConv(x, filters = filters, kernel_size = kernel_size, strides = strides, 
                            prefix=prefix, **config_dict)

    return x

def RFCR(inputs, config=None):
    config_dict = {
        'reg': config["model_config"]["neck"]["regularization"],
        'trainable': not config["model_config"]["neck"]["isFreeze"]
        }

    b1c = tf.keras.layers.MaxPooling2D()(tf.keras.layers.MaxPooling2D()(inputs[0])) #80
    b2c = tf.keras.layers.MaxPooling2D()(inputs[1]) #40
    b3c = inputs[2] #20
    b4c = tf.keras.layers.UpSampling2D()(inputs[3]) #10

    b1c = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', use_bias=False, name="RFCR/Conv1")(b1c)
    b2c = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', use_bias=False, name="RFCR/Conv2")(b2c)
    b3c = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', use_bias=False, name="RFCR/Conv3")(b3c)
    b4c = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', use_bias=False, name="RFCR/Conv4")(b4c)

    bc = WeightedSum()([b1c, b2c, b3c, b4c])

    bc=_SeparableDepthwiseConv(bc, filters=96, kernel_size=5, strides=1, use_bias=False, prefix="RFCR/", **config_dict)
    #bc=_IBN(bc, expansion=1, filters=96, kernel_size=5, stride=1, block_id='RFCR', Residual=False, **config_dict)
    
    F1=tf.keras.layers.Concatenate()([inputs[-3], tf.keras.layers.UpSampling2D()(bc)]) #40
    F2=tf.keras.layers.Concatenate()([inputs[-2], bc]) #20
    F3=tf.keras.layers.Concatenate()([inputs[-1], tf.keras.layers.MaxPooling2D()(bc)]) #10

    return [F1, F2, F3]