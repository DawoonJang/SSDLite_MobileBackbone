import tensorflow as tf
from model.customLayer import _Conv, _DeptwiseConv

def _Fblock(inputs, filters, kernel_size, strides, block_id=0, **config_dict):
    prefix = 'Fblock{}/'.format(block_id)
    x = _Conv(inputs, filters = filters//2, kernel_size = 1, strides = 1, prefix=prefix+"Expand", **config_dict)
    x = _DeptwiseConv(x, kernel_size=kernel_size, strides=strides, prefix=prefix, **config_dict)
    x = _Conv(x, filters = filters, kernel_size=1, strides=1, padding='valid', prefix=prefix+"Pointwise", **config_dict)

    return x

def PFH(x, config = None):
    filtersList = config["model_config"]["neck"]["filters"]

    config_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["neck"]["regularization"]),
        'kernel_initializer': tf.initializers.TruncatedNormal(mean=0.0, stddev=0.03),
        'trainable':not config["model_config"]["backbone"]["isFreeze"],
        'use_bias':False
    }

    F1 = x[-2]
    F2 = x[-1]
    F3 = _Fblock(F2, filters = filtersList[2], kernel_size = 3, strides = 2, block_id=3, **config_dict)
    F4 = _Fblock(F3, filters = filtersList[3], kernel_size = 3, strides = 2, block_id=4, **config_dict)
    F5 = _Fblock(F4, filters = filtersList[4], kernel_size = 3, strides = 2, block_id=5, **config_dict)
    F6 = _Fblock(F5, filters = filtersList[5], kernel_size = 3, strides = 2, block_id=6, **config_dict)


    return [F1, F2, F3, F4, F5, F6]