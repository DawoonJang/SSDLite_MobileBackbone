import tensorflow as tf
from model.customLayer import _DeptwiseConv, _Conv

def _SSDLiteClfHead(inputs, filters, kernel_size=3, strides=1, padding="same", block_id=None, **config_dict):
    prefix='ClfHead{}/'.format(block_id)
    classNum = config_dict["classNum"]
    config_dict.pop('classNum')

    x = _DeptwiseConv(inputs, kernel_size=kernel_size, strides=strides, padding=padding, prefix=prefix, **config_dict)
    x = _Conv(x, filters = filters, kernel_size=1, strides=1, padding='valid', prefix=prefix, normalization=None, activation=None, **config_dict)

    x=tf.keras.layers.Reshape((-1, classNum), name=prefix + 'Reshape')(x)
    return x

def _SSDLiteBoxHead(inputs, filters, kernel_size=3, strides=1, padding="same", block_id=None, **config_dict):
    prefix='BoxHead{}/'.format(block_id)
    x = _DeptwiseConv(inputs, kernel_size=kernel_size, strides=strides, padding=padding, prefix=prefix, **config_dict)
    x = _Conv(x, filters = filters, kernel_size=1, strides=1, padding='valid', prefix=prefix, normalization=None, activation=None, **config_dict)

    x=tf.keras.layers.Reshape((-1, 4), name=prefix + 'Reshape')(x)
    return x


def SSD(x, config=None):
    num_classes=config["training_config"]["num_classes"]
    numBoxes=config["model_config"]["numAnchors"]

    box_config_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["head"]["regularization"]),
        'kernel_initializer': tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
        #'bias_initializer': tf.constant_initializer(0.0),
        'trainable':not config["model_config"]["head"]["isFreeze"],
        'use_bias':True
    }

    clf_config_dict = {
        'kernel_regularizer': tf.keras.regularizers.l2(config["model_config"]["head"]["regularization"]),
        'kernel_initializer': tf.initializers.RandomNormal(mean=0.0, stddev=0.03),
        'bias_initializer': tf.constant_initializer(-4.6),
        'trainable':not config["model_config"]["head"]["isFreeze"],
        'use_bias':True,
        'classNum':config["training_config"]["num_classes"],
    }

    box_outputs=[]
    clf_outputs=[]
    for idx, x_single in enumerate(x):
        clf_outputs.append(_SSDLiteClfHead(x_single, 
                                        filters=numBoxes[idx]*num_classes, 
                                        kernel_size=3,
                                        strides=1, 
                                        padding="same", 
                                        block_id=idx, 
                                        **clf_config_dict))
        box_outputs.append(_SSDLiteBoxHead(x_single, 
                                        filters=numBoxes[idx]*4,           
                                        kernel_size=3, 
                                        strides=1, 
                                        padding="same", 
                                        block_id=idx, 
                                        **box_config_dict))
        

    box_outputs = tf.keras.layers.Concatenate(axis=-2, name="BoxConcat")(box_outputs)
    clf_outputs = tf.keras.layers.Concatenate(axis=-2, name="ClfConcat")(clf_outputs)
    output = tf.keras.layers.Concatenate(axis=-1, name="FinalConcat")([box_outputs, clf_outputs])
    return tf.keras.layers.Activation('linear', dtype='float32', name="output_layer")(output)
            