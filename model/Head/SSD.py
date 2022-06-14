import tensorflow as tf
from model.customLayer import ReLU6, _SeparableDepthwiseConv, _SeparableConv

def _SSDLiteClfHead(inputs, filters, kernel_size=3, strides=1, padding="same", block_id=None, **config_dict):
    prefix='ClfHead{}/'.format(block_id)
    x = _SeparableDepthwiseConv(inputs, filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=True, 
                                normalization=None, activation=None, prefix=prefix, **config_dict)
                                

    x=tf.keras.layers.Reshape((-1, config_dict["classNum"]), name=prefix + 'Reshape')(x)
    return x

def _SSDLiteBoxHead(inputs, filters, kernel_size=3, strides=1, padding="same", block_id=None, **config_dict):
    prefix='BoxHead{}/'.format(block_id)
    x = _SeparableDepthwiseConv(inputs, filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=True, 
                                normalization=None, activation=None, prefix=prefix, **config_dict)
    x=tf.keras.layers.Reshape((-1, 4), name=prefix + 'Reshape')(x)
    return x


def SSD(x, config=None):
    num_classes=config["training_config"]["num_classes"]
    numBoxes=config["model_config"]["numAnchors"]

    box_config_dict={
            'reg': config["model_config"]["head"]["regularization"],
            'bias_initializer': tf.constant_initializer(0.0),
            'trainable': not config["model_config"]["head"]["isFreeze"]
            }

    clf_config_dict={
            'reg': config["model_config"]["head"]["regularization"],
            'bias_initializer': tf.constant_initializer(-4.6),
            'classNum':config["training_config"]["num_classes"],
            'trainable': not config["model_config"]["head"]["isFreeze"],
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
            