import tensorflow as tf
from model.customLayer import ReLU6, _Conv, _SeparableDepthwiseConv, WeightedSum, _IBN

def FPN(inputs, config=None):
    config_dict = {
        'reg': config["model_config"]["neck"]["regularization"],
        'trainable': not config["model_config"]["neck"]["isFreeze"]
        }

    P5 = _Conv(inputs[-1],  #10
            filters=config["model_config"]["neck"]["filters"], 
            kernel_size=1, 
            strides=1, 
            padding='same', 
            use_bias=False,
            activation=None,
            prefix='FPN_P5/',
            **config_dict)
    P5_up = tf.keras.layers.UpSampling2D(name="FPN_P5/Up")(P5)
    P5 = _SeparableDepthwiseConv(P5,
        filters=config["model_config"]["neck"]["filters"],
        kernel_size=3,
        strides=1, 
        padding='same', 
        use_bias=False,
        prefix='FPN_P5_2/',
        **config_dict
    )

    P4 = _Conv(inputs[-2], #20
            filters=config["model_config"]["neck"]["filters"], 
            kernel_size=1, 
            strides=1, 
            padding='same', 
            use_bias=False,
            prefix='FPN_P4/',
            **config_dict)
    P4 = tf.keras.layers.Add(name="FPN_P4/Add")([P4, P5_up])
    P4_up = tf.keras.layers.UpSampling2D(name="FPN_P4/Up")(P4)
    P4 = _SeparableDepthwiseConv(P4,
        filters=config["model_config"]["neck"]["filters"],
        kernel_size=3,
        strides=1, 
        padding='same', 
        use_bias=False,
        activation=None,
        prefix='FPN_4_2/',
        **config_dict
    )

    P3 = _Conv(inputs[-3], #40
            filters=config["model_config"]["neck"]["filters"], 
            kernel_size=1, 
            strides=1, 
            padding='same', 
            use_bias=False,
            activation=None,
            prefix='FPN_P3/',
            **config_dict)
    P3 = tf.keras.layers.Add(name="FPN_P3/Add")([P3, P4_up])
    P3 = _SeparableDepthwiseConv(P3,
        filters=config["model_config"]["neck"]["filters"],
        kernel_size=3,
        strides=1, 
        padding='same', 
        use_bias=False,
        prefix='FPN_3_2/',
        **config_dict
    )

    P6 = _SeparableDepthwiseConv(inputs[-1],  #5
            filters=config["model_config"]["neck"]["filters"], 
            kernel_size=3, 
            strides=2, 
            padding='same', 
            use_bias=False,
            prefix='FPN_P6/',
            **config_dict)

    P7 = _SeparableDepthwiseConv(P6,  #3
            filters=config["model_config"]["neck"]["filters"], 
            kernel_size=3, 
            strides=2, 
            padding='same', 
            use_bias=False,
            prefix='FPN_P7/',
            **config_dict)
            
    return [P3, P4, P5, P6, P7]