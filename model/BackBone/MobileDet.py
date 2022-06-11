import tensorflow as tf
from model.customLayer import ReLU6, HSwish6, _depth, _IBN, _Conv, backend

def MobileDetCPU(x, config=None):
    '''
        Reference:
                "MobileDets: Searching for Object Detection Architectures for Mobile Accelerators (IEEE/CVF 2021)"
    '''
    alpha=config["model_config"]["backbone"]["width_multiplier"]
    
    conf_dict = {
        'reg': config["model_config"]["backbone"]["regularization"],
        'trainable':not config["model_config"]["backbone"]["isFreeze"],
        'dropout': config["model_config"]["backbone"]["dropout"]
    }

    x=_Conv(x, filters=_depth(16*alpha), kernel_size=3, strides=2, padding='same', use_bias=False, activation=HSwish6, prefix='Initial', **conf_dict)
    x=_IBN(x, expansion=1, filters=_depth(8*alpha), kernel_size=3, strides=1, activation=ReLU6, block_id=0, **conf_dict)

    x=_IBN(x, expansion=4, filters=_depth(16*alpha), kernel_size=3, strides=2, activation=HSwish6, block_id=1, **conf_dict)
    out1=x

    x=_IBN(x, expansion=8, filters=_depth(32*alpha), kernel_size=3, strides=2, activation=HSwish6, block_id=2, **conf_dict)
    x=_IBN(x, expansion=4, filters=_depth(32*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=3, **conf_dict)
    x=_IBN(x, expansion=4, filters=_depth(32*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=4, **conf_dict)
    x=_IBN(x, expansion=4, filters=_depth(32*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=5, **conf_dict)
    out2=x

    x=_IBN(x, expansion=8, filters=_depth(72*alpha), kernel_size=5, strides=2, activation=HSwish6, block_id=6, **conf_dict)
    x=_IBN(x, expansion=8, filters=_depth(72*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=7, **conf_dict)
    x=_IBN(x, expansion=4, filters=_depth(72*alpha), kernel_size=5, strides=1, activation=HSwish6, block_id=8, **conf_dict)
    x=_IBN(x, expansion=4, filters=_depth(72*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=9, **conf_dict)
    x=_IBN(x, expansion=8, filters=_depth(72*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=10, Residual=False, **conf_dict)
    x=_IBN(x, expansion=8, filters=_depth(72*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=11, **conf_dict)
    x=_IBN(x, expansion=8, filters=_depth(72*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=12, **conf_dict)
    x=_IBN(x, expansion=8, filters=_depth(72*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=13, **conf_dict)
    out3=x

    x=_IBN(x, expansion=8, filters=_depth(104*alpha), kernel_size=5, strides=2, activation=HSwish6, block_id=14, **conf_dict)
    x=_IBN(x, expansion=4, filters=_depth(104*alpha), kernel_size=5, strides=1, activation=HSwish6, block_id=15, **conf_dict)
    x=_IBN(x, expansion=4, filters=_depth(104*alpha), kernel_size=5, strides=1, activation=HSwish6, block_id=16, **conf_dict)
    x=_IBN(x, expansion=4, filters=_depth(104*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=17, **conf_dict)
    x=_IBN(x, expansion=8, filters=_depth(144*alpha), kernel_size=3, strides=1, activation=HSwish6, block_id=18, **conf_dict)
    return out1, out2, out3, x