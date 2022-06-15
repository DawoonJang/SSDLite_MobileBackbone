import tensorflow as tf
import os
import json
import tensorboard

from model.ModelBuilder import ModelBuilder
from utils_train.customLoss import MultiBoxLoss
from utils_train.customCallback import CallbackBuilder
from utils_train.customOptimizer import GCSGD
from utils_train.Datagenerator import DatasetBuilder, DatasetBuilder_temp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

if __name__ =="__main__":
    ######################################## Setting
    tf.config.optimizer.set_jit(True)
    #tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy('mixed_float16'))
    
    ######################################### MODEL
    modelName = "MobileDet_PFH_SSD"
    #modelName = "MobileViT_PFH_SSD"

    model_dir = "checkpoints/"
    modelPart = modelName.split("_")

    with open(os.path.join("model", "0_Config", modelName+".json"), "r") as config_file:
        config = json.load(config_file)
    
    config['modelName'] = modelName
    config['training_config']['num_classes'] = 80

    model = ModelBuilder(config = config)
    #model.load_weights("logs/MobileDet_PFH_SSD/weights/_epoch40_mAP0.143").expect_partial()


    ######################################### DATA
    train_dataset = DatasetBuilder(config, mode = 'train')
    test_dataset = DatasetBuilder(config, mode = 'validation')


    ######################################### Compile
    optimizer = GCSGD(momentum=0.9, nesterov=False)
    #optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)


    #model.summary(expand_nested=True, show_trainable=True)
    model.compile(loss=MultiBoxLoss(config), optimizer=optimizer, weighted_metrics=[])
    model.fit(train_dataset.dataset,
            epochs=config["training_config"]["epochs"],
            initial_epoch=0,
            validation_data=test_dataset.dataset,
            callbacks=CallbackBuilder(config, test_dataset.dataset).get_callbacks()
            )
    