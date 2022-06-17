import tensorflow as tf
import os
import json
import tensorboard

from absl import app, logging, flags

from model.ModelBuilder import ModelBuilder
from utils_train.customLoss import MultiBoxLoss
from utils_train.customCallback import CallbackBuilder
from utils_train.customOptimizer import GCSGD
from utils_train.Datagenerator import DatasetBuilder_COCO, DatasetBuilder_Pascal, DatasetBuilder_COCO_Temp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

flags.DEFINE_boolean(
    name='fp16',
    default=False,
    help='Mixed Precision')

flags.DEFINE_string(
    name='dataset',
    default='coco',
    help='Dataset to train')

flags.DEFINE_string(
    name='model',
    default='MobileNetV3',
    help='Model to train')

FLAGS = flags.FLAGS

def main(_argv):
    tf.config.optimizer.set_jit(True)
    
    if FLAGS.fp16:
        logging.info('Training Precision: FP16')
        tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy('mixed_float16'))
    else:
        logging.info('Training Precision: FP32')
    
    logging.info('Training model: {}'.format(FLAGS.model))
    if FLAGS.model == 'MobileNetV3':
        modelName = "MobileNetV3_PFH_SSD"
    elif FLAGS.model == 'MobileDet':
        modelName = "MobileDet_PFH_SSD"
    
    with open(os.path.join("model/0_Config", modelName+".json"), "r") as config_file:
        config = json.load(config_file)
    config['modelName'] = modelName
    model = ModelBuilder(config = config)
    #model.load_weights("logs/MobileDet_PFH_SSD/weights/_epoch40_mAP0.143").expect_partial()

    logging.info('Training dataset: {}'.format(FLAGS.dataset))
    if FLAGS.dataset == 'pascal':
        train_dataset = DatasetBuilder_Pascal(config, mode = 'train')
        test_dataset = DatasetBuilder_Pascal(config, mode = 'validation')
        config['training_config']['num_classes'] = 20
        val_file = "data/pascal_test2007.json"
    elif FLAGS.dataset == 'coco':
        train_dataset = DatasetBuilder_COCO(config, mode = 'train')
        test_dataset = DatasetBuilder_COCO(config, mode = 'validation')
        config['training_config']['num_classes'] = 80
        val_file = "data/coco_val2017.json"

    ######################################### Compile
    optimizer = GCSGD(momentum=0.9, nesterov=False)
    #optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)


    #model.summary(expand_nested=True, show_trainable=True)
    model.compile(loss=MultiBoxLoss(config), optimizer=optimizer, weighted_metrics=[])
    model.fit(train_dataset.dataset,
            epochs=config["training_config"]["epochs"],
            #steps_per_epoch = len(train_dataset),
            initial_epoch=0,
            validation_data=test_dataset.dataset,
            callbacks=CallbackBuilder(config, test_dataset.dataset, val_file).get_callbacks()
            )

if __name__ =="__main__":
    app.run(main)
    
    