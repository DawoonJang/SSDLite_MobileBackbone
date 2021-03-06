import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import tensorflow as tf
import json
import tensorboard

from absl import app, logging, flags

from model.ModelBuilder import ModelBuilder
from utils_train.customLoss import MultiBoxLoss
from utils_train.customCallback import CallbackBuilder
from utils_train.customOptimizer import GCSGD
from utils_train.Datagenerator import Dataset_COCO, Dataset_Pascal, Dataset_COCO_Temp


flags.DEFINE_boolean(
    name='fp16',
    default=True,
    help='Mixed Precision')

flags.DEFINE_string(
    name='dataset',
    default='coco',
    help='Dataset to train')

flags.DEFINE_string(
    name='model',
    default='MobileNetV3_PFH_SSD',
    help='Model to train')

FLAGS = flags.FLAGS

def main(_argv):
    tf.config.optimizer.set_jit("autoclustering")
    tf.random.set_seed(22)
    logging.set_verbosity(logging.WARNING)
    strategy = tf.distribute.MirroredStrategy()

    logging.warning('Training model: {}'.format(FLAGS.model))
    modelName = FLAGS.model
    
    with open(os.path.join("model/0_Config", modelName+".json"), "r") as config_file:
        config = json.load(config_file)
    
    logging.warning('Training dataset: {}'.format(FLAGS.dataset))
    if FLAGS.dataset == 'pascal':
        train_dataset = Dataset_Pascal(config, mode = 'train')
        test_dataset = Dataset_Pascal(config, mode = 'validation')
        config['training_config']['num_classes'] = 20
        val_file = "data/pascal_test2007.json"
    elif FLAGS.dataset == 'coco':
        train_dataset = Dataset_COCO(config, mode = 'train')
        test_dataset = Dataset_COCO(config, mode = 'validation')
        config['training_config']['num_classes'] = 80
        val_file = "data/coco_val2017.json"

    optimizer = GCSGD(momentum=0.9, nesterov=False)
    if FLAGS.fp16:
        logging.warning('Training Precision: FP16')
        tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy('mixed_float16'))
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    else:
        logging.warning('Training Precision: FP32')

    ######################################### Compile
    config['modelName'] = modelName
    with strategy.scope():
        model = ModelBuilder(config = config)
        #model.load_weights("logs/_epoch373_mAP0.201").expect_partial()
        model.compile(loss=MultiBoxLoss(config), optimizer=optimizer, weighted_metrics=[])
        print(model)

    model.fit(train_dataset.dataset,
            epochs=config["training_config"]["epochs"],
            #steps_per_epoch = len(train_dataset),
            initial_epoch=0,
            validation_data=test_dataset.dataset,
            callbacks=CallbackBuilder(config, test_dataset.dataset, val_file).get_callbacks()
            )

if __name__ =="__main__":
    app.run(main)
    
    