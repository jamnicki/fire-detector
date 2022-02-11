from mrcnn.config import Config
from tensorflow.compat.v1 import logging as tf_logging
tf_logging.set_verbosity(tf_logging.ERROR)


RCNN_PATH = "./models/1000wnofire_30x50__fire_mask_rcnn_trained.h5"
VGG16_PATH = "./models/vgg16_5e_fire_danger.pt"

IN_VIDEO = "./videos/test_vid.mp4"

OUT_DIR = "./videos"
OUT_CODEC = "MP4V"
WRITE_OUT = True

DANGER_SIGN = "./static/icons/fire-call.png"


class ModelConfig(Config):
    EPOCHS = 1
    STEPS_PER_EPOCH = 5
    LEARNING_RATE = 0.001

    NAME = f"{EPOCHS}x{STEPS_PER_EPOCH}__fire_cfg"

    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
