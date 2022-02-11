import torch
from mrcnn.model import MaskRCNN

from cfg import ModelConfig


def load_rcnn(path):
    rcnn_cfg = ModelConfig()
    rcnn = MaskRCNN(mode="inference", model_dir="./", config=rcnn_cfg)
    rcnn.load_weights(path, by_name=True)
    return rcnn


def load_vgg16(path):
    vgg16 = torch.load(path)
    vgg16.eval()
    return vgg16
