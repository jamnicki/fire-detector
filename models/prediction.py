import cv2
import torch
import skimage
import numpy as np
from mrcnn.model import mold_image
from torchvision import transforms as T

from cfg import ModelConfig


def detect_fire(model, image, min_score=0.85):
    cfg = ModelConfig()
    # image = skimage.io.imread(img_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]

    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = np.expand_dims(scaled_image, 0)

    yhat = model.detect(sample, verbose=0)[0]

    ok_score = np.argwhere(yhat["scores"] >= min_score)
    if ok_score.any():
        ok_score = np.hstack(ok_score)
    else:
        ok_score = ok_score.flatten()
    yhat["rois"] = np.array(yhat["rois"])[ok_score]
    yhat["scores"] = np.array(yhat["scores"])[ok_score]
    return yhat


def get_danger(model, image):
    img_transforms = T.Compose([
        T.ToTensor(),
        T.Resize(size=(64, 64))
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = img_transforms(image)
    image = image[None, ...]

    with torch.no_grad():
        pred = model(image)
        return np.argmax(pred).item()
