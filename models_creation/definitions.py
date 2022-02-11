import re
from os import listdir, path
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import mold_image
from mrcnn.utils import Dataset


class FireDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "fire")
        # define data locations
        images_dir = dataset_dir + "/images/"
        annotations_dir = dataset_dir + "/annotations/"
        # find all images
        for filename in listdir(images_dir):
            image_id = int(filename.split(".", maxsplit=1)[0])
            img_path = images_dir + filename
            ann_path = annotations_dir + str(image_id) + ".xml"
            # add to dataset
            self.add_image(
                "dataset",
                image_id=image_id,
                path=img_path,
                annotation=ann_path
            )
    
    def load_nofire_dataset(self, dataset_dir, is_train=True):
        self.add_class("nofire_dataset", 0, "nofire")
        images_dir = dataset_dir + "/images/"
        annotations_dir = dataset_dir + "/annotations/"
        for filename in listdir(images_dir):
            image_id = int(filename.split(".", maxsplit=1)[0])
            img_path = path.join(images_dir, filename)
            ann_path = annotations_dir + str(image_id) + ".xml"
            # add to dataset
            self.add_image(
                "nofire_dataset",
                image_id=image_id+1000,
                path=img_path,
                annotation=ann_path
            )

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall(".//bndbox"):
            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find(".//size/width").text)
        height = int(root.find(".//size/height").text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info["annotation"]
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype="uint8")
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index("fire"))
        return masks, asarray(class_ids, dtype="int32")

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]


class ModelConfig(Config):
    # training config
    EPOCHS = 1
    STEPS_PER_EPOCH = 5
    LEARNING_RATE = 0.001

    # define the name of the configuration
    NAME = f"{EPOCHS}x{STEPS_PER_EPOCH}__fire_cfg"

    # number of classes (background + fire)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, figname, plots_dir=".",
                             n_images=5, save=False, min_score=0.8):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        plt.subplot(n_images, 2, i*2+1)
        # plot raw pixel data
        plt.imshow(image)
        plt.title("Actual")
        # plot masks
        for j in range(mask.shape[2]):
            plt.imshow(mask[:, :, j], cmap="gray", alpha=0.3)
        # get the context for drawing boxes
        plt.subplot(n_images, 2, i*2+2)
        # plot raw pixel data
        plt.imshow(image)
        plt.title("Predicted")
        ax = plt.gca()
        # plot each box
        for box, score in zip(yhat["rois"], yhat["scores"]):
            if score <= min_score:
                continue
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color="red")
            # draw the box
            ax.add_patch(rect)
    # show the figure
    fig = ax.get_figure()
    fig.set_size_inches(20, 4*n_images)
    if save:
        fig.savefig(path.join(plots_dir, figname+".png"), facecolor='white')
    plt.show()
