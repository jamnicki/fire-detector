import os
from pathlib import Path
from xml.etree import ElementTree
from PIL import Image


def main():
    DATASET_DIR = "/media/jamnicki/HDD/__Datasets/Fire-dataset/renamed_FireNet_wdanger"
    DANGER_DATASET = "/media/jamnicki/HDD/__Datasets/Fire-dataset/fire_danger"

    for stage in os.listdir(DATASET_DIR):
        # test, train, validation
        for danger in range(4):
            danger_path = os.path.join(DANGER_DATASET, stage, str(danger))
            Path(danger_path).mkdir(parents=True, exist_ok=True)

        img_dir = os.path.join(DATASET_DIR, stage, "images")
        ann_dir = os.path.join(DATASET_DIR, stage, "annotations")
        for img, ann in zip(os.listdir(img_dir), os.listdir(ann_dir)):
            img_path = os.path.join(img_dir, img)
            ann_path = os.path.join(ann_dir, ann)
            img_obj = Image.open(img_path)
            img_name, img_format = img.split(".", maxsplit=1)

            ann_tree = ElementTree.parse(ann_path)
            ann_root = ann_tree.getroot()
            for i, object in enumerate(ann_root.findall(".//object")):
                box = object.find("bndbox")
                xmin = int(box.find("xmin").text)
                ymin = int(box.find("ymin").text)
                xmax = int(box.find("xmax").text)
                ymax = int(box.find("ymax").text)
                danger = object.find("danger").text

                bbox_img = img_obj.crop(
                    (xmin, ymin, xmax, ymax)
                )
                bbox_img_path = os.path.join(
                    DANGER_DATASET, stage, danger, f"{img_name}_{i}.{img_format}"
                )
                bbox_img.save(bbox_img_path, format="JPEG")

            img_obj.close()


if __name__ == "__main__":
    main()
