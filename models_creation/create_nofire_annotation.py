import os
import cv2


def main():
    NOFIRE_DIR = "/media/jamnicki/HDD/__Datasets/Fire-dataset/no_fire"
    for stage in os.listdir(NOFIRE_DIR):
        images_dir = os.path.join(NOFIRE_DIR, stage, "images")
        annotations_dir = os.path.join(NOFIRE_DIR, stage, "annotations")
        for img_filename in os.listdir(images_dir):
            img_path = os.path.join(images_dir, img_filename)
            im_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            height, width = im_gray.shape
            ann_content = "<annotation>"
            ann_content += "\n\t<size>"
            ann_content += f"\n\t\t<width>{width}</width>"
            ann_content += f"\n\t\t<height>{height}</height>"
            ann_content += "\n\t</size>"
            ann_content += "\n</annotation>"

            img_id = img_filename.split(".", maxsplit=1)[0]
            ann_path = os.path.join(annotations_dir, img_id+".xml")
            with open(ann_path, "wb") as ann_f:
                ann_f.write(ann_content.encode("utf-8"))


if __name__ == "__main__":
    main()
