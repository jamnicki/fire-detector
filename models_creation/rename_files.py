import os
from pathlib import Path


def main():
    FIRENET_DIR = "/home/jamnicki/Documents/test/FireNet"
    NEW_FIRENET_DIR = "./new_FireNet"

    image_id = 0
    for dir in os.listdir(FIRENET_DIR):
        img_dir = os.path.join(FIRENET_DIR, dir, "images")
        ann_dir = os.path.join(FIRENET_DIR, dir, "annotations")
        new_img_dir = os.path.join(NEW_FIRENET_DIR, dir, "images")
        new_ann_dir = os.path.join(NEW_FIRENET_DIR, dir, "annotations")
        Path(new_img_dir).mkdir(parents=True, exist_ok=True)
        Path(new_ann_dir).mkdir(parents=True, exist_ok=True)
        for i, (img, ann) in enumerate(zip(sorted(os.listdir(img_dir)),
                                           sorted(os.listdir(ann_dir)))):
            if img.split(".")[0] != ann.split(".")[0]:
                print(img.split(".")[0])
                print(ann.split(".")[0])
                continue
            img_path = os.path.join(img_dir, img)
            ann_path = os.path.join(ann_dir, ann)

            with open(img_path, "rb") as img_f:
                img_b = img_f.read()
            with open(ann_path, "rb") as ann_f:
                ann_b = ann_f.read()

            img_ext = img.split(".")[-1]
            ann_ext = ann.split(".")[-1]

            new_img_filename = f"{image_id}.{img_ext}"
            new_img_path = os.path.join(new_img_dir, new_img_filename)

            new_ann_filename = f"{image_id}.{ann_ext}"
            new_ann_path = os.path.join(new_ann_dir, new_ann_filename)

            with open(new_img_path, "wb") as new_img_f:
                new_img_f.write(img_b)
            with open(new_ann_path, "wb") as new_ann_f:
                new_ann_f.write(ann_b)

            image_id += 1


if __name__ == "__main__":
    main()
