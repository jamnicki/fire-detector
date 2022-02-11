import os
import cv2
import time
from pathlib import Path

from models.initialization import load_rcnn, load_vgg16
from models.prediction import detect_fire, get_danger
from utils import draw_png, draw_bbox
from cfg import (
    OUT_DIR, RCNN_PATH, VGG16_PATH, IN_VIDEO, WRITE_OUT, OUT_CODEC, DANGER_SIGN
)


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    rcnn = load_rcnn(RCNN_PATH)
    vgg16 = load_vgg16(VGG16_PATH)

    cap = cv2.VideoCapture(filename=os.path.abspath(IN_VIDEO))
    if WRITE_OUT:
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        out_filename = "{curr_time}_{in_filename}".format(
            curr_time=time.strftime('%Y%m%d-%H%M%S'),
            in_filename=os.path.basename(IN_VIDEO)
        )
        video_writer = cv2.VideoWriter(
            filename=os.path.join(OUT_DIR, out_filename),
            fourcc=cv2.VideoWriter_fourcc(*OUT_CODEC.upper()),
            fps=cap.get(cv2.CAP_PROP_FPS),
            frameSize=(frame_width, frame_height)
        )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fire_yhat = detect_fire(model=rcnn, image=frame, min_score=0.70)

        for bbox, score in zip(fire_yhat["rois"], fire_yhat["scores"]):
            if not bbox.any():
                continue
            y1, x1, y2, x2 = bbox
            flame_img = frame[y1:y2, x1:x2]

            danger = get_danger(model=vgg16, image=flame_img)
            if danger == 3:
                frame = draw_png(frame, DANGER_SIGN, pos=(10, 10))

            draw_bbox(frame, danger, score, y1, x1, y2, x2)

        cv2.imshow("Fire detector  ['q' - quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if WRITE_OUT:
            video_writer.write(frame)

    cap.release()
    if WRITE_OUT:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
