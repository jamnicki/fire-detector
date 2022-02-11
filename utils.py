import cv2
import numpy as np
from PIL import Image


def draw_bbox(frame, danger, score, y1, x1, y2, x2):
    label = f"Danger: {danger}  ({score:.2f})"
    color = (255, 255, 255)
    if danger == 1:
        color = (0, 255, 255)
    elif danger == 2:
        color = (0, 165, 255)
    elif danger == 3:
        color = (0, 0, 255)
    cv2.rectangle(
        img=frame,
        pt1=(x1, y1),
        pt2=(x2, y2),
        color=color,
        thickness=2
    )
    cv2.putText(
        img=frame,
        text=label,
        org=(x1, y1 - 6),
        fontFace=0,
        fontScale=6e-4 * frame.shape[1],
        color=color,
        thickness=1
    )


def draw_png(frame, png_path, pos=(0, 0)):
    png = Image.open(png_path).convert("RGBA")
    R,G,B,A = png.split()
    png = Image.merge("RGBA", (B, G, R, A))
    pilim = Image.fromarray(frame)
    pilim.paste(png, box=pos, mask=png)
    return np.array(pilim)
