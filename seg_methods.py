import cv2
import numpy as np


def OTSU(in_img: np.ndarray) -> np.ndarray:
    """
    最大类间方差法， OTSU
    """
    thresh, out_img = cv2.threshold(in_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    out_img = out_img / 255
    return out_img
