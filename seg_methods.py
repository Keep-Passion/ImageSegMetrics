import cv2
import numpy as np
# 所有能找到的 自动化 方法, 交互式方法列出来，需要交互，弹出交互框
# 所有结果的像素值为[0,C]

def OTSU(in_img: np.ndarray) -> np.ndarray:
    """
    最大类间方差法， OTSU
    """
    thresh, out_img = cv2.threshold(in_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    out_img = out_img / 255
    return out_img

# TODO:其他图像分割方法