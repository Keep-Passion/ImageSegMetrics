import cv2
import numpy as np
from skimage.feature import canny
from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker, active_contour
from skimage.future import graph
from skimage import segmentation
from model.net_inference import NetInference


# 所有结果的像素值为[0,C] all result is in integer of [0, C]
def OTSU(in_img: np.ndarray) -> np.ndarray:
    """
    最大类间方差法， OTSU
    """
    thresh, out_img = cv2.threshold(in_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out_img = out_img / 255
    return out_img


def Canny(in_img: np.ndarray, threshold1=420, threshold2=430) -> np.ndarray:
    """
    canny
    """
    # out_img = canny(in_img,sigma=0.001)
    out_img = cv2.Canny(in_img, threshold1, threshold2)
    out_img = out_img / 255
    out_img = 1 - out_img
    return out_img


def RandomWalker(in_img: np.ndarray) -> np.ndarray:
    """
    随机游走，Run random walker algorithm
    """
    markers = np.zeros(in_img.shape, dtype=np.uint)
    markers[in_img < 200] = 1
    markers[in_img > 200] = 2

    out_img = random_walker(in_img, markers)
    out_img = out_img / 2
    out_img[out_img == 0.5] = 0
    return out_img


def Kmeans(in_img: np.ndarray) -> np.ndarray:
    """
    K均值聚类，k-means
    """
    in_img_flat = in_img.reshape((in_img.shape[0] * in_img.shape[1], 1))
    in_img_flat = np.float32(in_img_flat)
    K = 2
    bestLabels = None
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, bestLabels, centers = cv2.kmeans(in_img_flat, K, bestLabels, criteria, attempts, flags)

    out_img = bestLabels.reshape((in_img.shape[0], in_img.shape[1]))
    m = np.max(out_img)
    out_img = out_img / m
    out_img[out_img == 0.5] = 0
    return out_img


def Watershed(in_img: np.ndarray) -> np.ndarray:
    """
    分水岭方法，watershed
    """
    markers = np.zeros(in_img.shape, dtype=np.uint)
    markers[in_img < 200] = 1
    markers[in_img > 200] = 2
    out_img = watershed(in_img, markers)
    m = np.max(out_img)
    out_img = out_img / float(m)
    out_img[out_img == 0.5] = 0
    return out_img


net_inference = NetInference()


def unet(in_img: np.ndarray) -> np.ndarray:
    """
    Deep learning, unet
    """
    out_img = net_inference._forward_one_image(in_img)
    return out_img
