import numpy as np
from skimage import io


def load_img(img_path: str) -> np.ndarray:
    """
        加载图像，Load images
    """
    img = io.imread(img_path)
    if np.amax(img) == 255 and len(np.unique(img)) == 2:
        img = img * 1.0 / 255
    return img.astype(np.uint8)
