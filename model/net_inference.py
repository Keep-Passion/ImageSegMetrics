import os
import cv2
import random
import math

import numpy as np
from skimage import io
from skimage import morphology

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr

from typing import Callable, Iterable, List, Set, Tuple
from model.unet import UNet


class NetInference():
    """
    Net Inference for 2D image segmentation
    """

    def __init__(self, mean: float = 0.9410404628082503, std: float = 0.12481161024777744,
                 model=UNet, pth_address: str = os.path.join(os.getcwd(), "model", "unet_parameter.pth"),
                 use_post_process: bool = True):

        self._mean = mean
        self._std = std
        self._z_score_norm = tr.Compose([
            tr.ToTensor(),
            tr.Normalize(mean=[self._mean],
                         std=[self._std])
        ])
        self._use_post_process = use_post_process

        self._model = model()
        if torch.cuda.is_available():
            self._model = nn.DataParallel(self._model).cuda()

        # model parameters setting
        self._pth_address = pth_address
        self._load_pth()

    def _forward_one_image(self, in_img: np.ndarray) -> np.ndarray:
        """
        Forward one image, return segmentation result, some errers will be raised
        """
        self._model.eval()
        with torch.no_grad():
            h, w = in_img.shape
            in_tensor = self._z_score_norm(in_img).unsqueeze(0)
            if torch.cuda.is_available():
                in_tensor = in_tensor.cuda()
            out_tensor = self._model.forward(in_tensor)  # b,c,w,h c=2
            out_tensor = F.softmax(out_tensor, dim=1)
            if torch.cuda.is_available():
                out_tensor = out_tensor.cpu()
            torch.cuda.empty_cache()
            out_img = out_tensor.squeeze().numpy().transpose((1, 2, 0))
        out_img = np.argmax(out_img, axis=2)
        if self._use_post_process:
            out_img = self._post_process(out_img).astype(np.uint8)
        return out_img

    def _post_process(self, in_img: np.ndarray) -> np.ndarray:
        """
        Post processing method after network inference.
        """
        # skeletonization of result
        out_img = morphology.skeletonize(1 - in_img, method="lee")  # Lee Skeleton method
        # dialiation with 3 pixels
        out_img = 1 - morphology.dilation(out_img, morphology.square(3))
        # Todo: delete the noise region whose area smaller than threshold
        # Add this depend on the user's decision
        return out_img

    def _load_pth(self):
        """
        Load model parameters
        """
        if torch.cuda.is_available():  # load gpu parameters for gpu
            self._model.load_state_dict(torch.load(self._pth_address))
        else:  # load gpu parameters for cpu
            self._model.load_state_dict(
                {k.replace('module.', ''): v for k, v in torch.load(self._pth_address, map_location='cpu').items()})
            # self._model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(self._pth_address).items()})
