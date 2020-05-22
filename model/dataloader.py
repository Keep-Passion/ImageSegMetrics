import os, time, random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path
from skimage.measure import label, regionprops
import torch.nn as nn
import scipy.ndimage as ndimage
from skimage import morphology
import torchvision.transforms as tr
from utils import *
from typing import Callable, Iterable, List, Set, Tuple


class WeightMapDataset(Dataset):

    def __init__(self, imgs_dir: Path, data_names: List, use_augment: List = [False, False, False], class_num: int = 2,
                 depth: int = None, crop_size: int = None, norm_transform=None):
        self._img_path = []
        self._label_path = []
        for item in data_names:
            self._img_path.append(Path(imgs_dir, 'images', item))
            self._label_path.append(Path(imgs_dir, 'labels', item))
        self._use_augment = use_augment
        self._depth = depth
        self._crop_size = crop_size
        self._norm_transform = norm_transform
        self._class_num = class_num

        img = load_img(self._img_path[0])
        self._h, self._w = img.shape[:2]

    def __getitem__(self, index):
        # read images
        if self._depth is None:  # 2D analysis
            img = load_img(self._img_path[index])
            label = load_img(self._label_path[index])
            weight = self._bce_weight(label)
        else:  # 3D analysis
            random_depth = random.randint(0, len(self._img_path) - self._depth)
            img = np.zeros((self._h, self._w, self._depth))
            label = np.zeros((self._h, self._w, self._depth))
            weight = np.zeros((self._h, self._w, self._depth, 2))
            for idx, depth_idx in enumerate(range(random_depth, random_depth + self._depth)):
                img[:, :, idx] = load_img(self._img_path[depth_idx])
                label[:, :, idx] = load_img(self._label_path[depth_idx])
                weight[:, :, idx, :] = self._bce_weight(label[:, :, idx])
        # transform
        if self._crop_size is not None:
            img, label, weight = self._rand_crop(img, label, weight, size=self._crop_size)
        if self._use_augment[0]:
            img, label, weight = self._rand_rotation(img, label, weight)
        if self._use_augment[1]:
            img, label, weight = self._rand_vertical_flip(img, label, weight)  # p<0.5, flip
        if self._use_augment[2]:
            img, label, weight = self._rand_horizontal_flip(img, label, weight)  # p<0.5, flip
        if len(self._use_augment) == 4 and self._use_augment[3] and self._depth is not None:
            img, label, weight = self._rand_z_filp(img, label, weight)  # p<0.5, flip
        # to tensor
        if self._depth is None:  # 2D analysis->[C,H,W]
            img = self._norm_transform(img)
            label = torch.from_numpy(label[np.newaxis, :, :])
            weight = torch.from_numpy(weight.transpose((2, 0, 1)))
        else:  # 3D analysis->[C,D,H,W]
            img = np.ascontiguousarray(img, dtype=np.float32)
            label = np.ascontiguousarray(label, dtype=np.float32)
            weight = np.ascontiguousarray(weight, dtype=np.float32)
            img = self._norm_transform(img)
            img = img.unsqueeze(0)
            label = torch.from_numpy(label.transpose((2, 0, 1))[np.newaxis, :, :, :])
            weight = torch.from_numpy(weight.transpose((3, 2, 0, 1)))  # shape(H, W, D, C) -> (C, D, H, W)
        return {'img': img, 'label': label, 'weight': weight}

    def __len__(self):
        """
        Return the length of dataset
        """
        return len(self._img_path)

    def _rand_rotation(self, data: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> Tuple:
        """
        Rand Rotation for 0, 90, 180, 270 degrees
        """
        # 随机选择旋转角度  Randomly select the rotation angle
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            rotate_idx = 0
        elif angle == 90:
            rotate_idx = 1
        elif angle == 180:
            rotate_idx = 2
        else:
            rotate_idx = 3
        data = np.rot90(data, rotate_idx).copy()
        mask = np.rot90(mask, rotate_idx).copy()
        weight = np.rot90(weight, rotate_idx).copy()
        return data, mask, weight

    def _rand_vertical_flip(self, data: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> Tuple:
        """
        Rand vertical flip by 0.5 threshold
        """
        p = random.random()
        if p < 0.5:
            data = np.flipud(data).copy()
            mask = np.flipud(mask).copy()
            weight = np.flipud(weight).copy()
        return data, mask, weight

    def _rand_horizontal_flip(self, data: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> Tuple:
        """
        Rand horizontal flip by 0.5 threshold
        """
        p = random.random()
        if p < 0.5:
            data = np.fliplr(data).copy()
            mask = np.fliplr(mask).copy()
            weight = np.fliplr(weight).copy()
        return data, mask, weight

    def _rand_z_filp(self, data: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> Tuple:
        """
        Random z flip by 0.5 threshold
        """
        p = random.random()
        if p < 0.5:
            data = np.flip(data, 2)
            mask = np.flip(mask, 2)
            weight = np.flip(weight, 2)
        return data, mask, weight

    def _rand_crop(self, data: np.ndarray, mask: np.ndarray, weight: np.ndarray, size: int = 512) -> Tuple:
        """
        Rand crop with certain size
        """
        # 随机选择裁剪区域   Randomly select the crop area
        random_h = random.randint(0, data.shape[0] - size)
        random_w = random.randint(0, data.shape[1] - size)

        data = data[random_h: random_h + size, random_w: random_w + size]
        mask = mask[random_h: random_h + size, random_w: random_w + size]
        weight = weight[random_h: random_h + size, random_w: random_w + size]
        return data, mask, weight

    def _bce_weight(self, label: np.ndarray) -> np.ndarray:
        """
        Balace cross entropy with corresponding weight map
        """
        weight = np.zeros((self._h, self._w, self._class_num))

        class_weight = np.zeros((self._class_num, 1))
        for idx in range(self._class_num):
            idx_num = np.count_nonzero(label == idx)
            class_weight[idx, 0] = idx_num
        min_num = np.amin(class_weight)
        class_weight = class_weight * 1.0 / min_num
        class_weight = np.sum(class_weight) - class_weight
        for idx in range(self._class_num):
            weight[:, :, idx][label == idx] = class_weight[idx, 0]
        return weight


class WeightMapLoss(nn.Module):
    """
    calculate weighted loss with weight maps
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight_maps: torch.Tensor,
                eps: float = 1e-20) -> torch.Tensor:
        """
        target: The target map, LongTensor, unique(target) = [0 1]
        weight_maps: The weights for two channels，weight_maps = [weight_bck_map, weight_obj_map]
        method：Select the type of loss function
        """
        class_num = weight_maps.size()[1]
        mask = target.float()
        logit = torch.softmax(pred, dim=1)
        loss = 0
        weight_maps = weight_maps.float()
        for idx in range(class_num):
            if weight_maps.dim() == 4:
                loss += -1 * weight_maps[:, idx, :, :] * (torch.log(logit[:, idx, :, :]) + eps)
            elif weight_maps.dim() == 5:
                loss += -1 * weight_maps[:, idx, :, :, :] * (torch.log(logit[:, idx, :, :, :]) + eps)
        # loss = -1 * weight_maps * (torch.log(logit) + eps)
        return loss.sum() / weight_maps.sum()