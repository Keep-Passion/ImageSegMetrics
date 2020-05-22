import os
import numpy as np
from skimage import io
import os
import cv2
import random
import numpy as np
import torch
from pathlib import Path
from skimage import io
from torch import Tensor
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
from typing import Callable, Iterable, List, Set, Tuple


def load_img(img_path: Path) -> np.ndarray:
    """
        加载图像，Load images
    """
    img = io.imread(str(img_path))
    if np.amax(img) == 255 and len(np.unique(img)) == 2:
        img = img * 1.0 / 255
    return img.astype(np.uint8)


def create_folder(folder_dir: Path):
    """
        创建目录， create folder if there is no folder
    """
    if not os.path.exists(str(folder_dir)):
        os.mkdir(str(folder_dir))


def setup_seed(seed: int = 2020) -> None:
    """
        设置种子点，Set random seed to make experiments repeatable
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # implement same config in cpu and gpu
    torch.backends.cudnn.benchmark = True


def count_mean_and_std(img_dir: Path) -> Tuple:  # (mean, std, num)
    """
        计算原始图像中的均值和方差，Calculate mean and std of data
    """
    assert img_dir.is_dir(), "The input is not a dir"
    mean, std, num = 0, 0, 0

    imgs_path = img_dir.glob("*.png")

    for img_path in imgs_path:
        num += 1
        img = cv2.imread(str(img_path), 0) / 255
        assert np.max(np.unique(img)) <= 1, "The img value should lower than 1 when calculate mean and std"
        mean += np.mean(img)
        std += np.std(img)
    mean /= num
    std /= num
    return mean, std, num


class Printer():
    """
    输出类，Control the printing to screen and txt
    """

    def __init__(self, is_out_log_file=True, file_address=None):
        self.is_out_log_file = is_out_log_file
        self.file_address = file_address

    def print_and_log(self, content):
        print(content)
        if self.is_out_log_file:
            f = open(os.path.join(self.file_address), "a")
            f.write(content)
            f.write("\n")
            f.close()


def check_result(epoch, img, label, output, weight, checkpoint_path, printer,
                 description="val_"):  # check the output of dataset
    """
    检查训练过程中的数据，Check training result
    """
    print(img.shape, label.shape, output.shape, weight.shape)
    printer.print_and_log("Image size is {},    min is {}, max is {}".format(img.shape, np.amin(img), np.amax(img)))
    printer.print_and_log(
        "Label size is {},    min is {}, max is {}".format(label.shape, np.amin(label), np.amax(label)))
    printer.print_and_log(
        "Output size is {},   min is {}, max is {}".format(output.shape, np.amin(output), np.amax(output)))
    printer.print_and_log(
        "Weight-0 size is {}, min is {}, max is {}".format(weight[:, :, 0].shape, np.amin(weight[:, :, 0]),
                                                           np.amax(weight[:, :, 0])))
    printer.print_and_log(
        "Weight-1 size is {}, min is {}, max is {}".format(weight[:, :, 1].shape, np.amin(weight[:, :, 1]),
                                                           np.amax(weight[:, :, 1])))
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 5, 1), plt.imshow(img), plt.title('img'), plt.axis("off")
    plt.subplot(1, 5, 2), plt.imshow(label, cmap="gray"), plt.title('label'), plt.axis("off")
    plt.subplot(1, 5, 3), plt.imshow(output, cmap="gray"), plt.title('output'), plt.axis("off")
    plt.subplot(1, 5, 4), plt.imshow(weight[:, :, 0], cmap="plasma"), plt.title('weight-0'), plt.axis("off")
    plt.subplot(1, 5, 5), plt.imshow(weight[:, :, 1], cmap="plasma"), plt.title('weight-1'), plt.axis("off")
    plt.savefig(str(Path(checkpoint_path, description + str(epoch).zfill(3) + '_result.png')))
    plt.show()


def plot(epoch, train_value_list, val_value_list, checkpoint_path, find_min_value=True, curve_name='loss'):
    """
    打印训练和测试曲线，Plot the curve of train and validation
    """
    clear_output(True)
    plt.figure()
    target_value = 0
    target_func = 'None'
    if find_min_value and len(val_value_list) > 10:
        target_value = min(val_value_list[10:])
        target_func = 'min'
    elif find_min_value is False and len(val_value_list) > 10:
        target_value = max(val_value_list[10:])
        target_func = 'max'
    title_name = 'Epoch {}. train ' + curve_name + ': {:.4f}. val ' + curve_name + ': {:.4f}. ' + ' val_' + target_func + ' ' + curve_name + ': {:.4f}. '
    plt.title(title_name.format(epoch, train_value_list[-1], val_value_list[-1], target_value))
    plt.plot(train_value_list, color="r", label="train " + curve_name)
    plt.plot(val_value_list, color="b", label="val " + curve_name)
    if len(val_value_list) > 10:
        plt.axvline(x=val_value_list.index(target_value), ls="-", c="green")
        plt.legend(loc="best")
    plt.savefig(str(Path(checkpoint_path, curve_name + '_curve.png')))
    plt.show()


def file_name_convert(file_list, zfill_num):
    """
    文件名转换，Change file name "0" -> "000.png"
    """
    result_list = [str(item).zfill(zfill_num) + ".png" for item in file_list]
    return result_list
