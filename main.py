import os
import matplotlib.pyplot as plt
from skimage import io
from utils import *
from seg_methods import *
from metrics import *

seg_models = [OTSU]
seg_names  = ["otsu"]

cwd = os.getcwd()
img_dir = os.path.join(cwd, "data", "images")
label_dir = os.path.join(cwd, "data", "masks")
out_dir = os.path.join(cwd, "results")
img_names = os.listdir(img_dir)

for seg_idx in range(len(seg_models)):
    # Iterate all methods in seg_models
    for img_name in img_names:
        # Iterate all images
        img = load_img(os.path.join(img_dir, img_name))
        mask = load_img(os.path.join(label_dir, img_name))
        pred = seg_models[seg_idx](img)

        io.imsave(os.path.join(out_dir, seg_names[seg_idx] + "_" + img_name), (pred * 255).astype(np.uint8))

        # 评估 Evaluation
        metric_values, metric_names = get_total_evaluation(pred, mask)
        print("For {} method, the evaluation of {} have shown below:".format(seg_names[seg_idx], img_name))
        for metric_value, metric_name in zip(metric_values, metric_names):
            print(" {}: {}".format(metric_name, metric_value))

        # 可视化 Visualization
        plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title('img'), plt.axis("off")
        plt.subplot(1, 3, 2), plt.imshow(mask * 255, cmap="gray"), plt.title('mask'), plt.axis("off")
        plt.subplot(1, 3, 3), plt.imshow(pred * 255, cmap="gray"), plt.title(seg_names[seg_idx]), plt.axis("off")
        plt.show()
