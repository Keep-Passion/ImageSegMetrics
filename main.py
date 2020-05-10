import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
from utils import *
from seg_methods import *
from metrics import *


metric_names = ["pixel_accuracy", "mean_accuracy",
                "iou", "fwiou", "dice",
                "figure_of_merit", "completeness", "correctness", "quality",
                "ri", "ari", "me", "se", "vi",
                "cardinality_difference", "map"]
seg_models = [OTSU, Canny, Watershed, Kmeans, RandomWalker, unet]
seg_names  = ["otsu", "canny", "watershed", "k-means", "random_walker", "unet"]

cwd = os.getcwd()
img_dir = os.path.join(cwd, "data", "images")
mask_dir = os.path.join(cwd, "data", "masks")
out_dir = os.path.join(cwd, "results")
create_folder(out_dir)
img_names = os.listdir(img_dir)
if ".ipynb_checkpoints" in img_names:
    img_names.remove(".ipynb_checkpoints")

# We provide 14 metrics, +2 means that vi includes merge error and split error
seg_total_eval = np.zeros((len(seg_names), len(metric_names), len(img_names)))

for model_idx in range(len(seg_models)):
    # Iterate all methods in seg_models
    for img_idx, img_name in enumerate(img_names):
        # Iterate all images
        img = load_img(os.path.join(img_dir, img_name))
        mask = load_img(os.path.join(mask_dir, img_name))
        pred = seg_models[model_idx](img)
        io.imsave(os.path.join(out_dir, str(img_idx).zfill(3) + "_" + seg_names[model_idx] + ".png"),
                  (pred * 255).astype(np.uint8))

        # 评估 Evaluation
        metric_values = get_total_evaluation(pred, mask)
        print("For {} method, the evaluation of {} have shown below:".format(seg_names[model_idx], img_name))
        for metric_idx, metric_value in enumerate(metric_values):
            print(" {}: {:.4}".format(metric_names[metric_idx], metric_value), end=", ")
            seg_total_eval[model_idx, metric_idx, img_idx] = metric_value
        print("")

        # 可视化 Visualization
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title('img'), plt.axis("off")
        plt.subplot(1, 3, 2), plt.imshow(mask * 255, cmap="gray"), plt.title('mask'), plt.axis("off")
        plt.subplot(1, 3, 3), plt.imshow(pred * 255, cmap="gray"), plt.title(seg_names[model_idx]), plt.axis("off")
        plt.show()

# export excel result
mean_np = seg_total_eval.mean(axis=2)
metrics_df = pd.DataFrame(index=seg_names, columns=metric_names)
print("Total evaluation")
for seg_idx in range(len(seg_names)):
    print("{}: ".format(seg_names[seg_idx]))
    for metric_idx in range(len(metric_names)):
        metrics_df.iloc[seg_idx, metric_idx] = mean_np[seg_idx, metric_idx]
        print(" {}: {:.4}, ".format(metric_names[metric_idx], mean_np[seg_idx, metric_idx]), end="")
    print("")
metrics_df.to_excel(os.path.join(out_dir, "total_evaluation.xlsx"))
