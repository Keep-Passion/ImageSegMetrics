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
seg_names = ["otsu", "canny", "watershed", "k-means", "random_walker", "unet"]

cwd = os.getcwd()
result_dir = os.path.join(cwd, "results")


def eval_seg_methods(data_name: str, excel_writer):
    img_dir = os.path.join(cwd, "data", data_name, "images")
    label_dir = os.path.join(cwd, "data", data_name, "labels")

    out_dir = os.path.join(result_dir, data_name)

    pth_address = os.path.join("model", "unet_{}_parameter.pth".format(data_name))
    net_inference._set_pth_address(pth_address)

    create_folder(out_dir)
    img_names = os.listdir(img_dir)
    if ".ipynb_checkpoints" in img_names:
        img_names.remove(".ipynb_checkpoints")

    temp_metric_names = metric_names.copy()
    if data_name == "al_la":
        temp_metric_names.remove("figure_of_merit")
        temp_metric_names.remove("completeness")
        temp_metric_names.remove("correctness")
        temp_metric_names.remove("quality")

    # We provide 14 metrics, +2 means that vi includes merge error and split error
    seg_total_eval = np.zeros((len(seg_names), len(temp_metric_names), len(img_names)))

    for model_idx in range(len(seg_models)):
        # Iterate all methods in seg_models
        for img_idx, img_name in enumerate(img_names):
            # Iterate all images
            img = load_img(os.path.join(img_dir, img_name))
            label = load_img(os.path.join(label_dir, img_name))
            pred = seg_models[model_idx](img)
            io.imsave(os.path.join(out_dir, str(img_idx).zfill(3) + "_" + seg_names[model_idx] + ".png"),
                      (pred * 255).astype(np.uint8))

            # 评估 Evaluation
            if data_name == "iron":
                metric_values = get_total_evaluation(pred, label, require_edge=True)
            else:
                metric_values = get_total_evaluation(pred, label, require_edge=False)
            print("For {} method, the evaluation of {} in {} have shown below:".format(seg_names[model_idx], img_name,
                                                                                       data_name))
            for metric_idx, metric_value in enumerate(metric_values):
                print(" {}: {}".format(temp_metric_names[metric_idx], metric_value), end=", ")
                seg_total_eval[model_idx, metric_idx, img_idx] = metric_value
            print("")

            # 可视化 Visualization
            # plt.figure(figsize=(15, 15))
            # plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title('img'), plt.axis("off")
            # plt.subplot(1, 3, 2), plt.imshow(label * 255, cmap="gray"), plt.title('label'), plt.axis("off")
            # plt.subplot(1, 3, 3), plt.imshow(pred * 255, cmap="gray"), plt.title(seg_names[model_idx]), plt.axis("off")
            # plt.show()

    # export excel result
    mean_np = seg_total_eval.mean(axis=2)
    metrics_df = pd.DataFrame(index=seg_names, columns=temp_metric_names)
    print("Total evaluation")
    for seg_idx in range(len(seg_names)):
        print("{}: ".format(seg_names[seg_idx]))
        for metric_idx in range(len(temp_metric_names)):
            metrics_df.iloc[seg_idx, metric_idx] = mean_np[seg_idx, metric_idx]
            print(" {}: {}, ".format(temp_metric_names[metric_idx], mean_np[seg_idx, metric_idx]), end="")
        print("")

    metrics_df.to_excel(excel_writer, sheet_name=data_name + "_seg_methods", index=seg_names, columns=temp_metric_names)
    # metrics_df.to_excel(os.path.join(result_dir, "total_evaluation.xlsx"), sheet_name=data_name)


def eval_noises(data_name: str, excel_writer):
    img_dir = os.path.join(cwd, "data", "noise")
    label = load_img(os.path.join(img_dir, data_name + "_label.png"))

    temp_metric_names = metric_names.copy()
    noise_types = ["noise", "scratch", "miss"]
    if data_name == "al_la":
        temp_metric_names.remove("figure_of_merit")
        temp_metric_names.remove("completeness")
        temp_metric_names.remove("correctness")
        temp_metric_names.remove("quality")
        noise_types.remove("miss")

    noises_total_eval = np.zeros((len(noise_types), len(temp_metric_names)))
    for noise_idx, noise_name in enumerate(noise_types):
        noise_img = load_img(os.path.join(img_dir, data_name + "_" + noise_name + ".png"))
        print("The difference of the number of background pixel is {}.".format(
            np.count_nonzero(noise_img == 0) - np.count_nonzero(label == 0)))
        # 评估 Evaluation
        if data_name == "iron":
            metric_values = get_total_evaluation(noise_img, label, require_edge=True)
        else:
            metric_values = get_total_evaluation(noise_img, label, require_edge=False)

        print("For {}, the evaluation have shown below:".format(data_name + "_" + noise_name))
        for metric_idx, metric_value in enumerate(metric_values):
            print(" {}: {}".format(temp_metric_names[metric_idx], metric_value), end=", ")
            noises_total_eval[noise_idx, metric_idx] = metric_value
        print("")

    noises_df = pd.DataFrame(index=noise_types, columns=temp_metric_names)
    for noise_idx in range(len(noise_types)):
        for metric_idx in range(len(temp_metric_names)):
            noises_df.iloc[noise_idx, metric_idx] = noises_total_eval[noise_idx, metric_idx]

    noises_df.to_excel(excel_writer, sheet_name=data_name + "_noises", index=noise_types, columns=temp_metric_names)


if __name__ == '__main__':
    excel_writer = pd.ExcelWriter(os.path.join(result_dir, "total_evaluation.xlsx"))

    # Evaluate different segmentation methods
    # eval_seg_methods("iron", excel_writer)
    # eval_seg_methods("al_la", excel_writer)

    # Evaluate different noises
    eval_noises("iron", excel_writer)
    eval_noises("al_la", excel_writer)

    excel_writer.close()
