import os
import cv2, time
import skimage
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from sklearn.model_selection import KFold
from pathlib import Path
from utils import *
import metrics
from model.unet import UNet
from model.dataloader import WeightMapLoss, WeightMapDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_name = "iron"  # iron and al_la
model_class = UNet
model_name = "unet"

seed_num = 2020
kf_num = 5
kf = KFold(n_splits=kf_num, shuffle=True, random_state=seed_num)
val_rate = 0.1

if dataset_name == 'iron':
    z_score_norm = tr.Compose([
        tr.ToTensor(),
        tr.Normalize(mean=[0.9410404628082503],
                     std=[0.12481161024777744])
    ])
    file_num = 296

elif dataset_name == 'al_la':
    z_score_norm = tr.Compose([
        tr.ToTensor(),
        tr.Normalize(mean=[0.5275053008263852],
                     std=[0.21730661894972672])
    ])
    file_num = 49

crop_size = 512
zfill_num = 3
file_list = [item for item in range(file_num)]

learning_rate = 1e-4
epochs = 50
batch_size = 10
use_augment = [True, True, True]  # In training, rand_rotation, rand_vertical_flip, rand_horizontal_flip
no_augment = [False, False, False]  # In Validate and test


def train(net, epoch, dataloader, optimizer, learning_rate):
    net.train()
    for sample in dataloader:
        if torch.cuda.is_available():
            img, label, weight = sample['img'].cuda(), sample['label'].cuda(), sample['weight'].cuda()
        output = net.forward(img)
        loss = wm_criterion.forward(output, label, weight)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
    scheduler.step()


def val(net, epoch, dataloader, checkpoint_path, printer, check_first_img=True):
    net.eval()
    loss = 0
    is_first = True
    with torch.no_grad():
        for sample in dataloader:
            img = sample['img'];
            label = sample['label'];
            weight = sample['weight']
            if torch.cuda.is_available():
                img = img.cuda();
                label = label.cuda();
                weight = weight.cuda()
            output = net.forward(img)
            loss += wm_criterion.forward(output, label, weight)
            if torch.cuda.is_available():
                img = img.cpu();
                output = output.cpu();
                label = label.cpu();
                weight = weight.cpu()
            if check_first_img and is_first and epoch % 10 == 0:
                output = output.max(1)[1].data
                check_result(epoch,
                             img[0, :, :, :].squeeze().numpy(),
                             label[0, :, :, :].squeeze().numpy(),
                             output.squeeze().numpy(),
                             weight[0, :, :, :].squeeze().numpy().transpose((1, 2, 0)),
                             checkpoint_path, printer, description="val_")
            is_first = False
        loss /= len(dataloader.dataset)
        return loss.item()


wm_criterion = WeightMapLoss()
cwd = os.getcwd()

# K-fold Cross Validation
for kf_idx, (train_index, test_index) in enumerate(kf.split(np.array(file_list))):
    if kf_idx != 0:
        continue
    # Dataset split
    # we use k-fold cv to produce train set and test set, and then sample some data (20%) from train set as val set (sampling without replacement)
    random.seed(seed_num)
    val_index = random.sample(list(train_index), int(len(train_index) * val_rate) + 1)
    train_index = list(set(train_index).difference(set(val_index)))
    train_index.sort();
    val_index.sort()

    # convert index to file name (zfill): 10->010.png
    train_names = file_name_convert(train_index, zfill_num)
    val_names = file_name_convert(val_index, zfill_num)
    test_names = file_name_convert(test_index, zfill_num)

    experiment_name = 'cv_' + str(kf_idx + 1) + '_' + dataset_name + '_min_loss_' + model_name
    imgs_dir = Path(cwd, 'dataset', dataset_name, 'data_experiment')
    checkpoint_path = Path(cwd, 'model', 'parameters', experiment_name)
    if not os.path.exists(str(checkpoint_path)):
        os.mkdir(str(checkpoint_path))
    printer = Printer(True, str(Path(checkpoint_path, "loss.txt")))

    # train setting
    setup_seed(seed_num)

    train_dataset = WeightMapDataset(imgs_dir, train_names, use_augment=use_augment, crop_size=crop_size,
                                     norm_transform=z_score_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_one_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_dataset = WeightMapDataset(imgs_dir, val_names, use_augment=no_augment, crop_size=crop_size,
                                   norm_transform=z_score_norm)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = model_class(num_channels=1, num_classes=2)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    train_loss_list = [];
    val_loss_list = []
    val_baseline = 10000
    val_best_epoch = 0

    # Training
    st_total = time.time()
    printer.print_and_log("Training:")
    for i in range(1, epochs + 1):
        print("Experiment name: {}, {}/{}".format(experiment_name, i, epochs))
        st = time.time()
        train(model, i, train_loader, optimizer, learning_rate)
        train_loss = val(model, i, train_one_loader, checkpoint_path, printer, check_first_img=False)
        val_loss = val(model, i, val_loader, checkpoint_path, printer, check_first_img=True)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        printer.print_and_log(
            "Epoch {}: train_loss {:.4f}; val_loss {:.4f} \n".format(i, train_loss_list[-1], val_loss_list[-1]))
        plot(i, train_loss_list, val_loss_list, checkpoint_path, curve_name='loss')

        if val_loss < val_baseline and i > 10:
            val_baseline = val_loss
            val_best_epoch = i
            torch.save(model.state_dict(), str(Path(checkpoint_path, "best_model_state.pth")))
        ed = time.time()
        printer.print_and_log("Epoch Duration: {}'s".format(ed - st))
    ed_total = time.time()
    printer.print_and_log("Total duration is: {}'s".format(ed_total - st_total))
    printer.print_and_log("The best epoch is at: {} th epoch".format(val_best_epoch))
    printer.print_and_log("Train Loss list is: {}".format(train_loss_list))
    printer.print_and_log("Val Loss list is: {}".format(val_loss_list))