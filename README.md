# Image segmentation metrics for microscopic image analysis
This is the official implementation of paper "Image Segmentation Metric and its Application in the Analysis of Material Microscopic Image" or 《图像分割评估方法及其在材料显微图像分析中的应用》

Image segmentation is an important branch in the field of computer vision. It aims to divide the image into several specific and unique regions. With the improvement of computer hardware computing ability and the progress of computing methods, a large number of image segmentation methods based on different theories have made great progress. Therefore, it is necessary to select appropriate evaluation metric to evaluate the accuracy and applicability of segmentation results, so as to select the optimal segmentation methods. In this paper, 14 evaluation metrics of image segmentation are summarized, which are divided into five categories: pixel based, intra class coincidence based, edge based, clustering based and instance based. In the application of material microscopic image analysis, the performance of different segmentation methods and different typical noises in different evaluation metrics are discussed through experiments. Finally, this paper discusses the advantages and applicability of various evaluation metrics.

## Requirements
All code are based on python37 and Unet is based on Pytorch.  
Gala is installed according to https://github.com/janelia-flyem/gala.  
Besides, you can install all packages by using:  

    pip install -r requirements.txt

## Citation
Still in submission

## Acknowledgement
The authors acknowledge the financial support from the National Key Research and Development Program of China (No. 2016YFB0700500).
