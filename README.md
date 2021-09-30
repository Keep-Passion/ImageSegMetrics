# Image segmentation metrics for microscopic image analysis
This is the official implementation of paper "Image Segmentation Metric and its Application in the Analysis of Material Microscopic Image" or 《图像分割评估方法及其在材料显微图像分析中的应用》

Image segmentation is an important branch in the field of computer vision. It aims to divide the image into several specific and unique regions. With the improvement of computer hardware computing ability and the progress of computing methods, a large number of image segmentation methods based on different theories have made great progress. Therefore, it is necessary to select appropriate evaluation metric to evaluate the accuracy and applicability of segmentation results, so as to select the optimal segmentation methods. In this paper, 14 evaluation metrics of image segmentation are summarized, which are divided into five categories: pixel based, intra class coincidence based, edge based, clustering based and instance based. In the application of material microscopic image analysis, the performance of different segmentation methods and different typical noises in different evaluation metrics are discussed through experiments. Finally, this paper discusses the advantages and applicability of various evaluation metrics.

## Requirement
All code are based on python37 and Unet is based on Pytorch.  
Gala is installed according to https://github.com/janelia-flyem/gala.  
Besides, you can install all packages by using:  

    pip install -r requirements.txt

## Citation
MA Bo-yuan, JIANG Shu-fang, YIN Dou, SHEN Hao-kai, BAN Xiao-juan, HUANG Hai-you, WANG Hao, XUE Wei-hua, FENG Hua. Image segmentation metric and its application in the analysis of microscopic image[J]. Chinese Journal of Engineering, 2021, 43(1): 137-149. doi: 10.13374/j.issn2095-9389.2020.05.28.002

or

马博渊, 姜淑芳, 尹豆, 申昊锴, 班晓娟, 黄海友, 王浩, 薛维华, 封华. 图像分割评估方法在显微图像分析中的应用[J]. 工程科学学报, 2021, 43(1): 137-149. doi: 10.13374/j.issn2095-9389.2020.05.28.002

## Acknowledgement
The authors acknowledge the financial support from the National Key Research and Development Program of China (No. 2016YFB0700500).
