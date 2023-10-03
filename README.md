# Crowd Counting and Localization for Surveillance Videos

### Introduction
---
This project implements a crowd counting method and is heavily inspired by the repository at [CrowdCounting-P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet).

The overall architecture of the P2PNet. Built upon the VGG16, it firstly introduce an upsampling path to obtain fine-grained feature map. Then it exploits two branches to simultaneously predict a set of point proposals and their confidence scores.

## Test

A trained model on the MALL_DATASET, SHTechPartA, and JHU-CROWD++ datasets is available in the './weights' directory. To predict the locations of individuals in test images, please run the following commands:

<pre>
!CUDA_VISIBLE_DEVICES=0 python test.py --threshold 0.8 \
    --images_path $IMAGE_ROOT \
    --weight_path  ./pretrained_model/best_mae.pth \
    --output_dir ./output/images/ \
    --predicts_txt_dir ./output/predict_txt.txt \
    --predicts_point_dir ./logs/new_thr=0.8
</pre>
  
**image_path** : test image folder address

**weight_path** : weights of best model that trained on 3 famous crowd counting dataset: 1- **MALL Dataset** ([Dataset Link](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)) 2- **ShTech Dataset** ([Dataset Link](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0)) 3- **JHU-CROWD++** ([Dataset Link](http://www.crowd-counting.com/))

**output_dir** : predicted images adress

**predicted_txt_dir** : address of the text file that contains predicted nubmer of person in each image.

## Visualization of Predictions

There are sampels of predicting location of individuals in some test images.

<p align="center">
  <img src="images/img2.jpg" width="1000" >
</p>

<p align="center">
  <img src="images/img3.jpg" width="1000">
</p>

## Heatmap

To extract heatmap, please run the following commands:

<pre>
!CUDA_VISIBLE_DEVICES=0 python density_map.py --images_path $IMG_PATH \
    --points_path ./output/predict_txt.txt \
    --method  'fixed' \
    --output_dir ./heatmaps 
</pre>

**images_path** : Test images path

**point_path** : predicted coordinates of individual path

**output_dir** : address to save  heatmaps

A sample of heatmap is provided in below:

<p align="center">
  <img src="images/heatmap.jpg" width="1000">
</p>

Backbone weights(vgg16) is so big(540MB), It is not uploaded on the Github and you have to download it with the following code
> !wget https://download.pytorch.org/models/vgg16_bn-6c64b313.pth

## Demo Video
> [**Youtube Link**](https://youtu.be/fyVCOq6zjss)
> [![Watch the video](images/video_cover.png)](https://youtu.be/fyVCOq6zjss)


