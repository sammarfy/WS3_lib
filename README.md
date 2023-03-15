# WS3
Pipeline for the WS3 analysis experiment.

# Requirements
Code tested with environment listed in `requirements.txt`.

# Download Dataset
Download the [`PASCAL VOC 2012 dataset`](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and store that in 'datasets/' directory. 

# Download Ground Truths
Download and unzip Superpixels, DR Ground Truth, and NDR Ground Truth in the 'gts/' directory.
```
gdown -O gts/ https://drive.google.com/uc?id=1FACqnJ_9mEozvtv0GGZ2diAtj7KiynQx

gdown -O gts/ https://drive.google.com/uc?id=1Uf9c6NK2pE-YlyV5LMrpuTudNvivFTvG

gdown -O gts/ https://drive.google.com/uc?id=1dSSMOHAk0S9_WkAjDHz6evUbEI_F4r0i

```


# Download Model
Use gdown to download the saved resnet model into the 'sess/' directory.

Original Model:
```
gdown -O sess/ https://drive.google.com/uc?id=14XtzLupSKVmcM4I3shH6k72HzXz53WhM
```
Noisy Model (Binary Mask Perturbation):
```
gdown -O sess/ https://drive.google.com/uc?id=1cZ0XMXdhqxLNkG2K8jbxpgbZK2GmPOpD
```
Noisy Model (Gaussian Noise Perturbation):
```
gdown -O sess/ https://drive.google.com/uc?id=1BYsy0-X1Ksc0n2yY-Q7dBNvUXwZ-Hee_
```
Alternatively, we can fine-tune these models using the following command:

```
bash fine-tuning/generate_pseudo_mask.sh
```

# Example of directory hierarchy:

```
WS3
|--- sess
|    |--- res50_cam_original_version.pth.pth
|    |--- res50_cam_gaussian_noise_version.pth.pth
|    |--- res50_cam_bin_noise_version.pth.pth
|--- voc12
|    |--- train.txt
|    | ...
|--- datasets
|    |--- VOCdevkit/VOC2012
|         | ...
|--- notebooks
| ...

```


# Notebook
CAM and Saliency generation and analysis notebooks are in the 'notebook\' directory.
