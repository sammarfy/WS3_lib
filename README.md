# WS3
Pipeline for the WS3 analysis experiment.

# Requirements
Code tested with environment listed in `requirements.txt`.

# Download Dataset
Download the [`PASCAL VOC 2012 dataset`](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and store that in 'datasets/' directory. 

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
| ...

```


# Example Notebook
CAM **mIoU, DR-NDR Recall, Foreground-precision** in [`ipynb/insight_paper_experiments/cam/org_model/basic_cam.ipynb`](https://github.com/marufvt/WS3_lib/blob/923f2063917b3575c97947aef564cc393e7151b9/ipynb/insight_paper_experiments/cam/org_model/basic_cam.ipynb) notebook.

<!-- Change the voc12/dataloader.py 'cls_labels_dict' variable accordingly before running the pipeline.
 -->
