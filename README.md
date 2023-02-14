# WS3_lib
Pipeline for the WS3 analysis experiment.

# Requirements
Code tested in Lambdapgml server with environment listed in `requirements.txt`

# Download Model
Use gdown to download the saved resnet model into the 'sess/' directory.

Original Model:
```
gdown -O sess/ https://drive.google.com/uc?id=14XtzLupSKVmcM4I3shH6k72HzXz53WhM
```
Noisy Model:
```
gdown -O sess/ https://drive.google.com/uc?id=1cZ0XMXdhqxLNkG2K8jbxpgbZK2GmPOpD
```


# Example Notebook
CAM **mIoU, DR-NDR Recall, Foreground-precision** in [`ipynb/insight_paper_experiments/cam/org_model/basic_cam.ipynb`](https://github.com/marufvt/WS3_lib/blob/923f2063917b3575c97947aef564cc393e7151b9/ipynb/insight_paper_experiments/cam/org_model/basic_cam.ipynb) notebook.

<!-- Change the voc12/dataloader.py 'cls_labels_dict' variable accordingly before running the pipeline.
 -->
