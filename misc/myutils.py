import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

def unnormalize_image(image_tensor):
    
    ''' This method takes a torch tensor (Image) as an input and 
    produces numpy array as output. 
    Goal: Unnormalize image using Imagenet mean and std deviation
    Input: image_tensor (3, W, H) [Torch tensor]
    Output:  out_image (W, H, 3) [Numpy array]
    '''

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    image_tensor = image_tensor.squeeze(dim=0)
    norm_img = image_tensor.permute(1,2,0).detach().cpu().numpy()
    W, H, C =  norm_img.shape[0], norm_img.shape[1], norm_img.shape[2]
    out_image = np.tile(std, (W, H, 1)) * norm_img + np.tile(mean, (W, H, 1))
    out_image = np.clip(out_image, a_min=0, a_max=1)

    return out_image

from chainercv.evaluations import calc_semantic_segmentation_confusion

def get_miou(labels, preds):
    confusion = calc_semantic_segmentation_confusion(preds, labels)
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    
    return np.nanmean(iou)