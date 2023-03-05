import numpy as np
from chainercv.evaluations import calc_semantic_segmentation_confusion

def compute_miou(preds, gts):
    '''
    This function takes two lists as inputs, where:
    preds: [list of 2-D prediction where each pixel corresponds to pixel-label]
    gts: [list of ground-truths in 2-D space, contains -1 value for critical pixels]
    
    returns mean of the iou for all classes
    '''
    confusion = calc_semantic_segmentation_confusion(preds, gts)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    return np.nanmean(iou)

def compute_mRecall(preds, gts):
    confusion = calc_semantic_segmentation_confusion(preds, gts)
    correct_fg_pred = np.diag(confusion[1:, 1:])
    fg_gt = confusion.sum(axis=1)[1:]
    
    recall = correct_fg_pred/fg_gt
    return np.nanmean(recall)

def compute_dr_recall(preds, dr_gts):
    '''
    This function takes two lists as inputs, where:
    preds: [list of 2-D prediction where each pixel corresponds to pixel-label]
    dr_gts: [list of discriminative region ground-truths in 2-D space, contains -1 value for all pixels except DR]
    
    returns mean recall of all classes for discriminative region
    '''
    return compute_mRecall(preds, dr_gts)

def compute_ndr_recall(preds, ndr_gts):
    '''
    This function takes two lists as inputs, where:
    preds: [list of 2-D prediction where each pixel corresponds to pixel-label]
    ndr_gts: [list of non-discriminative region ground-truths in 2-D space, contains -1 value for all pixels except NDR]
    
    returns mean recall of all classes for non-discriminative region
    '''
    return compute_mRecall(preds, ndr_gts)

def compute_mPrecision(preds, gts):
    '''
    This function takes two lists as inputs, where:
    preds: [list of 2-D prediction where each pixel corresponds to pixel-label]
    gts: [list of ground-truths in 2-D space, contains -1 value for critical pixels]
    
    returns mean of the precision for all foreground-classes
    '''
    confusion = calc_semantic_segmentation_confusion(preds, gts)
    correct_fg_pred = np.diag(confusion[1:, 1:])
    fg_pred = confusion.sum(axis=0)[1:]
    
    precision = correct_fg_pred/fg_pred
    
    return np.nanmean(precision)
    
