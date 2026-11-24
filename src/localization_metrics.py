"""Ungated binary-mask metrics from per-image TP, ground-truth and prediction counts."""
import numpy as np


def localization_metrics(counts,threshold,pixels_per_image=256*1600):
    values=np.asarray(counts,dtype=np.int64);tp,gt,pred=values.sum(0);positive=values[:,1]>0;normal=~positive
    dice=2*values[positive,0]/np.maximum(1,values[positive,1]+values[positive,2])
    return {'threshold':float(threshold),'ungated':True,'images':len(values),'global_dice':float(2*tp/max(1,gt+pred)),
            'global_iou':float(tp/max(1,gt+pred-tp)),'pixel_recall':float(tp/max(1,gt)),
            'pixel_precision':float(tp/max(1,pred)),'mean_positive_image_dice':float(dice.mean()) if len(dice) else 0.,
            'normal_pixel_fpr':float(values[normal,2].sum()/max(1,normal.sum()*pixels_per_image))}
