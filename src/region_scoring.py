"""Spatial probability aggregation with explicit mask and threshold semantics."""
import numpy as np
from scipy import ndimage


def clean_mask(probability, pixel_threshold, minimum_area):
    p=np.asarray(probability)
    if p.ndim!=2 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected finite 2-D probabilities in [0,1]')
    if not 0<=pixel_threshold<=1 or minimum_area<0:raise ValueError('Invalid threshold/area')
    labels,count=ndimage.label(p>pixel_threshold,structure=np.ones((3,3),dtype=np.uint8))
    areas=np.bincount(labels.ravel(),minlength=count+1)
    keep=areas>=minimum_area;keep[0]=False
    return keep[labels]


def region_scores(probability, pixel_threshold, minimum_area):
    p=np.asarray(probability)
    mask=clean_mask(p,pixel_threshold,minimum_area)
    labels,count=ndimage.label(mask,structure=np.ones((3,3),dtype=np.uint8))
    mass=np.bincount(labels.ravel(),weights=p.ravel(),minlength=count+1);mass[0]=0
    k=max(1,int(np.ceil(p.size*.001)))
    return {'max_probability':float(p.max()),
            'top_0.1pct_mean':float(np.partition(p.ravel(),p.size-k)[-k:].mean()),
            'mask_area_fraction':float(mask.mean()),
            'largest_component_probability_mass':float(mass.max()/p.size)},mask


def pooled_dice(counts):
    tp,fp,fn=counts
    return float(2*tp/max(1,2*tp+fp+fn))
