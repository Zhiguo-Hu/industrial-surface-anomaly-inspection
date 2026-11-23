"""Binary segmentation metrics with explicit empty-mask semantics."""
import numpy as np


def mask_statistics(prediction, target):
    p,t=np.asarray(prediction,dtype=bool),np.asarray(target,dtype=bool)
    if p.shape!=t.shape:raise ValueError('Mask shapes differ')
    tp=int((p&t).sum());fp=int((p&~t).sum());fn=int((~p&t).sum())
    return {'tp_pixels':tp,'fp_pixels':fp,'fn_pixels':fn,'gt_pixels':int(t.sum()),
            'pred_pixels':int(p.sum()),'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'iou':tp/(tp+fp+fn) if tp+fp+fn else None,
            'pixel_recall':tp/(tp+fn) if tp+fn else None,
            'pixel_precision':tp/(tp+fp) if tp+fp else None}


def area_group(fraction):
    if fraction<=0:return 'normal'
    if fraction<.001:return 'under_0.1pct'
    if fraction<.01:return '0.1_to_1pct'
    if fraction<.05:return '1_to_5pct'
    return 'at_least_5pct'


def summarize(rows):
    if not rows:return {'images':0}
    positive=[r for r in rows if r['label']]
    tp=sum(r['tp_pixels'] for r in rows);fp=sum(r['fp_pixels'] for r in rows);fn=sum(r['fn_pixels'] for r in rows)
    return {'images':len(rows),'defective':len(positive),'image_detections':sum(r['alarm'] for r in positive),
            'image_recall':sum(r['alarm'] for r in positive)/len(positive) if positive else None,
            'mean_defective_image_dice':float(np.mean([r['dice'] for r in positive])) if positive else None,
            'pooled_union_dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'pooled_union_pixel_recall':tp/(tp+fn) if tp+fn else None,
            'pooled_union_pixel_precision':tp/(tp+fp) if tp+fp else None}
