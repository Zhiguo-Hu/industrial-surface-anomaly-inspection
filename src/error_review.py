"""Descriptive frozen-error review helpers; no training or threshold selection."""
import numpy as np
from scipy import ndimage


def select_errors(records,threshold):
    result=[]
    for row in records:
        label=int(row['label']);prediction=int(row['prediction']);score=float(row['score'])
        if not np.isfinite(score) or not np.isfinite(threshold) or label not in (0,1) or prediction!=int(score>threshold):raise ValueError('Archived prediction contradicts frozen policy')
        if label!=prediction:result.append({**row,'label':label,'prediction':prediction,'score':score,'kind':'FN' if label else 'FP'})
    return sorted(result,key=lambda r:(r['kind']!='FN',r['score'] if r['kind']=='FN' else -r['score']))


def boundary(mask):
    mask=np.asarray(mask,dtype=bool)
    return mask&~ndimage.binary_erosion(mask,structure=np.ones((3,3)),border_value=0)


def patch_start(truth,probability,cam,width=256):
    if truth.any():anchor=int(truth.sum(0).argmax())
    else:
        response=cam if cam.max()>0 else probability
        anchor=int(np.unravel_index(response.argmax(),response.shape)[1])
    return max(0,min(truth.shape[1]-width,anchor-width//2))


def case_evidence(array,truth,probability,cam,view_logits,score,threshold,pixel_threshold):
    truth=np.asarray(truth,dtype=bool);pred=probability>pixel_threshold;tp=int((truth&pred).sum());gt=int(truth.sum());pp=int(pred.sum())
    gray=array.astype(np.float32).mean(2);dark=(gray<8).mean(0)>.95;transitions=np.flatnonzero(np.diff(dark.astype(np.int8)))+1
    result={'gt_pixels':gt,'gt_area_fraction':gt/truth.size,'predicted_pixels':pp,'predicted_area_fraction':pp/truth.size,
            'overlap_pixels':tp,'pixel_dice':2*tp/max(1,gt+pp) if gt else None,'pixel_recall':tp/gt if gt else None,
            'mask_max':float(probability.max()),'near_black_column_fraction':float(dark.mean()),
            'score_margin':float(score-threshold),'view_logits':[float(v) for v in view_logits],
            'views_cross_frozen_threshold':bool(min(view_logits)<=threshold<max(view_logits)),
            'cam_positive_energy_present':bool(cam.max()>0),'cam_energy_fraction_in_gt':float(cam[truth].sum()/cam.sum()) if gt and cam.sum()>0 else None,
            'gt_side_distance_px':None,'gt_to_dark_boundary_px':None,'local_mean_contrast_snr':None,
            'largest_component_min_side_px':None,'gt_components':0}
    if gt:
        y,x=np.where(truth);result['gt_side_distance_px']=int(min(x.min(),truth.shape[1]-1-x.max()))
        columns=np.flatnonzero(truth.any(0))
        if len(transitions):result['gt_to_dark_boundary_px']=int(np.abs(columns[:,None]-transitions[None,:]).min())
        ring=ndimage.binary_dilation(truth,iterations=8)&~truth
        if ring.any():result['local_mean_contrast_snr']=float(abs(gray[truth].mean()-gray[ring].mean())/(gray[ring].std()+1e-6))
        labels,n=ndimage.label(truth,structure=np.ones((3,3)));result['gt_components']=int(n)
        sizes=np.bincount(labels.ravel())[1:];box=ndimage.find_objects(labels)[int(sizes.argmax())]
        result['largest_component_min_side_px']=int(min(s.stop-s.start for s in box))
    flags=[]
    if abs(score-threshold)<=.5:flags.append('near_threshold_0p5_logit')
    if result['views_cross_frozen_threshold']:flags.append('flip_view_disagreement')
    if gt:
        if gt/truth.size<.001:flags.append('gt_area_below_0p1pct')
        if result['largest_component_min_side_px']<=8:flags.append('largest_gt_component_thin_8px')
        if result['gt_side_distance_px']<=32:flags.append('gt_near_side_edge_32px')
        if result['gt_to_dark_boundary_px'] is not None and result['gt_to_dark_boundary_px']<=16:flags.append('gt_near_dark_boundary_16px')
        if result['local_mean_contrast_snr'] is not None and result['local_mean_contrast_snr']<.5:flags.append('low_local_mean_contrast')
        if score<=threshold and result['pixel_dice']>=.2 and result['pixel_recall']>=.25:flags.append('localized_overlap_without_image_alarm')
    elif pp/truth.size<.0001:flags.append('image_alarm_with_sparse_pixel_mask')
    result['descriptive_flags']=flags;return result
