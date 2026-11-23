"""Matched architecture/loss-budget control for explicit tile classification supervision."""
import csv,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.reproducibility import configure,state_hash
from src.resnet_unet import JointResNet18UNet
from scripts.train_steel_unet import predict_outputs
from scripts.run_steel_pilot import read_image
BASE='steel_unet_balanced_pretrained_resnet18_expanded_r30_det_s42'


def main():
    names=[BASE+'_cls0',BASE+'_cls0p5']
    m={n:json.loads((ROOT/f'reports/{n}_metrics.json').read_text()) for n in names}
    s={n:json.loads((ROOT/f'configs/{n}_selection.json').read_text()) for n in names}
    assert s[names[0]]==s[names[1]]==json.loads((ROOT/f'configs/{BASE}_selection.json').read_text())
    assert m[names[0]]['initial_state_sha256']==m[names[1]]['initial_state_sha256']
    for key in ['parameters','total_updates','optimizer','sampling','actual_sampled_tiles_per_class','reproducibility']:
        assert m[names[0]][key]==m[names[1]][key]
    historical=json.loads((ROOT/f'reports/{BASE}_metrics.json').read_text())
    control=torch.load(ROOT/f'runs/{names[0]}/best.pt',map_location='cpu',weights_only=True)
    segmentation={k:v for k,v in control['model'].items() if not k.startswith('classifier.')}
    same_state=state_hash(segmentation)==historical['best_state_sha256']
    same_history=len(m[names[0]]['history'])==len(historical['history']) and all(a['training_loss']==b['training_loss'] and
                     a['checkpoint_validation_pooled_dice_at_05']==b['checkpoint_validation_pooled_dice_at_05']
                     for a,b in zip(m[names[0]]['history'],historical['history']))
    assert same_state and same_history,'Weight-zero control did not reproduce historical segmentation'
    for key in ['best_round','image_auroc','confusion','pixel_threshold','image_threshold',
                'pooled_pixel_dice_all_200_images','pooled_pixel_iou_all_200_images']:
        assert m[names[0]][key]==historical[key],f'Historical segmentation metric changed: {key}'
    historical_diagnosis=json.loads((ROOT/f'reports/{BASE}_diagnosis.json').read_text())['normal_diagnostics']
    assert m[names[0]]['normal_masks']['nonempty_of_100']==historical_diagnosis['images_with_any_predicted_pixels']
    assert m[names[0]]['normal_masks']['normal_pixel_fpr']==historical_diagnosis['pooled_pixel_fpr']
    predictions={}
    for n in names:
        with (ROOT/f'runs/{n}/validation_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
            predictions[n]=list(csv.DictReader(f))
    assert [r['image_id'] for r in predictions[names[0]]]==[r['image_id'] for r in predictions[names[1]]]
    transitions={}
    for label,title in [(0,'normal_images'),(1,'defective_images')]:
        pairs=[(int(a['prediction']),int(b['classification_alarm']))
               for a,b in zip(predictions[names[0]],predictions[names[1]]) if int(a['label'])==label]
        transitions[title]={f'baseline_{a}_joint_head_{b}':pairs.count((a,b)) for a in (0,1) for b in (0,1)}
    configure(42,True);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    rows={r['image_id']:r for r in json.loads((ROOT/'configs/severstal_manifest.json').read_text())['samples']}
    joint_rows=predictions[names[1]]
    classification_diagnosis={
        'score_quantiles_10_50_90':{title:np.quantile([float(r['classification_score']) for r in joint_rows if int(r['label'])==label],[.1,.5,.9]).tolist()
                                    for label,title in [(0,'normal'),(1,'defective')]},
        'per_class_overlapping_image_groups':{str(c):{
            'images':sum(c in rows[r['image_id']]['classes'] for r in joint_rows),
            'head_true_positives':sum(int(r['classification_alarm']) for r in joint_rows if c in rows[r['image_id']]['classes'])}
            for c in (1,2,3,4)}}
    with (ROOT/f'runs/{names[1]}_diagnosis/per_image.csv').open(encoding='utf-8-sig',newline='') as f:
        pixel_rows=list(csv.DictReader(f))
    alarms={r['image_id']:int(r['classification_alarm']) for r in joint_rows}
    gt=sum(int(r['gt_pixels']) for r in pixel_rows)
    raw_tp=sum(int(r['tp_pixels']) for r in pixel_rows)
    lost_tp=sum(int(r['tp_pixels']) for r in pixel_rows if not alarms[r['image_id']])
    classification_diagnosis['gating_pixel_tradeoff']={
        'raw_pixel_recall':raw_tp/gt,'gated_pixel_recall':(raw_tp-lost_tp)/gt,
        'true_positive_pixels_removed':lost_tp,'fraction_raw_tp_removed':lost_tp/max(1,raw_tp),
        'defective_images_with_some_true_positive_pixels_but_negative_head':sum(int(r['label'])==1 and int(r['tp_pixels'])>0 and not alarms[r['image_id']] for r in pixel_rows)}
    classification_diagnosis['head_recall_by_gt_area']={group:{
        'images':len(subset),'detected':sum(alarms[r['image_id']] for r in subset)}
        for group in ['under_0.1pct','0.1_to_1pct','1_to_5pct','at_least_5pct']
        for subset in [[r for r in pixel_rows if r['area_group']==group]]}
    reload_checks={}
    for n in names:
        saved=torch.load(ROOT/f'runs/{n}/best.pt',map_location=device,weights_only=True)
        model=JointResNet18UNet().to(device);model.load_state_dict(saved['model']);model.eval()
        first=predictions[n][0]
        heat,cls=predict_outputs(model,read_image(rows[first['image_id']]),device)
        np.testing.assert_allclose([float(heat.max()),cls],[float(first['score']),float(first['classification_score'])],rtol=1e-5,atol=1e-5)
        assert state_hash(saved['model'])==m[n]['best_state_sha256']
        reload_checks[n]={'status':'passed','image_id':first['image_id'],'mask_max':float(heat.max()),'classification_score':cls}
    report={'scope':'Seed42 deterministic, matched architecture/data/initialization/update budget; no test images',
            'changed_factor':'Classification loss weight 0 -> 0.5; checkpoint selection still segmentation development Dice',
            'controls_verified':True,'control_head_warning':'Weight-0 classifier is untrained; do not present its chance scores as a learned detector.',
            'historical_segmentation_best_state_reproduced':same_state,
            'historical_segmentation_full_history_reproduced':same_history,
            'checkpoint_reload_checks':reload_checks,'paired_alarm_transitions':transitions,
            'joint_classification_diagnosis':classification_diagnosis,
            'results':{n:{key:m[n][key] for key in ['classification_weight','best_round','image_auroc','confusion',
                        'pooled_pixel_dice_all_200_images','pooled_pixel_iou_all_200_images','normal_masks',
                        'classification_metrics','completed_optimizer_steps','wall_seconds']} for n in names}}
    text=json.dumps(report,indent=2)
    (ROOT/'reports/steel_joint_comparison.json').write_text(text,encoding='utf-8');print(text)


if __name__=='__main__':main()
