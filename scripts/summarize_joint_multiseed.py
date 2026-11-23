"""Audit paired seeds and summarize unchanged joint supervision; no test evaluation."""
import argparse,csv,hashlib,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.reproducibility import configure,state_hash
from src.resnet_unet import ResNet18UNet,JointResNet18UNet
from scripts.train_steel_unet import predict_outputs
from scripts.run_steel_pilot import read_image

BASE='steel_unet_balanced_pretrained_resnet18_expanded_r30_det_s'
SEEDS=(42,43,44)


def read_json(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))


def predictions(name):
    with (ROOT/f'runs/{name}/validation_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def initial_checks():
    """CPU-only audit avoids competing with a running GPU training process."""
    torch.set_num_threads(4);result={}
    for seed in SEEDS:
        historical=read_json(f'reports/{BASE}{seed}_metrics.json')
        configure(seed,True);original=ResNet18UNet(ROOT/'weights/resnet18-f37072fd.pth')
        original_hash=state_hash(original.state_dict())
        configure(seed,True);joint=JointResNet18UNet(ROOT/'weights/resnet18-f37072fd.pth')
        segmentation={k:v for k,v in joint.state_dict().items() if not k.startswith('classifier.')}
        assert original_hash==historical['initial_state_sha256']==state_hash(segmentation)
        result[str(seed)]={'segmentation_initial_state_matches_historical':True,
                           'segmentation_initial_state_sha256':original_hash,
                           'joint_initial_state_sha256':state_hash(joint.state_dict())}
    return result


def descriptive(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'sample_std_ddof1':float(a.std(ddof=1)),
            'min':float(a.min()),'max':float(a.max())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialization-only',action='store_true');args=parser.parse_args()
    initialization=initial_checks()
    if args.initialization_only:
        print(json.dumps(initialization,indent=2));return
    manifest=read_json('configs/severstal_manifest.json')
    rows={r['image_id']:r for r in manifest['samples']}
    manifest_hash=hashlib.sha256((ROOT/'configs/severstal_manifest.json').read_bytes()).hexdigest()
    common_selection=None;per_seed={};alarm_matrices={};pixel_rows=None
    for seed in SEEDS:
        old_name=BASE+str(seed);name=old_name+'_cls0p5'
        old=read_json(f'reports/{old_name}_metrics.json');new=read_json(f'reports/{name}_metrics.json')
        old_diag=read_json(f'reports/{old_name}_diagnosis.json')
        diag=read_json(f'reports/{name}_diagnosis.json')
        selection=read_json(f'configs/{name}_selection.json')
        assert selection==read_json(f'configs/{old_name}_selection.json')
        if common_selection is None:common_selection=selection
        assert selection==common_selection
        for key in ['optimizer','sampling','total_updates','actual_sampled_tiles_per_class',
                    'training_positive_tiles','training_negative_tiles','reproducibility']:
            assert old[key]==new[key],f'Changed paired protocol: {seed} {key}'
        assert new['classification_weight']==.5 and new['total_updates']==2400
        assert new['initial_state_sha256']==initialization[str(seed)]['joint_initial_state_sha256']
        settings=new['reproducibility']
        assert settings['deterministic_algorithms'] and settings['cudnn_deterministic']
        assert not any(settings[k] for k in ['cudnn_benchmark','cudnn_allow_tf32','matmul_allow_tf32'])
        assert settings['cublas_workspace_config']==':4096:8'
        assert new['classification_metrics']['head_trained']
        assert len(new['history'])==30
        old_preds,new_preds=predictions(old_name),predictions(name)
        assert [r['image_id'] for r in old_preds]==[r['image_id'] for r in new_preds]==selection['evaluation']
        for p in old_preds+new_preds:assert int(p['label'])==rows[p['image_id']]['label']
        with (ROOT/f'runs/{name}_diagnosis/per_image.csv').open(encoding='utf-8-sig',newline='') as f:
            pixels=list(csv.DictReader(f))
        if pixel_rows is None:pixel_rows=pixels
        assert [r['image_id'] for r in pixels]==selection['evaluation']
        for a,b in zip(pixel_rows,pixels):
            for key in ['gt_pixels','area_group','classes']:assert a[key]==b[key]
        cls=new['classification_metrics'];alarms={r['image_id']:int(r['classification_alarm']) for r in new_preds}
        labels=[int(r['label']) for r in new_preds]
        head_scores=[float(r['classification_score']) for r in new_preds]
        assert all(int(r['classification_alarm'])==int(float(r['classification_score'])>cls['threshold']) for r in new_preds)
        counts=confusion_matrix(labels,[alarms[r['image_id']] for r in new_preds],labels=[0,1]).ravel()
        assert dict(zip(['tn','fp','fn','tp'],map(int,counts)))==cls['confusion']
        np.testing.assert_allclose(roc_auc_score(labels,head_scores),cls['image_auroc'])
        gt=sum(int(r['gt_pixels']) for r in pixels);raw_tp=sum(int(r['tp_pixels']) for r in pixels)
        lost=sum(int(r['tp_pixels']) for r in pixels if not alarms[r['image_id']])
        per_class={str(c):{'images':sum(c in rows[r['image_id']]['classes'] for r in new_preds),
                            'detected':sum(int(r['classification_alarm']) for r in new_preds if c in rows[r['image_id']]['classes'])}
                   for c in (1,2,3,4)}
        by_area={g:{'images':len(subset),'detected':sum(alarms[r['image_id']] for r in subset)}
                 for g in ['under_0.1pct','0.1_to_1pct','1_to_5pct','at_least_5pct']
                 for subset in [[r for r in pixels if r['area_group']==g]]}
        transitions={}
        for label,title in [(0,'normal'),(1,'defective')]:
            pairs=[(int(a['prediction']),int(b['classification_alarm']))
                   for a,b in zip(old_preds,new_preds) if int(a['label'])==label]
            transitions[title]={f'baseline_{a}_head_{b}':pairs.count((a,b)) for a in (0,1) for b in (0,1)}
        configure(seed,True);torch.set_num_threads(4)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        saved=torch.load(ROOT/f'runs/{name}/best.pt',map_location=device,weights_only=True)
        assert state_hash(saved['model'])==new['best_state_sha256']==saved['model_state_sha256']
        assert saved['manifest_sha256']==manifest_hash and saved['classification_weight']==.5
        assert saved['classification_threshold']==cls['threshold']
        model=JointResNet18UNet().to(device);model.load_state_dict(saved['model']);model.eval()
        first=new_preds[0];heat,score=predict_outputs(model,read_image(rows[first['image_id']]),device)
        np.testing.assert_allclose([float(heat.max()),score],[float(first['score']),float(first['classification_score'])],rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(diag['overall']['pooled_union_dice'],new['pooled_pixel_dice_all_200_images'])
        per_seed[str(seed)]={
            'initialization':initialization[str(seed)],'checkpoint_reload_one_image':'passed at tolerance 1e-5',
            'best_round':new['best_round'],'attempted_updates':new['total_updates'],
            'completed_updates':new['completed_optimizer_steps'],'amp_skipped_updates':new['optimizer_steps_skipped_by_amp'],
            'baseline_completed_updates':old['completed_optimizer_steps'],
            'baseline':{'image_auroc':old['image_auroc'],'recall':old['recall'],'false_positive_rate':old['false_positive_rate'],
                        'confusion':old['confusion'],'dice':old['pooled_pixel_dice_all_200_images'],
                        'normal_pixel_fpr':old_diag['normal_diagnostics']['pooled_pixel_fpr']},
            'joint_segmentation_alarm':{'image_auroc':new['image_auroc'],'confusion':new['confusion']},
            'joint_head':{key:cls[key] for key in ['image_auroc','threshold','recall','false_positive_rate','confusion']},
            'joint_masks':{'dice':new['pooled_pixel_dice_all_200_images'],'iou':new['pooled_pixel_iou_all_200_images'],
                           'normal_pixel_fpr':new['normal_masks']['normal_pixel_fpr'],
                           'normal_nonempty_of_100':new['normal_masks']['nonempty_of_100'],
                           'pixel_threshold':new['pixel_threshold'],'pixel_recall':raw_tp/gt},
            'gated_masks_exploratory':{**cls['gated_masks_exploratory'],'pixel_recall':(raw_tp-lost)/gt,
                                        'fraction_raw_tp_removed':lost/max(1,raw_tp)},
            'paired_deltas':{'image_auroc':cls['image_auroc']-old['image_auroc'],
                             'recall':cls['recall']-old['recall'],
                             'false_positive_rate':cls['false_positive_rate']-old['false_positive_rate'],
                             'dice':new['pooled_pixel_dice_all_200_images']-old['pooled_pixel_dice_all_200_images']},
            'per_class_overlapping_groups':per_class,'by_gt_area':by_area,'paired_transitions':transitions}
        alarm_matrices[str(seed)]={'baseline':[int(r['prediction']) for r in old_preds],
                                  'joint_head':[int(r['classification_alarm']) for r in new_preds]}
        del model,saved
    sets={k:set(v) for k,v in common_selection.items()}
    assert len(sets['train'])==4000 and all(rows[i]['split']=='train' for i in sets['train'])
    for k in ['checkpoint_validation','calibration','evaluation']:
        assert all(rows[i]['split']=='val' for i in sets[k])
    keys=list(sets)
    assert all(not sets[a]&sets[b] for j,a in enumerate(keys) for b in keys[j+1:])
    assert len(sets['evaluation'])==200 and sum(rows[i]['label'] for i in sets['evaluation'])==100
    assert len(sets['checkpoint_validation'])==80
    assert len(sets['calibration'])==200 and all(not rows[i]['label'] for i in sets['calibration'])
    stability={}
    for method in ['baseline','joint_head']:
        counts=np.array([alarm_matrices[str(s)][method] for s in SEEDS]).sum(axis=0)
        stability[method]={title:{f'alarmed_in_{k}_of_3_seeds':int(sum(counts[i]==k for i,n in enumerate(common_selection['evaluation']) if rows[n]['label']==label))
                                  for k in range(4)} for label,title in [(0,'normal'),(1,'defective')]}
        common_misses=[r for r in pixel_rows if int(r['label'])==1 and counts[common_selection['evaluation'].index(r['image_id'])]==0]
        stability[method]['persistent_miss_by_area']={g:sum(r['area_group']==g for r in common_misses)
                                                     for g in ['0.1_to_1pct','1_to_5pct','at_least_5pct']}
    aggregate={}
    for group,fields in [('baseline',['image_auroc','recall','false_positive_rate','dice','normal_pixel_fpr']),
                         ('joint_head',['image_auroc','recall','false_positive_rate']),
                         ('joint_masks',['dice','iou','normal_pixel_fpr','pixel_recall']),
                         ('paired_deltas',['image_auroc','recall','false_positive_rate','dice'])]:
        aggregate[group]={k:descriptive([per_seed[str(s)][group][k] for s in SEEDS]) for k in fields}
    result={'scope':'Three predefined paired seeds on reused development images, NOT final test',
            'seeds':list(SEEDS),'classification_weight':.5,'unique_evaluation_images':200,
            'test_images_used':0,'controls_verified':True,
            'control_scope':'Historical segmentation baselines reused for seeds 43/44; exact joint-architecture weight-zero control exists only for seed 42. Initial segmentation states and sampling counts audited for all seeds.',
            'selection_policy':'Checkpoint development Dice, normal calibration q95, no best-seed selection; raw masks ungated',
            'per_seed':per_seed,'aggregate':aggregate,'cross_seed_alarm_stability':stability,
            'limitations':['All runs reuse 200 development images, not 600 independent observations.',
                           'Seed 42 motivated this replication and is not a new independent confirmation.',
                           'Seed SD describes training variation, not population uncertainty or significance.',
                           'Matched attempted-update budgets can have different AMP skipped steps.',
                           'Class groups overlap; small subgroup estimates are uncertain.']}
    path=ROOT/'reports/steel_joint_multiseed.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'aggregate':aggregate,'cross_seed_alarm_stability':stability},indent=2))


if __name__=='__main__':main()
