"""Audit positive-tile sampling interventions on fixed development images."""
import csv,hashlib,json,os,random,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix,roc_auc_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.tile_sampling import ClassAwareSampler,SMALL_MAX_PIXELS,update_negative_flip_hash
from src.reproducibility import configure,state_hash
from src.resnet_unet import JointResNet18UNet
from scripts.train_steel_unet import predict_outputs
from scripts.run_steel_pilot import read_image

BASE='steel_unet_balanced_pretrained_resnet18_expanded_r30_det_s42_cls0p5'
NAMES={'uniform':BASE,'class-balanced':BASE+'_tile_class_balanced','class-small':BASE+'_tile_class_small'}


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))


def sampling_replay(index,policy):
    classes={r['index']:r['classes'] for r in index['positive']}
    pixels={r['index']:r['pixels'] for r in index['positive']}
    positive=sorted(classes);negative=[i for i in range(index['total_tiles']) if i not in classes]
    all_classes={i:classes.get(i,[]) for i in range(index['total_tiles'])}
    all_pixels={i:pixels.get(i,0) for i in range(index['total_tiles'])}
    sampler=ClassAwareSampler(all_classes,all_pixels,policy,42) if policy!='uniform' else None
    rng=random.Random(42);digest=hashlib.sha256();counts=Counter();drawn=set();small=0
    for _ in range(2400):
        uniform=rng.choices(positive,k=4);neg=rng.choices(negative,k=4)
        pos=sampler.draw() if sampler else uniform
        update_negative_flip_hash(digest,neg,rng.random()<.5,rng.random()<.5)
        counts.update(c for i in pos for c in classes[i]);drawn.update(pos)
        small+=sum(pixels[i]<=SMALL_MAX_PIXELS for i in pos)
    return {'actual_sampled_tiles_per_class':{str(c):counts[c] for c in (1,2,3,4)},
            'small_draws':small,'positive_draws':9600,'unique_tiles':len(drawn),
            'unique_parents':len({i//index['tiles_per_parent'] for i in drawn}),
            'negative_and_flip_sequence_sha256':digest.hexdigest()}


def main():
    checks=read('reports/steel_sampling_prefix_checks.json');assert all(checks.values())
    protocol=read('configs/steel_tile_sampling_protocol.json')
    assert protocol['small_tile_union_foreground_pixels']==[1,SMALL_MAX_PIXELS]
    index=read(f'runs/{NAMES["class-balanced"]}/tile_sampling_index.json')
    assert index==read(f'runs/{NAMES["class-small"]}/tile_sampling_index.json')
    baseline=read(f'reports/{BASE}_metrics.json');selection=read(f'configs/{BASE}_selection.json')
    manifest=read('configs/severstal_manifest.json');rows={r['image_id']:r for r in manifest['samples']}
    assert all(rows[i]['split']=='train' for i in selection['train'])
    for k in ['checkpoint_validation','calibration','evaluation']:assert all(rows[i]['split']=='val' for i in selection[k])
    identities=[set(v) for v in selection.values()]
    assert all(not a&b for j,a in enumerate(identities) for b in identities[j+1:])
    configure(42,True);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    results={};expected_pixel_rows=None;expected_digest=None
    for policy,name in NAMES.items():
        m=read(f'reports/{name}_metrics.json');d=read(f'reports/{name}_diagnosis.json')
        assert selection==read(f'configs/{name}_selection.json')
        for key in ['initial_state_sha256','parameters','total_updates','optimizer','classification_weight','reproducibility',
                    'training_positive_tiles','training_negative_tiles','train_images']:
            assert m[key]==baseline[key],(policy,key)
        assert m['total_updates']==2400 and m['classification_weight']==.5
        sampled=sampling_replay(index,policy)
        assert sampled['actual_sampled_tiles_per_class']==m['actual_sampled_tiles_per_class']
        if policy!='uniform':
            assert m['tile_sampling']['policy']==policy and m['negative_and_flip_sequence_matches_uniform_reference']
            assert all(m['actual_sampled_tiles_per_class'][str(c)]>=2400 for c in (1,2,3,4))
            if policy=='class-small':
                guard=m['tile_sampling']['small_pool_guard']
                assert guard['minimum_small_tiles']==protocol['minimum_small_tiles_per_class']
                assert guard['minimum_small_parents']==protocol['minimum_small_parent_images_per_class']
                assert not m['tile_sampling']['per_class']['2']['small_bias_eligible']
            for a,b in [('small_draws','actual_small_positive_tile_draws'),('unique_tiles','unique_positive_tiles_drawn'),
                        ('unique_parents','unique_positive_parents_drawn'),('negative_and_flip_sequence_sha256','negative_and_flip_sequence_sha256')]:
                assert sampled[a]==m[b]
        if expected_digest is None:expected_digest=sampled['negative_and_flip_sequence_sha256']
        assert sampled['negative_and_flip_sequence_sha256']==expected_digest
        with (ROOT/f'runs/{name}/validation_predictions.csv').open(encoding='utf-8-sig',newline='') as f:pred=list(csv.DictReader(f))
        with (ROOT/f'runs/{name}_diagnosis/per_image.csv').open(encoding='utf-8-sig',newline='') as f:pixels=list(csv.DictReader(f))
        assert [r['image_id'] for r in pred]==[r['image_id'] for r in pixels]==selection['evaluation']
        if expected_pixel_rows is None:expected_pixel_rows=pixels
        for a,b in zip(expected_pixel_rows,pixels):
            assert a['gt_pixels']==b['gt_pixels'] and a['area_group']==b['area_group']
        cls=m['classification_metrics'];labels=[int(r['label']) for r in pred]
        alarms={r['image_id']:int(r['classification_alarm']) for r in pred}
        for r in pred:
            assert int(r['label'])==rows[r['image_id']]['label']
            assert int(r['classification_alarm'])==int(float(r['classification_score'])>cls['threshold'])
        counts=confusion_matrix(labels,[alarms[r['image_id']] for r in pred],labels=[0,1]).ravel()
        assert cls['confusion']==dict(zip(['tn','fp','fn','tp'],map(int,counts)))
        np.testing.assert_allclose(roc_auc_score(labels,[float(r['classification_score']) for r in pred]),cls['image_auroc'])
        saved=torch.load(ROOT/f'runs/{name}/best.pt',map_location=device,weights_only=True)
        assert state_hash(saved['model'])==m['best_state_sha256']
        assert saved['manifest_sha256']==hashlib.sha256((ROOT/'configs/severstal_manifest.json').read_bytes()).hexdigest()
        model=JointResNet18UNet().to(device);model.load_state_dict(saved['model']);model.eval()
        heat,score=predict_outputs(model,read_image(rows[pred[0]['image_id']]),device)
        np.testing.assert_allclose([float(heat.max()),score],[float(pred[0]['score']),float(pred[0]['classification_score'])],rtol=1e-5,atol=1e-5)
        results[policy]={'sampling_replay_verified':True,'sampling_exposure':sampled,
            'sampler_inventory':m.get('tile_sampling',{'policy':'uniform'}),'best_round':m['best_round'],
            'completed_updates':m['completed_optimizer_steps'],'amp_skips':m['optimizer_steps_skipped_by_amp'],
            'image_head':{k:cls[k] for k in ['image_auroc','confusion','recall','false_positive_rate','threshold']},
            'head_score_distribution':{'min':min(float(r['classification_score']) for r in pred),
                                       'max':max(float(r['classification_score']) for r in pred),
                                       'std':float(np.std([float(r['classification_score']) for r in pred]))},
            'final_round_classification_loss':m['history'][-1]['classification_loss'],
            'raw_masks':{'dice':m['pooled_pixel_dice_all_200_images'],'iou':m['pooled_pixel_iou_all_200_images'],
                         'pixel_threshold':m['pixel_threshold'],**m['normal_masks']},
            'by_image_gt_area':{g:{'images':len(sub),'head_detected':sum(alarms[r['image_id']] for r in sub),
                                    'raw_pixel_recall':sum(int(r['tp_pixels']) for r in sub)/max(1,sum(int(r['gt_pixels']) for r in sub))}
                                for g in ['0.1_to_1pct','1_to_5pct','at_least_5pct'] for sub in [[r for r in pixels if r['area_group']==g]]},
            'per_class_overlapping_groups':{str(c):{'images':sum(c in rows[r['image_id']]['classes'] for r in pred),
                                                    'head_detected':sum(alarms[r['image_id']] for r in pred if c in rows[r['image_id']]['classes']),
                                                    'class_pixel_recall':d['per_class'][str(c)]['class_specific_pixel_recall']} for c in (1,2,3,4)},
            'reload_one_image':'passed, scores at tolerance 1e-5'}
        del saved,model
    result={'scope':'Seed42 fixed joint model; sequential class balance then small-tile preference interventions; no test images',
            'prefix_checks':checks,'controls_verified':True,'unique_evaluation_images':200,
            'small_tile_definition':'1..655 union-mask pixels in a 256x256 training tile; differs from image GT area groups',
            'historical_baseline_scope':'Uniform baseline reused; only its first 80 steps were replayed after integration, not a new full 2400-step run.',
            'results':results,'limitations':['Single predeclared seed, reused development evaluation, not independent final testing.',
                                            'Two sequential interventions, not a full factorial study or separate small-only arm.',
                                            'AMP skipped steps may differ; budgets match at 2400 attempts.',
                                            'All default masks are ungated; no threshold or checkpoint selection using evaluation outcomes.']}
    (ROOT/'reports/steel_tile_sampling_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
