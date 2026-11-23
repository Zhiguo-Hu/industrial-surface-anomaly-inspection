"""Paired uniform/class-balanced joint-model seeds; no best-seed promotion."""
import csv,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.compare_tile_sampling import read,sampling_replay
from scripts.train_steel_unet import predict_outputs
from scripts.run_steel_pilot import read_image
from src.reproducibility import configure,state_hash
from src.resnet_unet import JointResNet18UNet

BASE='steel_unet_balanced_pretrained_resnet18_expanded_r30_det_s'


def main():
    manifest=read('configs/severstal_manifest.json');rows={r['image_id']:r for r in manifest['samples']}
    reference_selection=None;reference_index=None;result={}
    configure(42,True);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    for seed in (42,43,44):
        old_name=f'{BASE}{seed}_cls0p5';name=old_name+'_tile_class_balanced'
        a=read(f'reports/{old_name}_metrics.json');b=read(f'reports/{name}_metrics.json')
        selection=read(f'configs/{name}_selection.json');assert selection==read(f'configs/{old_name}_selection.json')
        index=read(f'runs/{name}/tile_sampling_index.json')
        if reference_selection is None:reference_selection=selection;reference_index=index
        assert selection==reference_selection and index==reference_index
        for k in ['initial_state_sha256','parameters','total_updates','optimizer','classification_weight','reproducibility',
                  'training_positive_tiles','training_negative_tiles','train_images']:assert a[k]==b[k],(seed,k)
        assert b['tile_sampling']['policy']=='class-balanced' and b['total_updates']==2400
        uniform=sampling_replay(index,'uniform',seed);balanced=sampling_replay(index,'class-balanced',seed)
        assert uniform['actual_sampled_tiles_per_class']==a['actual_sampled_tiles_per_class']
        assert balanced['actual_sampled_tiles_per_class']==b['actual_sampled_tiles_per_class']
        assert balanced['negative_and_flip_sequence_sha256']==uniform['negative_and_flip_sequence_sha256']==b['negative_and_flip_sequence_sha256']
        seed_result={}
        for method,experiment,m in [('uniform',old_name,a),('class-balanced',name,b)]:
            with (ROOT/f'runs/{experiment}/validation_predictions.csv').open(encoding='utf-8-sig',newline='') as f:pred=list(csv.DictReader(f))
            with (ROOT/f'runs/{experiment}_diagnosis/per_image.csv').open(encoding='utf-8-sig',newline='') as f:pixels=list(csv.DictReader(f))
            assert [r['image_id'] for r in pred]==[r['image_id'] for r in pixels]==selection['evaluation']
            cls=m['classification_metrics'];labels=[rows[r['image_id']]['label'] for r in pred]
            alarms={r['image_id']:int(r['classification_alarm']) for r in pred}
            assert all(alarms[r['image_id']]==int(float(r['classification_score'])>cls['threshold']) for r in pred)
            values=confusion_matrix(labels,[alarms[r['image_id']] for r in pred],labels=[0,1]).ravel()
            assert dict(zip(['tn','fp','fn','tp'],map(int,values)))==cls['confusion']
            np.testing.assert_allclose(roc_auc_score(labels,[float(r['classification_score']) for r in pred]),cls['image_auroc'])
            diag=read(f'reports/{experiment}_diagnosis.json')
            seed_result[method]={'head_auroc':cls['image_auroc'],'head_confusion':cls['confusion'],
                'head_recall':cls['recall'],'head_fpr':cls['false_positive_rate'],'raw_dice':m['pooled_pixel_dice_all_200_images'],
                'normal_pixel_fpr':m['normal_masks']['normal_pixel_fpr'],'best_round':m['best_round'],
                'completed_updates':m['completed_optimizer_steps'],'amp_skips':m['optimizer_steps_skipped_by_amp'],
                'small_area_head_tp':sum(alarms[r['image_id']] for r in pixels if r['area_group']=='0.1_to_1pct'),
                'class2_head_tp':sum(alarms[r['image_id']] for r in pred if 2 in rows[r['image_id']]['classes']),
                'class2_pixel_recall':diag['per_class']['2']['class_specific_pixel_recall']}
        saved=torch.load(ROOT/f'runs/{name}/best.pt',map_location=device,weights_only=True)
        assert state_hash(saved['model'])==b['best_state_sha256']
        model=JointResNet18UNet().to(device);model.load_state_dict(saved['model']);model.eval()
        with (ROOT/f'runs/{name}/validation_predictions.csv').open(encoding='utf-8-sig') as f:first=next(csv.DictReader(f))
        heat,score=predict_outputs(model,read_image(rows[first['image_id']]),device)
        np.testing.assert_allclose([float(heat.max()),score],[float(first['score']),float(first['classification_score'])],rtol=1e-5,atol=1e-5)
        seed_result['reload_check']='passed';seed_result['exposure']=balanced;result[str(seed)]=seed_result
        del saved,model
    assert all(rows[i]['split']=='train' for i in reference_selection['train'])
    for k in ['checkpoint_validation','calibration','evaluation']:assert all(rows[i]['split']=='val' for i in reference_selection[k])
    groups=[set(v) for v in reference_selection.values()];assert all(not a&b for j,a in enumerate(groups) for b in groups[j+1:])
    aggregate={}
    for method in ['uniform','class-balanced']:
        aggregate[method]={}
        for key in ['head_auroc','head_recall','head_fpr','raw_dice','normal_pixel_fpr','small_area_head_tp','class2_head_tp','class2_pixel_recall']:
            values=[result[str(s)][method][key] for s in (42,43,44)]
            aggregate[method][key]={'mean':float(np.mean(values)),'sample_std_ddof1':float(np.std(values,ddof=1)),
                                    'min':min(values),'max':max(values)}
    report={'scope':'Paired joint-model positive sampling replication, reused development images; no test images',
            'seeds':[42,43,44],'unique_evaluation_images':200,'controls_verified':True,
            'bank_encoder_policy':'Fixed class-balanced seed42 encoder selected before these replication results; no best-seed selection',
            'per_seed':result,'aggregate':aggregate,
            'limits':['Seed42 motivated replication; seeds reuse 200 images, not 600 independent observations.',
                      'SD is training-seed variation, not population uncertainty or significance.',
                      'Class2 has only 6 evaluation images; small-area group has 26.',
                      'Budgets match 2400 attempts, with separately recorded AMP skips.']}
    (ROOT/'reports/steel_balanced_multiseed.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
