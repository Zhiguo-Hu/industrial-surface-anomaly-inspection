"""Compare fixed-model data expansion and training-budget experiments."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE='steel_unet_balanced_pretrained_resnet18'


def main():
    names=[BASE,BASE+'_expanded',BASE+'_expanded_r30']
    m={n:json.loads((ROOT/f'reports/{n}_metrics.json').read_text()) for n in names}
    s={n:json.loads((ROOT/f'configs/{n}_selection.json').read_text()) for n in names}
    for n in names[1:]:
        for key in ['checkpoint_validation','calibration','evaluation']:assert s[n][key]==s[BASE][key]
        for key in ['parameters','architecture','loss','optimizer','sampling']:assert m[n][key]==m[BASE][key]
    assert set(s[BASE]['train'])<=set(s[names[1]]['train'])
    assert s[names[1]]['train']==s[names[2]]['train']
    assert m[BASE]['total_updates']==m[names[1]]['total_updates']==480
    assert m[names[2]]['total_updates']==2400
    prefix=np.allclose([[h['training_loss'],h['checkpoint_validation_pooled_dice_at_05']] for h in m[names[1]]['history']],
                       [[h['training_loss'],h['checkpoint_validation_pooled_dice_at_05']] for h in m[names[2]]['history'][:6]],
                       rtol=1e-4,atol=1e-4)
    report={'scope':'Development-only scale/budget comparisons; no test images',
            'checks':{'validation_identities_and_training_rules_equal':True,'expanded_parent_pool_contains_original_pool':True,
                      'expanded_runs_train_identities_equal':True,'expanded_first_480_steps_history_agrees':bool(prefix)},
            'results':{n:{k:m[n][k] for k in ['train_images','total_updates','best_round','image_auroc','confusion',
                        'pooled_pixel_dice_all_200_images','pooled_pixel_iou_all_200_images',
                        'peak_torch_allocated_mib','wall_seconds']} for n in names},
            'skipped_amp_steps':{n:m[n].get('optimizer_steps_skipped_by_amp','not instrumented in historical run') for n in names},
            'limitations':['One seed, repeatedly inspected development evaluation set.',
                           'Expanded-pool experiment changes coverage including class quotas; no independent attribution to each coverage component.',
                           'Longer run has more checkpoint-selection opportunities; no final test claim.']}
    text=json.dumps(report,indent=2)
    (ROOT/'reports/steel_training_scale_comparison.json').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
