"""Verify matched experimental controls and summarize development-only U-Net runs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    names=['steel_unet','steel_unet_balanced','steel_unet_balanced_random_resnet18',
           'steel_unet_balanced_pretrained_resnet18']
    metrics={n:json.loads((ROOT/f'reports/{n}_metrics.json').read_text()) for n in names}
    selected={n:json.loads((ROOT/f'configs/{n}_selection.json').read_text()) for n in names}
    for name in names[1:]:
        for key in ['checkpoint_validation','calibration','evaluation']:
            assert selected[name][key]==selected[names[0]][key]
        for key in ['loss','optimizer','total_updates','sampling','rounds','checkpoint_validation_images']:
            assert metrics[name][key]==metrics[names[0]][key]
    assert selected[names[1]]==selected[names[2]]==selected[names[3]]
    for key in ['parameters','actual_sampled_tiles_per_class','positive_tiles_per_class','training_negative_tiles']:
        assert metrics[names[2]][key]==metrics[names[3]][key]
    comparison={'scope':'Single-seed short-budget validation experiments; no test result',
                'controls':{'validation_identities_equal':True,'loss_optimizer_and_budget_equal':True,
                            'balanced_variants_training_identities_equal':True,
                            'encoder_pair_architecture_and_tile_class_draws_equal':True},
                'results':{n:{k:metrics[n][k] for k in ['parameters','best_round','image_auroc','confusion',
                    'pooled_pixel_dice_all_200_images','pooled_pixel_iou_all_200_images',
                    'mean_per_defect_image_pixel_auroc','peak_torch_allocated_mib','wall_seconds']} for n in names}}
    text=json.dumps(comparison,indent=2)
    (ROOT/'reports/steel_unet_comparison.json').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
