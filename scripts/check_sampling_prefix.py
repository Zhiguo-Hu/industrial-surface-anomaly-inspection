"""Confirm sampler integration preserves historical uniform first-80-step training."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='steel_unet_balanced_pretrained_resnet18_expanded'


def main():
    old=json.loads((ROOT/f'reports/{BASE}_r30_det_s42_cls0p5_metrics.json').read_text())
    new=json.loads((ROOT/f'reports/{BASE}_r1_det_s42_cls0p5_metrics.json').read_text())
    checks={key:new[key]==old[key] for key in ['initial_state_sha256','first_80_steps_state_sha256']}
    checks['first_round_history_identical']=new['history'][0]==old['history'][0]
    checks['negative_flip_reference_verified']=new['negative_and_flip_sequence_matches_uniform_reference']
    assert all(checks.values()),checks
    (ROOT/'reports/steel_sampling_prefix_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks))


if __name__=='__main__':main()
