"""Deterministic class-coverage intervention, keeping the parent-image budget fixed."""
import random
from collections import Counter


def coverage_balanced_rows(all_rows, baseline, minimum=60, positive_budget=250, seed=42):
    if minimum<1 or positive_budget<1:raise ValueError('Invalid coverage budget')
    normals=[r for r in baseline if r['label']==0]
    if len({r['image_id'] for r in baseline})!=len(baseline):raise ValueError('Duplicate baseline identity')
    if any(r['split']!='train' for r in baseline):raise ValueError('Baseline must be train-only')
    preferred={r['image_id'] for r in baseline if r['label']}
    candidates=sorted([r for r in all_rows if r['split']=='train' and r['label']],key=lambda r:r['image_id'])
    rng=random.Random(seed);rng.shuffle(candidates)
    # Preserve original positives where possible, without exceeding the fixed budget.
    candidates.sort(key=lambda r:r['image_id'] not in preferred)
    availability=Counter(c for r in candidates for c in r['classes'])
    if any(availability[c]<minimum for c in (1,2,3,4)):raise ValueError('Insufficient class availability')
    selected={};counts=Counter()
    for category in sorted((1,2,3,4),key=lambda c:(availability[c],c)):
        for row in candidates:
            if counts[category]>=minimum:break
            if category in row['classes'] and row['image_id'] not in selected:
                selected[row['image_id']]=row;counts.update(row['classes'])
    if len(selected)>positive_budget:raise ValueError('Coverage quotas exceed parent-image budget')
    for row in candidates:
        if len(selected)>=positive_budget:break
        if row['image_id'] not in selected:selected[row['image_id']]=row
    if len(selected)!=positive_budget:raise ValueError('Insufficient positive images')
    return sorted(normals+list(selected.values()),key=lambda r:r['image_id'])


def expanded_training_rows(all_rows, baseline, per_label=2000, minimum=120, seed=42):
    if per_label<sum(r['label']==0 for r in baseline):raise ValueError('Expansion cannot shrink normal pool')
    normals=[r for r in baseline if not r['label']]
    existing={r['image_id'] for r in baseline}
    extras=sorted([r for r in all_rows if r['split']=='train' and not r['label'] and r['image_id'] not in existing],
                  key=lambda r:r['image_id'])
    random.Random(seed).shuffle(extras)
    needed=per_label-len(normals)
    if len(extras)<needed:raise ValueError('Insufficient normal training images')
    preferred=normals+extras[:needed]+[r for r in baseline if r['label']]
    result=coverage_balanced_rows(all_rows,preferred,minimum=minimum,positive_budget=per_label,seed=seed)
    if not existing.issubset({r['image_id'] for r in result}):raise ValueError('Expansion lost baseline identities')
    return result
