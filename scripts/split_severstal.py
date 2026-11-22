"""Create a frozen, image-level 70/15/15 Severstal split without moving files."""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'val', 'test')


def allocate(count):
    """Largest-remainder allocation; tiny strata need not appear in every split."""
    exact = [count * r for r in (0.7, 0.15, 0.15)]
    counts = [int(x) for x in exact]
    order = sorted(range(3), key=lambda i: (-(exact[i]-counts[i]), i))
    for i in order[:count-sum(counts)]:
        counts[i] += 1
    return counts


def assign(rows, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row['classes'])].append(row)
    generator = random.Random(seed)
    assigned = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda r: r['image_id'])
        generator.shuffle(members)
        offset = 0
        for split, count in zip(SPLITS, allocate(len(members))):
            for row in members[offset:offset+count]:
                assigned.append({**row, 'split': split,
                                 'use_for_memory_bank': split == 'train' and not row['classes']})
            offset += count
    return sorted(assigned, key=lambda r: r['image_id'])


def main():
    folder = ROOT/'data/severstal'
    image_dir = folder/'train_images' if (folder/'train_images').is_dir() else folder
    annotation = folder/'train.csv'
    labels = defaultdict(set)
    pairs = set()
    with annotation.open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            name, category = row['ImageId'], int(row['ClassId'])
            if Path(name).name != name or category not in (1,2,3,4):
                raise ValueError('Invalid annotation identity')
            if (name, category) in pairs:
                raise ValueError('Duplicate image/class annotation')
            pairs.add((name,category))
            if row['EncodedPixels'].strip():
                labels[name].add(category)
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if not paths:
        raise ValueError('No images')
    names = {p.name for p in paths}
    if {name for name,_ in pairs} - names:
        raise ValueError('CSV references missing images')
    rows = []
    seen = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f'Duplicate image content: {seen[digest]}, {path.name}; group before splitting')
        seen[digest] = path.name
        classes = sorted(labels[path.name])
        rows.append({'image_id': path.name, 'image': path.relative_to(ROOT).as_posix(),
                     'classes': classes, 'label': int(bool(classes)), 'sha256': digest})
    samples = assign(rows)
    if samples != assign(list(reversed(rows))):
        raise RuntimeError('Split depends on input enumeration order')
    sets = {s: {r['sha256'] for r in samples if r['split']==s} for s in SPLITS}
    if any(sets[a] & sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise RuntimeError('Content leakage')
    if len(samples) != len(rows) or len({r['image_id'] for r in samples}) != len(rows):
        raise RuntimeError('Incomplete or duplicate assignment')
    totals = {}
    strata = defaultdict(Counter)
    for split in SPLITS:
        subset = [r for r in samples if r['split']==split]
        totals[split] = {'total': len(subset), 'normal': sum(r['label']==0 for r in subset),
                         'defect': sum(r['label']==1 for r in subset),
                         'class_counts': {str(c):sum(c in r['classes'] for r in subset) for c in (1,2,3,4)}}
    for row in samples:
        strata[','.join(map(str,row['classes'])) or 'normal'][row['split']] += 1
    report = {'seed':42, 'target_ratios':{'train':0.7,'val':0.15,'test':0.15},
              'method':'Stratify by exact class combination (including normal); largest remainder per stratum',
              'counts':totals, 'strata':dict(strata),
              'rare_strata':{k:dict(v) for k,v in strata.items() if sum(v.values())<7},
              'checks':{'all_images_assigned_once':True,'content_hash_sets_disjoint':True,
                        'reproducible_independent_of_input_order':True},
              'limitations':'No production batch IDs supplied; exact-file duplicate check does not rule out near duplicates or same-batch dependence.',
              'source_csv_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),
              'original_images_moved_or_modified':False}
    manifest = {'dataset':'severstal', 'annotation_csv':annotation.relative_to(ROOT).as_posix(),
                'seed':42,'source_csv_sha256':report['source_csv_sha256'],
                'split_policy':report['method'],'samples':samples}
    text = json.dumps(manifest, indent=2)
    destination = ROOT/'configs/severstal_manifest.json'
    # Freeze existing split: refuse to overwrite if data or assignment changed.
    if destination.exists() and destination.read_text(encoding='utf-8') != text:
        raise ValueError('Frozen manifest differs; inspect changes before deliberately creating a new split version')
    destination.write_text(text,encoding='utf-8')
    for split in SPLITS:
        with (ROOT/f'configs/severstal_{split}.csv').open('w',newline='',encoding='utf-8-sig') as file:
            writer=csv.DictWriter(file,fieldnames=['image_id','image','label','classes','sha256','use_for_memory_bank'])
            writer.writeheader()
            for row in samples:
                if row['split']==split:
                    writer.writerow({k: ';'.join(map(str,row[k])) if k=='classes' else row[k] for k in writer.fieldnames})
    report['manifest_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (ROOT/'reports/severstal_split_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
