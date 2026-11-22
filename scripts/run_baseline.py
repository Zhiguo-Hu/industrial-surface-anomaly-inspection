"""Build, calibrate and evaluate the frozen global-feature baseline, or infer an image."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import (ImageDataset, extract, load_model, nearest_scores,
                          validate_manifest, validation_threshold)


def distribution_plot(val_scores, test_scores, labels, threshold, output):
    canvas = Image.new('RGB', (1000, 520), 'white')
    draw = ImageDraw.Draw(canvas)
    series = [('Validation normal', val_scores, '#3478c5'),
              ('Test normal', test_scores[labels == 0], '#26965c'),
              ('Test anomaly', test_scores[labels == 1], '#d24b46')]
    all_scores = np.concatenate([s for _, s, _ in series])
    low, high = float(all_scores.min()) * 0.9, float(all_scores.max()) * 1.05
    bins = np.linspace(low, high, 21)
    # Each curve is a within-group fraction, so unequal group sizes do not dominate.
    fractions = [np.histogram(s, bins)[0] / len(s) for _, s, _ in series]
    ymax = max(float(f.max()) for f in fractions) * 1.15
    px = lambda x: 90 + (x-low)/(high-low)*850
    py = lambda y: 430-y/ymax*300
    draw.text((90, 16), 'BOTTLE: anomaly-score distribution (larger = more anomalous)', fill='black')
    for i, ((name, _, color), values) in enumerate(zip(series, fractions)):
        draw.text((90+i*280, 45), name, fill=color)
        points = [(px((bins[j]+bins[j+1])/2), py(value)) for j, value in enumerate(values)]
        draw.line(points, fill=color, width=3)
    draw.line([(90, 130), (90, 430), (940, 430)], fill='black', width=2)
    draw.line([(px(threshold), 110), (px(threshold), 430)], fill='black', width=2)
    draw.text((px(threshold)+5, 92), f'val-only threshold={threshold:.4f}', fill='black')
    for i in range(6):
        x = low+(high-low)*i/5
        draw.text((px(x)-15, 443), f'{x:.3f}', fill='black')
        y = ymax*i/5
        draw.text((35, py(y)), f'{y:.2f}', fill='black')
    draw.text((360, 480), 'Nearest-neighbor L2 distance of unit-normalized global features', fill='black')
    canvas.save(output)


def error_preview(rows, scores, prediction, output):
    wrong = [i for i, r in enumerate(rows) if prediction[i] != r['label']]
    if not wrong:
        return 0
    # Deterministic selection: show at most three false positives and three false negatives.
    selected = ([i for i in wrong if rows[i]['label'] == 0][:3] +
                [i for i in wrong if rows[i]['label'] == 1][:3])
    canvas = Image.new('RGB', (280*len(selected), 335), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, i in enumerate(selected):
        row = rows[i]
        with Image.open(ROOT / row['image']) as image:
            image = image.convert('RGB')
        image.thumbnail((270,270))
        canvas.paste(image, (col*280, 60))
        draw.text((col*280+5, 5), f"{'False positive' if row['label']==0 else 'False negative'} | score={scores[i]:.4f}", fill='black')
        draw.text((col*280+5, 25), f"{row['defect_type']}/{Path(row['image']).name}", fill='black')
    canvas.save(output)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image', type=Path, help='Inference only; use an existing saved bank/threshold')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'runs/global_baseline')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = load_model(ROOT/'weights/resnet18-f37072fd.pth', args.device)
    output = args.output_dir
    if args.image:
        if not args.image.is_file():
            parser.error(f'Image not found: {args.image}')
        artifact = torch.load(output/'bank.pt', map_location='cpu', weights_only=True)
        row = {'image': str(args.image.resolve())}
        loader = DataLoader(ImageDataset(ROOT, [row], artifact['input_size']), batch_size=1)
        features = extract(model, loader, args.device)
        scores, index = nearest_scores(features, artifact['features'])
        result = {'image': row['image'], 'score': float(scores[0]), 'threshold': artifact['threshold'],
                  'prediction': 'anomaly' if scores[0] > artifact['threshold'] else 'normal',
                  'nearest_normal_image': artifact['train_images'][int(index[0])],
                  'note': 'Calibrated for MVTec bottle images; arbitrary photos may be out of distribution.'}
        output.mkdir(parents=True, exist_ok=True)
        (output/'single_prediction.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))
        return

    manifest_path = ROOT/'configs/bottle_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = manifest['samples']
    validate_manifest(rows)
    for row in rows:
        path = ROOT/row['image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Image changed since audit: {path}')
    subsets = {s: [r for r in rows if r['split'] == s] for s in ['train','val','test']}
    size = 256
    features = {}
    elapsed = {}
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True, exist_ok=True)
    # Threshold is fixed and saved BEFORE extracting any test features.
    for split in ['train','val']:
        start = time.perf_counter()
        loader = DataLoader(ImageDataset(ROOT, subsets[split], size), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
        features[split] = extract(model, loader, args.device)
        elapsed[split] = time.perf_counter()-start
        print(f'{split}: {list(features[split].shape)}, {elapsed[split]:.2f}s', flush=True)
    val_scores, _ = nearest_scores(features['val'], features['train'])
    threshold = validation_threshold(val_scores)
    artifact = {'features': features['train'], 'threshold': threshold, 'input_size': size,
                'feature_normalization': 'L2 unit normalization', 'score': 'nearest-neighbor Euclidean distance',
                'weights': 'resnet18-f37072fd.pth', 'quantile': 0.95,
                'train_images': [r['image'] for r in subsets['train']],
                'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    torch.save(artifact, output/'bank.pt')
    print(f'Bank saved; val-only threshold={threshold:.6f}', flush=True)
    start = time.perf_counter()
    features['test'] = extract(model, DataLoader(ImageDataset(ROOT, subsets['test'], size),
                               batch_size=args.batch_size, shuffle=False, num_workers=0), args.device)
    elapsed['test'] = time.perf_counter()-start
    test_scores, indices = nearest_scores(features['test'], features['train'])
    labels = np.array([r['label'] for r in subsets['test']])
    predicted = (test_scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0,1]).ravel()
    metrics = {'method': 'Frozen ResNet18 global features + L2 normalization + nearest neighbor',
               'input_size': size, 'batch_size': args.batch_size, 'device': args.device,
               'counts': {s: len(v) for s,v in subsets.items()},
               'feature_bank_shape': list(features['train'].shape),
               'threshold': threshold, 'threshold_policy': 'Normal val 95th percentile, linear; predict anomaly iff score > threshold',
               'validation_false_positive_rate': float(np.mean(val_scores > threshold)),
               'test_image_auroc': float(roc_auc_score(labels, test_scores)),
               'test_confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
               'test_anomaly_recall': float(tp/(tp+fn)), 'test_false_positive_rate': float(fp/(fp+tn)),
               'test_accuracy': float(np.mean(predicted == labels)),
               'extraction_wall_seconds': elapsed,
               'timing_note': 'End-to-end feature extraction incl. image I/O and cold start; not steady-state latency',
               'peak_torch_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else None,
               'bank_file_bytes': (output/'bank.pt').stat().st_size,
               'manifest_sha256': artifact['manifest_sha256'],
               'gpu': torch.cuda.get_device_name(0) if args.device=='cuda' else None}
    metrics['per_defect_recall'] = {d: float(predicted[[i for i,r in enumerate(subsets['test']) if r['defect_type']==d]].mean())
        for d in sorted({r['defect_type'] for r in subsets['test'] if r['label']==1})}
    with (output/'predictions.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=['split','image','defect_type','label','score','threshold','prediction','nearest_normal_image'])
        writer.writeheader()
        for split in ['val','test']:
            scores, nn_indices = nearest_scores(features[split], features['train'])
            for row, score, index in zip(subsets[split], scores, nn_indices):
                writer.writerow({'split': split, 'image': row['image'], 'defect_type': row['defect_type'],
                                 'label': row['label'], 'score': float(score), 'threshold': threshold,
                                 'prediction': int(score > threshold),
                                 'nearest_normal_image': subsets['train'][int(index)]['image']})
    # Reload the bank and verify serialization does not change the computed scores.
    restored = torch.load(output/'bank.pt', map_location='cpu', weights_only=True)
    reloaded_scores, _ = nearest_scores(features['test'], restored['features'])
    if not np.array_equal(test_scores, reloaded_scores):
        raise RuntimeError('Saved-bank reproducibility check failed')
    metrics['saved_bank_reload_check'] = 'passed'
    distribution_plot(val_scores, test_scores, labels, threshold, output/'score_distribution.png')
    metrics['error_examples_shown'] = error_preview(subsets['test'], test_scores, predicted, output/'error_examples.png')
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
