"""Build lightweight PatchCore-style bank and evaluate image/pixel localization."""
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
from src.baseline import ImageDataset, load_model, validate_manifest, validation_threshold
from src.patchcore import LocalFeatures, anomaly_maps, coreset_indices, patch_distances


@torch.inference_mode()
def predict(model, bank, rows, device):
    loader = DataLoader(ImageDataset(ROOT, rows, 256), batch_size=8, shuffle=False, num_workers=0)
    scores, maps = [], []
    for batch in loader:
        features = model(batch.to(device))
        n,c,h,w = features.shape
        query = features.permute(0,2,3,1).reshape(-1,c)
        distances = patch_distances(query, bank)
        s,m = anomaly_maps(distances,n,h,w)
        scores.extend(s.cpu().tolist())
        maps.append(m.cpu().numpy())
    return np.array(scores), np.concatenate(maps)


def render(rows, maps, scores, threshold, scale, destination):
    canvas = Image.new('RGB', (256*4, 288*len(rows)+28), 'white')
    draw = ImageDraw.Draw(canvas)
    for i,(row,heat,score) in enumerate(zip(rows,maps,scores)):
        with Image.open(ROOT/row['image']) as image:
            image = image.convert('RGB').resize((256,256))
        mask = Image.new('L',(256,256),0)
        if row.get('mask'):
            with Image.open(ROOT/row['mask']) as m:
                mask = m.convert('L').resize((256,256),Image.Resampling.NEAREST)
        v = np.clip(heat/scale,0,1)
        stops=[0.,0.33,0.66,1.]
        anchors=np.array([[15,25,110],[0,180,220],[255,220,0],[220,0,0]])
        color=np.stack([np.interp(v,stops,anchors[:,c]) for c in range(3)],axis=-1).astype(np.uint8)
        overlay = (0.6*np.asarray(image)+0.4*color).astype(np.uint8)
        panels = [image,mask.convert('RGB'),Image.fromarray(color),Image.fromarray(overlay)]
        for j,panel in enumerate(panels):
            canvas.paste(panel,(j*256,i*288+32))
            draw.text((j*256+4,i*288+3), ['Image','Official mask','Predicted heatmap','Overlay'][j], fill='black')
        draw.text((4,i*288+17),f"{row.get('defect_type','custom')} {score:.3f} {'ANOMALY' if score>threshold else 'NORMAL'}",fill='black')
    draw.text((8,288*len(rows)+7),f'Shared scale: blue=0, red>={scale:.3f}; upper=3x normal-validation map q99.5. Not probability.',fill='black')
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,help='Use saved bank for a single image')
    args = parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model=LocalFeatures(load_model(ROOT/'weights/resnet18-f37072fd.pth',device)).to(device).eval()
    output=ROOT/'runs/patchcore_lite'
    output.mkdir(parents=True,exist_ok=True)
    if args.image:
        if not args.image.is_file():
            parser.error(f'Image not found: {args.image}')
        saved=torch.load(output/'bank.pt',map_location='cpu',weights_only=True)
        rows=[{'image':str(args.image.resolve())}]
        scores,maps=predict(model,saved['bank'].to(device),rows,device)
        render(rows,maps,scores,saved['threshold'],saved['visual_scale'],output/'single_heatmap.png')
        np.save(output/'single_anomaly_map.npy',maps[0])
        result={'score':float(scores[0]),'threshold':saved['threshold'],
                'prediction':'anomaly' if scores[0]>saved['threshold'] else 'normal',
                'warning':'Calibrated only on MVTec bottle; heatmap is not a defect probability.'}
        (output/'single_prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2)); return
    manifest_path=ROOT/'configs/bottle_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_manifest(manifest['samples'])
    for row in manifest['samples']:
        if hashlib.sha256((ROOT/row['image']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError(f"Image changed: {row['image']}")
    subsets={s:[r for r in manifest['samples'] if r['split']==s] for s in ['train','val','test']}
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    patches=[]
    with torch.inference_mode():
        for batch in DataLoader(ImageDataset(ROOT,subsets['train'],256),batch_size=8,shuffle=False):
            f=model(batch.to(device))
            patches.append(f.permute(0,2,3,1).reshape(-1,f.shape[1]).cpu())
    pool=torch.cat(patches).to(device)
    print(f'Training patches: {list(pool.shape)}; selecting 1% projected greedy coreset',flush=True)
    indices=coreset_indices(pool,max(1,int(len(pool)*0.01)))
    bank=pool[indices].clone()
    total_patches=len(pool)
    del pool,patches
    build_seconds=time.perf_counter()-start
    print(f'Bank: {list(bank.shape)}, build {build_seconds:.2f}s',flush=True)
    val_scores,val_maps=predict(model,bank,subsets['val'],device)
    threshold=validation_threshold(val_scores)
    scale=3*float(np.quantile(val_maps,0.995))
    saved={'bank':bank.cpu(),'threshold':threshold,'visual_scale':scale,'input_size':256,
           'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
           'method':'PatchCore-style ResNet18 layer2+layer3 / projected greedy 1% / max patch score',
           'train_images':[r['image'] for r in subsets['train']], 'coreset_indices':indices.cpu()}
    torch.save(saved,output/'bank.pt')
    print(f'Val-only threshold fixed: {threshold:.6f}; evaluating test',flush=True)
    test_start=time.perf_counter()
    scores,maps=predict(model,bank,subsets['test'],device)
    test_seconds=time.perf_counter()-test_start
    masks=[]
    for row in subsets['test']:
        if row['mask']:
            with Image.open(ROOT/row['mask']) as mask:
                masks.append(np.asarray(mask.convert('L').resize((256,256),Image.Resampling.NEAREST))>0)
        else: masks.append(np.zeros((256,256),dtype=bool))
    masks=np.stack(masks)
    labels=np.array([r['label'] for r in subsets['test']])
    predicted=scores>threshold
    tn,fp,fn,tp=confusion_matrix(labels,predicted,labels=[0,1]).ravel()
    metrics={'method':saved['method'],'device':device,'training_patch_count':total_patches,
             'bank_shape':list(bank.shape),'threshold':threshold,'visual_scale':scale,
             'validation_false_positive_rate':float(np.mean(val_scores>threshold)),
             'image_auroc':float(roc_auc_score(labels,scores)),
             'pixel_auroc_at_256':float(roc_auc_score(masks.ravel(),maps.ravel())),
             'pixel_metric_scope':'All 83 images, masks nearest-resized to 256; not original-resolution pixel AUROC',
             'confusion':dict(zip(['tn','fp','fn','tp'],[int(tn),int(fp),int(fn),int(tp)])),
             'recall':float(tp/(tp+fn)),'false_positive_rate':float(fp/(tn+fp)),
             'build_seconds':build_seconds,'test_wall_seconds':test_seconds,
             'timing_note':'Includes data loading/processing, not warmed-up latency benchmark',
             'peak_torch_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
             'bank_bytes':(output/'bank.pt').stat().st_size,'manifest_sha256':saved['manifest_sha256']}
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.writer(file);writer.writerow(['image','label','score','threshold','prediction'])
        for row,score,p in zip(subsets['test'],scores,predicted):
            writer.writerow([row['image'],row['label'],float(score),threshold,int(p)])
    np.savez_compressed(output/'test_anomaly_maps.npz',maps=maps,images=np.array([r['image'] for r in subsets['test']]))
    selection=[next(i for i,r in enumerate(subsets['test']) if r['defect_type']==name)
               for name in ['good','broken_large','broken_small','contamination']]
    render([subsets['test'][i] for i in selection],maps[selection],scores[selection],threshold,scale,output/'localization_preview.png')
    reload=torch.load(output/'bank.pt',map_location='cpu',weights_only=True)
    check_scores,check_maps=predict(model,reload['bank'].to(device),subsets['test'][:8],device)
    np.testing.assert_allclose(check_maps,maps[:8],rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(check_scores,scores[:8],rtol=1e-5,atol=1e-5)
    metrics['reload_check']='passed (first test batch)'
    text=json.dumps(metrics,indent=2)
    (output/'metrics.json').write_text(text,encoding='utf-8')
    (ROOT/'reports/patchcore_lite_metrics.json').write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__': main()
