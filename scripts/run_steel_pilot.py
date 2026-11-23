"""Fixed steel pilot: 300 normal train, 200 calibration, 200 validation evaluation; no test access."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image,ImageDraw
from sklearn.metrics import roc_auc_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_severstal import decode_rle
from src.baseline import load_model,validation_threshold
from src.patchcore import LocalFeatures,coreset_indices
from src.steel import pilot_selection,candidates,infer,heat_rgb


def read_image(row):
    path=ROOT/row['image']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed image')
    with Image.open(path) as image:
        if image.size!=(1600,256):raise ValueError('Expected steel strip')
        return np.asarray(image.convert('RGB')).copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=LocalFeatures(load_model(ROOT/'weights/resnet18-f37072fd.pth',device)).to(device).eval()
    output=ROOT/'runs/steel_pilot';output.mkdir(parents=True,exist_ok=True)
    if args.image:
        saved=torch.load(output/'bank.pt',map_location='cpu',weights_only=True)
        with Image.open(args.image) as image:array=np.asarray(image.convert('RGB')).copy()
        score,heat=infer(model,saved['bank'].to(device),array,device)
        Image.fromarray((.6*array+.4*heat_rgb(heat,saved['visual_scale'])).astype(np.uint8)).save(output/'single_overlay.png')
        np.save(output/'single_map.npy',heat)
        result={'score':score,'threshold':saved['threshold'],'prediction':int(score>saved['threshold']),
                'note':'Pilot model; arbitrary photos are unvalidated. Scores are not probabilities.'}
        (output/'single_prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2));return
    manifest_path=ROOT/'configs/severstal_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    validate_manifest_steel(manifest['samples'])
    selection=pilot_selection(manifest['samples'])
    (ROOT/'configs/steel_pilot_selection.json').write_text(json.dumps(
        {k:[r['image_id'] for r in v] for k,v in selection.items()},indent=2),encoding='utf-8')
    annotation=ROOT/manifest['annotation_csv']
    if hashlib.sha256(annotation.read_bytes()).hexdigest()!=manifest['source_csv_sha256']:raise ValueError('Changed CSV')
    annotations=defaultdict(list)
    with annotation.open(encoding='utf-8-sig',newline='') as file:
        for row in csv.DictReader(file):
            if row['EncodedPixels'].strip():annotations[row['ImageId']].append(row['EncodedPixels'])
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();parts=[];generator=torch.Generator().manual_seed(42)
    for i,row in enumerate(selection['train']):
        parts.append(candidates(model,read_image(row),device,generator))
        if (i+1)%50==0:print(f'Bank extraction {i+1}/300',flush=True)
    pool=torch.cat(parts).to(device);del parts
    bank=pool[coreset_indices(pool,2048)].clone();candidate_count=len(pool);del pool
    build_seconds=time.perf_counter()-start
    print('Bank selected; calibration 200 normal validation images',flush=True)
    calibration=[];visual=[]
    for i,row in enumerate(selection['calibration']):
        score,heat=infer(model,bank,read_image(row),device)
        calibration.append(score);visual.append(float(np.quantile(heat,.995)))
        if (i+1)%50==0:print(f'Calibration {i+1}/200',flush=True)
    threshold=validation_threshold(calibration)
    scale=3*float(np.quantile(visual,.995))
    saved={'bank':bank.cpu(),'threshold':threshold,'visual_scale':scale,
           'tile_size':256,'stride':128,'score':'maximum of stitched smoothed anomaly map',
           'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    torch.save(saved,output/'bank.pt')
    print(f'Threshold fixed {threshold:.4f}; evaluate disjoint validation subset, NOT test',flush=True)
    scores=[];pixel_auc=[];preview=[];seen=set()
    csv_rows=[]
    for i,row in enumerate(selection['evaluation']):
        array=read_image(row);score,heat=infer(model,bank,array,device);scores.append(score)
        mask=np.zeros((256,1600),dtype=np.uint8)
        for text in annotations[row['image_id']]:mask|=decode_rle(text)
        if mask.any() and not mask.all():pixel_auc.append(float(roc_auc_score(mask.ravel(),heat.ravel())))
        key=tuple(row['classes'])
        if key not in seen and len(preview)<6:
            seen.add(key);preview.append((row,array,mask,heat,score))
        csv_rows.append({'image_id':row['image_id'],'label':row['label'],'classes':';'.join(map(str,row['classes'])),
                         'score':score,'prediction':int(score>threshold),'threshold':threshold})
        if (i+1)%50==0:print(f'Validation evaluation {i+1}/200',flush=True)
    labels=np.array([r['label'] for r in selection['evaluation']]);prediction=np.array(scores)>threshold
    tn,fp,fn,tp=confusion_matrix(labels,prediction,labels=[0,1]).ravel()
    canvas=Image.new('RGB',(1600,len(preview)*455),'white');draw=ImageDraw.Draw(canvas)
    for i,(row,array,mask,heat,score) in enumerate(preview):
        marked=array.copy();marked[mask>0]=(.5*marked[mask>0]+.5*np.array([255,0,0])).astype(np.uint8)
        overlay=(.6*array+.4*heat_rgb(heat,scale)).astype(np.uint8)
        for j,panel in enumerate([array,marked,overlay]):
            canvas.paste(Image.fromarray(panel).resize((1600,128)),(0,i*455+35+j*138))
        draw.text((5,i*455+5),f"{row['image_id']} classes={row['classes']} score={score:.3f}: original / official annotation / predicted overlay",fill='black')
    canvas.save(output/'validation_preview.png')
    restored=torch.load(output/'bank.pt',map_location='cpu',weights_only=True)
    first=selection['evaluation'][0]
    repeat,_=infer(model,restored['bank'].to(device),read_image(first),device)
    np.testing.assert_allclose(repeat,scores[0],rtol=1e-5,atol=1e-5)
    report={'stage':'Steel pilot, validation only; test set untouched','seed':42,'device':device,
            'train_normal_images':300,'candidate_features':candidate_count,'candidate_sampling':'32 random patches per tile, seed 42',
            'bank_shape':list(bank.shape),'calibration_normal_images':200,'evaluation_normal_images':100,
            'evaluation_defect_images':100,'threshold':threshold,'calibration_fpr':float(np.mean(np.array(calibration)>threshold)),
            'validation_image_auroc':float(roc_auc_score(labels,scores)),
            'validation_confusion':dict(zip(['tn','fp','fn','tp'],map(int,[tn,fp,fn,tp]))),
            'validation_recall':float(tp/(tp+fn)),'validation_fpr':float(fp/(tn+fp)),
            'mean_per_defect_image_pixel_auroc':float(np.mean(pixel_auc)),
            'pixel_metric_note':'Mean of separate per-defect-image AUROCs at original 256x1600, NOT pooled pixel AUROC; no normal images included',
            'validation_class_counts':{str(c):sum(c in r['classes'] for r in selection['evaluation']) for c in (1,2,3,4)},
            'bank_build_seconds':build_seconds,'pipeline_wall_seconds':time.perf_counter()-start,
            'peak_torch_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
            'bank_bytes':(output/'bank.pt').stat().st_size,'reload_check':'passed',
            'manifest_sha256':saved['manifest_sha256'],'visual_scale':scale,
            'limitations':'Small pilot; validation can guide development, no final test claim; no probability or binary segmentation calibration.'}
    with (output/'validation_predictions.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    text=json.dumps(report,indent=2)
    (output/'metrics.json').write_text(text,encoding='utf-8')
    (ROOT/'reports/steel_pilot_metrics.json').write_text(text,encoding='utf-8')
    print(text,flush=True)


def validate_manifest_steel(rows):
    if len({r['sha256'] for r in rows})!=len(rows):raise ValueError('Duplicate image content')
    if any(r['split'] not in {'train','val','test'} for r in rows):raise ValueError('Invalid split')


if __name__=='__main__':main()
