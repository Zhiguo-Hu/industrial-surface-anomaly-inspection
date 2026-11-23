"""Development-only failure diagnosis using the frozen steel pilot; never reads test images."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_severstal import decode_rle
from scripts.run_steel_pilot import read_image
from src.baseline import load_model, validation_threshold
from src.patchcore import LocalFeatures
from src.steel import infer, heat_rgb


def summarize_map(heat, mask):
    y,x=np.unravel_index(heat.argmax(),heat.shape)
    median=float(np.median(heat)); peak=float(heat[y,x])
    seam_distance=min(abs(x-edge) for edge in [128,256,384,512,640,768,896,1024,1152,1280,1344,1408,1536])
    result={'peak':peak,'median':median,'q999':float(np.quantile(heat,.999)),
            'interior_peak':float(heat[16:-16,16:-16].max()),
            'contrast':peak-median,'ratio':peak/max(median,1e-8),
            'peak_x':int(x),'peak_y':int(y),
            'peak_outer_border16':bool(y<16 or y>=heat.shape[0]-16 or x<16 or x>=heat.shape[1]-16),
            'peak_internal_seam8':bool(seam_distance<=8),
            'peak_in_annotation':bool(mask[y,x])}
    if mask.any():
        result.update({'defect_max':float(heat[mask>0].max()),
                       'background_max':float(heat[mask==0].max()),
                       'defect_median':float(np.median(heat[mask>0])),
                       'background_median':float(np.median(heat[mask==0])),
                       'pixel_auroc':float(roc_auc_score(mask.ravel(),heat.ravel()))})
    return result


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    saved=torch.load(ROOT/'runs/steel_pilot/bank.pt',map_location='cpu',weights_only=True)
    manifest_path=ROOT/'configs/severstal_manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==saved['manifest_sha256']
    manifest=json.loads(manifest_path.read_text())
    by_id={r['image_id']:r for r in manifest['samples']}
    selection=json.loads((ROOT/'configs/steel_pilot_selection.json').read_text())
    calibration=[by_id[i] for i in selection['calibration']]
    evaluation=[by_id[i] for i in selection['evaluation']]
    assert all(r['split']=='val' for r in calibration+evaluation)
    assert not set(selection['calibration']) & set(selection['evaluation'])
    model=LocalFeatures(load_model(ROOT/'weights/resnet18-f37072fd.pth',device)).to(device).eval()
    bank=saved['bank'].to(device)
    annotations={}
    csv_path=ROOT/manifest['annotation_csv']
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest()==manifest['source_csv_sha256']
    with csv_path.open(encoding='utf-8-sig',newline='') as file:
        for r in csv.DictReader(file):
            if r['EncodedPixels'].strip():annotations.setdefault(r['ImageId'],[]).append(r['EncodedPixels'])
    details=[];cases={}
    for group,rows in [('calibration',calibration),('evaluation',evaluation)]:
        for i,row in enumerate(rows):
            array=read_image(row);_,heat=infer(model,bank,array,device)
            mask=np.zeros(heat.shape,dtype=np.uint8)
            for text in annotations.get(row['image_id'],[]):mask|=decode_rle(text)
            values=summarize_map(heat,mask)
            gray=np.asarray(Image.fromarray(array).convert('L'))
            values.update({'image_id':row['image_id'],'group':group,'label':row['label'],
                           'classes':';'.join(map(str,row['classes'])),'gray_mean':float(gray.mean()),
                           'gray_std':float(gray.std()),'dark_fraction':float(np.mean(gray<15)),
                           'bright_fraction':float(np.mean(gray>245))})
            details.append(values)
            if group=='evaluation':
                key=('FP' if values['peak']>saved['threshold'] else 'TN') if not row['label'] else ('TP' if values['peak']>saved['threshold'] else 'FN')
                priority=values['peak'] if key in ['FP','TP'] else -values['peak']
                if key not in cases or priority>cases[key][0]:cases[key]=(priority,row,array,mask,heat,values)
            if (i+1)%50==0:print(f'{group}: {i+1}/{len(rows)}',flush=True)
    cal=[d for d in details if d['group']=='calibration'];eva=[d for d in details if d['group']=='evaluation']
    labels=np.array([d['label'] for d in eva])
    alternatives={}
    for score in ['peak','interior_peak','q999','contrast','ratio']:
        threshold=validation_threshold([d[score] for d in cal])
        scores=np.array([d[score] for d in eva]);p=scores>threshold
        tn,fp,fn,tp=confusion_matrix(labels,p,labels=[0,1]).ravel()
        alternatives[score]={'calibration_threshold':threshold,'auroc':float(roc_auc_score(labels,scores)),
                             'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    groups={}
    for label,name in [(0,'normal'),(1,'defective')]:
        sub=[d for d in eva if d['label']==label]
        groups[name]={key:float(np.mean([d[key] for d in sub])) for key in ['peak_outer_border16','peak_internal_seam8','peak_in_annotation']}
        groups[name]['peak_quantiles_05_50_95']=np.quantile([d['peak'] for d in sub],[.05,.5,.95]).tolist()
        groups[name]['median_map_quantiles_05_50_95']=np.quantile([d['median'] for d in sub],[.05,.5,.95]).tolist()
        groups[name]['spearman_peak_vs_map_median']=float(spearmanr([d['peak'] for d in sub],[d['median'] for d in sub]).statistic)
        groups[name]['spearman_peak_vs_gray_std']=float(spearmanr([d['peak'] for d in sub],[d['gray_std'] for d in sub]).statistic)
    defects=[d for d in eva if d['label']]
    report={'scope':'Same frozen bank and disjoint validation subsets; exploratory diagnosis, no test images',
            'image_count':len(details),'original_threshold':saved['threshold'],'groups':groups,
            'defect_peak_in_annotation_count':sum(d['peak_in_annotation'] for d in defects),
            'defect_median_exceeds_background_median_count':sum(d['defect_median']>d['background_median'] for d in defects),
            'mean_within_defect_image_pixel_auroc':float(np.mean([d['pixel_auroc'] for d in defects])),
            'alternative_scores_exploratory_not_selected':alternatives,
            'outer_border16_area_fraction':1-(224*1568)/(256*1600),
            'seam_band_note':'Within 8 px of listed internal tile start/end boundaries; area is large, so presence alone is not evidence of an artifact.',
            'caveats':['Alternatives evaluated on already inspected validation; not independent test results.',
                       'Border exclusion also discards possible real border defects; not a deployment fix.',
                       'No ablation of backbone, candidate sampling, or bank capacity; their causal effect is unproven.',
                       'Exact diagnosis does not imply input annotation semantics are wrong.'],
            'cases':{k:{'image_id':v[1]['image_id'],'label':v[1]['label'],'peak':v[5]['peak'],
                        'peak_xy':[v[5]['peak_x'],v[5]['peak_y']]} for k,v in cases.items()}}
    assert abs(alternatives['peak']['auroc']-.5205)<1e-6
    out=ROOT/'runs/steel_diagnosis';out.mkdir(parents=True,exist_ok=True)
    fields=sorted({key for d in details for key in d})
    with (out/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(details)
    canvas=Image.new('RGB',(1600,450*len(cases)),'white');draw=ImageDraw.Draw(canvas)
    for i,key in enumerate(['FP','FN','TP','TN']):
        if key not in cases:continue
        _,row,array,mask,heat,values=cases[key]
        marked=array.copy();marked[mask>0]=(.5*marked[mask>0]+.5*np.array([255,0,0])).astype(np.uint8)
        overlay=(.6*array+.4*heat_rgb(heat,saved['visual_scale'])).astype(np.uint8)
        for j,panel in enumerate([array,marked,overlay]):
            image=Image.fromarray(panel).resize((1600,128))
            painter=ImageDraw.Draw(image);x,y=values['peak_x'],values['peak_y']//2
            painter.ellipse((x-9,y-9,x+9,y+9),outline='white',width=2)
            canvas.paste(image,(0,i*450+30+j*138))
        draw.text((5,i*450+5),f"{key}: {row['image_id']} score={values['peak']:.3f}; white circle=max score; original / annotation / prediction",fill='black')
    canvas.save(out/'failure_cases.png')
    text=json.dumps(report,indent=2)
    (ROOT/'reports/steel_failure_diagnosis.json').write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':main()
