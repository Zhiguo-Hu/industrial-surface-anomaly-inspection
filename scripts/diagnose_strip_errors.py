"""Private visual review and numerical error overlap of existing frozen candidates."""
import csv,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved
from src.strip_inference import inspect_image
from src.reproducibility import configure


def main():
    configure(42,True);torch.set_num_threads(2)
    cache=ROOT/'runs/strip_training_cache';meta=json.loads((cache/'metadata.json').read_text());lookup={r['image_id']:(i,r) for i,r in enumerate(meta['rows'])}
    records={};misses=[]
    for name in ('strip_inspector','strip_inspector_refined','strip_inspector_ensemble'):
        with (ROOT/'runs'/name/'qualification_predictions.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        records[name]=rows;misses.append({r['image_id'] for r in rows if r['label']=='1' and r['prediction']=='0'})
    rows=records['strip_inspector_ensemble'];fp=sorted([r for r in rows if r['label']=='0' and r['prediction']=='1'],key=lambda r:-float(r['score']))
    fn=sorted([r for r in rows if r['label']=='1' and r['prediction']=='0'],key=lambda r:float(r['score']))
    selected_fn=[]
    for c in (2,4):selected_fn.extend([r for r in fn if c in lookup[r['image_id']][1]['classes'] and r not in selected_fn][:2])
    selected=[('FP',r) for r in fp[:4]]+[('FN',r) for r in selected_fn]
    common=set.intersection(*misses)
    report={'baseline':'frozen two-stage linear ensemble','false_negatives_by_stage':[len(m) for m in misses],
            'common_misses_all_three':len(common),'common_miss_classes':{str(c):sum(c in lookup[name][1]['classes'] for name in common) for c in (1,2,3,4)},
            'review':[{'error':kind,'image_id':r['image_id'],'score':float(r['score']),'classes':lookup[r['image_id']][1]['classes']} for kind,r in selected],
            'scope':'Qualification validation errors for diagnosis only; none moved into train; no test access'}
    (ROOT/'reports/efficientnet_error_baseline.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    images=np.load(cache/'images.npy',mmap_mode='r');packed=np.load(cache/'masks.npy',mmap_mode='r')
    saved=torch.load(ROOT/'runs/strip_inspector_ensemble/candidate.pt',map_location='cpu',weights_only=True);model=inspector_from_saved(saved).cuda().eval();policy=saved['inspection_policy']
    sheet=Image.new('RGB',(1200,4*410),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    for number,(kind,r) in enumerate(selected):
        column=0 if kind=='FP' else 1;row=number if kind=='FP' else number-4;left=column*600;top=row*410
        index,info=lookup[r['image_id']];assert info['split']=='val';array=np.array(images[index],copy=True);truth=np.unpackbits(packed[index],axis=-1,count=1600).astype(bool)
        _,prob=inspect_image(model,array,'cuda',policy['flip4'],policy['fusion'])
        anchor=int(truth.sum(0).argmax()) if kind=='FN' else int(np.unravel_index(prob.argmax(),prob.shape)[1])
        start=max(0,min(1600-256,anchor-128));patch=array[:,start:start+256].copy();mask=truth[:,start:start+256]
        interior=mask.copy();interior[1:]&=mask[:-1];interior[:-1]&=mask[1:];interior[:,1:]&=mask[:,:-1];interior[:,:-1]&=mask[:,1:]
        patch[mask&~interior]=[255,40,40]
        draw.text((left+8,top+4),f"{kind} {r['image_id']}  score={float(r['score']):.2f} classes={info['classes']}",font=font,fill='black')
        sheet.paste(Image.fromarray(array).resize((600,96)),(left,top+30));sheet.paste(Image.fromarray(patch),(left+172,top+140))
        draw.text((left+8,top+140),'Native 256px\nGT edge: red',font=font,fill='black')
    output=ROOT/'runs/efficientnet_error_review';output.mkdir(exist_ok=True);sheet.save(output/'baseline_errors.png')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
