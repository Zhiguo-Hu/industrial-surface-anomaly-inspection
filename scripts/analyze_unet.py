"""Validation-only class/area and false-positive diagnosis of the saved U-Net."""
import argparse,csv,hashlib,json,os,sys
if '--deterministic' in sys.argv:
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.unet import SmallUNet
from src.resnet_unet import ResNet18UNet,JointResNet18UNet
from src.reproducibility import configure
from src.segmentation_diagnostics import mask_statistics,area_group,summarize
from src.steel import heat_rgb
from scripts.train_steel_unet import predict
from scripts.run_steel_pilot import read_image
from scripts.audit_severstal import decode_rle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage-balanced',action='store_true')
    parser.add_argument('--encoder',choices=['small','random-resnet18','pretrained-resnet18'],default='small')
    parser.add_argument('--expanded',action='store_true')
    parser.add_argument('--rounds',type=int,default=6)
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--classification-weight',type=float,default=None)
    args=parser.parse_args()
    if args.encoder!='small' and not args.coverage_balanced:parser.error('Use --coverage-balanced for ResNet18 comparisons')
    experiment='steel_unet_balanced' if args.coverage_balanced else 'steel_unet'
    if args.encoder!='small':experiment+='_'+args.encoder.replace('-','_')
    if args.expanded:experiment+='_expanded'
    if args.rounds!=6:experiment+=f'_r{args.rounds}'
    if args.deterministic:experiment+=f'_det_s{args.seed}'
    elif args.seed!=42:experiment+=f'_s{args.seed}'
    if args.classification_weight is not None:experiment+=f'_cls{args.classification_weight:g}'.replace('.','p')
    configure(args.seed,args.deterministic);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    saved=torch.load(ROOT/f'runs/{experiment}/best.pt',map_location=device,weights_only=True)
    manifest_path=ROOT/'configs/severstal_manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==saved['manifest_sha256']
    manifest=json.loads(manifest_path.read_text());byid={r['image_id']:r for r in manifest['samples']}
    selection=json.loads((ROOT/f'configs/{experiment}_selection.json').read_text())
    assert all(byid[n]['split']=='train' for n in selection['train'])
    assert all(byid[n]['split']=='val' for n in selection['evaluation'])
    assert not set(selection['evaluation']) & set(selection['checkpoint_validation']+selection['calibration'])
    annotation=ROOT/manifest['annotation_csv']
    assert hashlib.sha256(annotation.read_bytes()).hexdigest()==manifest['source_csv_sha256']
    annotations=defaultdict(dict)
    with annotation.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['EncodedPixels'].strip():annotations[r['ImageId']][int(r['ClassId'])]=r['EncodedPixels']
    factory=JointResNet18UNet if args.classification_weight is not None else ResNet18UNet
    model=(SmallUNet() if args.encoder=='small' else factory()).to(device)
    model.load_state_dict(saved['model']);model.eval()
    records=[];cases={}
    class_hits=defaultdict(lambda:[0,0])
    for i,name in enumerate(selection['evaluation']):
        row=byid[name];array=read_image(row);heat=predict(model,array,device)
        p=heat>saved['pixel_threshold'];mask=np.zeros((256,1600),dtype=np.uint8)
        for c,text in annotations[name].items():
            m=decode_rle(text)>0;mask|=m
            class_hits[c][0]+=int((p&m).sum());class_hits[c][1]+=int(m.sum())
        stats=mask_statistics(p,mask)
        stats.update({'image_id':name,'label':row['label'],'classes':';'.join(map(str,row['classes'])),
                      'score':float(heat.max()),'alarm':int(heat.max()>saved['image_threshold']),
                      'area_fraction':float(mask.mean()),'area_group':area_group(float(mask.mean())),
                      'pred_area_fraction':float(p.mean())})
        records.append(stats)
        category=stats['area_group'] if row['label'] else 'normal'
        priority=-stats['dice'] if row['label'] else stats['fp_pixels']
        if category not in cases or priority>cases[category][0]:cases[category]=(priority,row,array,mask,p,heat,stats)
        if (i+1)%50==0:print(f'UNet diagnosis {i+1}/200',flush=True)
    normals=[r for r in records if not r['label']];defects=[r for r in records if r['label']]
    missed=[r for r in defects if not r['alarm']]
    train=[byid[n] for n in selection['train']]
    per_class={}
    for c in (1,2,3,4):
        subset=[r for r in records if str(c) in r['classes'].split(';')]
        stats=summarize(subset)
        stats.update({'training_images_with_class':sum(c in r['classes'] for r in train),
                      'training_pixels_of_class':sum(int(decode_rle(annotations[r['image_id']][c]).sum()) for r in train if c in r['classes']),
                      'class_specific_pixel_recall':class_hits[c][0]/class_hits[c][1] if class_hits[c][1] else None})
        per_class[str(c)]=stats
    report={'scope':'Fixed saved U-Net, original validation-evaluation subset only; no model/threshold changes, test untouched',
            'image_threshold':saved['image_threshold'],'pixel_threshold':saved['pixel_threshold'],
            'overall':summarize(records),'per_class':per_class,
            'class_metric_note':'Binary union output. Image groups overlap for multi-class images; group Dice uses union masks, not class-specific Dice. class_specific_pixel_recall uses only that class GT pixels.',
            'by_gt_area':{g:summarize([r for r in records if r['area_group']==g]) for g in ['under_0.1pct','0.1_to_1pct','1_to_5pct','at_least_5pct']},
            'normal_diagnostics':{'images':len(normals),'alarms':sum(r['alarm'] for r in normals),
                                  'images_with_any_predicted_pixels':sum(r['pred_pixels']>0 for r in normals),
                                  'pooled_pixel_fpr':sum(r['fp_pixels'] for r in normals)/(len(normals)*256*1600),
                                  'predicted_pixel_area_quantiles_50_90_100':np.quantile([r['pred_area_fraction'] for r in normals],[.5,.9,1]).tolist()},
            'missed_image_diagnostics':{'missed':len(missed),'with_any_predicted_pixels':sum(r['pred_pixels']>0 for r in missed),
                                        'with_union_dice_at_least_05':sum(r['dice']>=.5 for r in missed)},
            'limitations':['Small class subsets; not causal proof that imbalance causes errors.',
                          'Architecture, loss, training coverage, and training duration not ablated here.',
                          'Pixel masks and image alarms have different thresholds; the map is not gated by the alarm.',
                          'No defect class names inferred from numeric class IDs.'],
            'failure_examples':{k:{'image_id':v[1]['image_id'],'dice':v[6]['dice'],'score':v[6]['score']} for k,v in cases.items()}}
    previous=json.loads((ROOT/f'reports/{experiment}_metrics.json').read_text())
    assert sum(r['alarm'] for r in defects)==previous['confusion']['tp']
    assert sum(r['alarm'] for r in normals)==previous['confusion']['fp']
    np.testing.assert_allclose(report['overall']['pooled_union_dice'],previous['pooled_pixel_dice_all_200_images'])
    out=ROOT/f'runs/{experiment}_diagnosis';out.mkdir(parents=True,exist_ok=True)
    with (out/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    canvas=Image.new('RGB',(1600,len(cases)*590),'white');draw=ImageDraw.Draw(canvas)
    for i,(key,(_,row,array,mask,p,heat,stats)) in enumerate(cases.items()):
        marked=array.copy();marked[mask>0]=(.5*marked[mask>0]+.5*np.array([255,0,0])).astype(np.uint8)
        overlay=(.6*array+.4*heat_rgb(heat,1)).astype(np.uint8)
        panels=[array,marked,overlay,np.repeat((p.astype(np.uint8)*255)[:,:,None],3,2)]
        for j,panel in enumerate(panels):canvas.paste(Image.fromarray(panel).resize((1600,128)),(0,i*590+30+j*138))
        draw.text((5,i*590+5),f"{key} {row['image_id']} dice={stats['dice']:.3f}; original / official mask / overlay / predicted mask",fill='black')
    canvas.save(out/'failure_preview.png')
    text=json.dumps(report,indent=2)
    (ROOT/f'reports/{experiment}_diagnosis.json').write_text(text,encoding='utf-8');print(text,flush=True)


if __name__=='__main__':main()
