"""Resource-bounded supervised pilot; development validation only, test untouched."""
import argparse,csv,hashlib,json,random,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from sklearn.metrics import roc_auc_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.unet import SmallUNet,segmentation_loss
from src.severstal_tiles import SeverstalTiles
from src.steel import tile_batches,heat_rgb
from src.tiling import Stitcher
from scripts.run_steel_pilot import read_image
from scripts.audit_severstal import decode_rle
from src.baseline import validation_threshold
from src.training_selection import coverage_balanced_rows,expanded_training_rows
from src.resnet_unet import ResNet18UNet


@torch.inference_mode()
def predict(model,image,device):
    stitch=Stitcher(*image.shape[:2])
    for batch,layout in tile_batches(image,device):
        maps=model(batch).sigmoid()[:,0].cpu().numpy()
        for heat,box in zip(maps,layout):stitch.add(heat,box)
    return stitch.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--image',type=Path)
    parser.add_argument('--coverage-balanced',action='store_true',help='Class coverage intervention with the same model and update budget')
    parser.add_argument('--encoder',choices=['small','random-resnet18','pretrained-resnet18'],default='small')
    parser.add_argument('--expanded',action='store_true',help='4000 train images including the coverage-balanced 500')
    parser.add_argument('--rounds',type=int,default=6,help='80 optimization attempts per round')
    args=parser.parse_args()
    if args.encoder!='small' and not args.coverage_balanced:parser.error('ResNet18 comparisons require --coverage-balanced')
    if args.rounds<1:parser.error('rounds must be positive')
    if args.expanded and not args.coverage_balanced:parser.error('Expansion requires --coverage-balanced')
    torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    experiment='steel_unet_balanced' if args.coverage_balanced else 'steel_unet'
    if args.encoder!='small':experiment+='_'+args.encoder.replace('-','_')
    if args.expanded:experiment+='_expanded'
    if args.rounds!=6:experiment+=f'_r{args.rounds}'
    output=ROOT/'runs'/experiment;output.mkdir(parents=True,exist_ok=True)
    model=(SmallUNet() if args.encoder=='small' else ResNet18UNet(
        ROOT/'weights/resnet18-f37072fd.pth' if args.encoder=='pretrained-resnet18' else None)).to(device)
    if args.image:
        saved=torch.load(output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(saved['model']);model.eval()
        with Image.open(args.image) as image:array=np.asarray(image.convert('RGB')).copy()
        heat=predict(model,array,device)
        Image.fromarray((.6*array+.4*heat_rgb(heat,1)).astype(np.uint8)).save(output/'single_overlay.png')
        Image.fromarray((heat>saved['pixel_threshold']).astype(np.uint8)*255).save(output/'single_mask.png')
        np.save(output/'single_probability.npy',heat)
        print(json.dumps({'score':float(heat.max()),'threshold':saved['image_threshold'],
                          'prediction':int(heat.max()>saved['image_threshold'])}));return
    dataset=SeverstalTiles(ROOT,'train');rng=random.Random(42)
    train=[]
    for label in (0,1):
        group=[r for r in dataset.rows if r['label']==label];rng.shuffle(group);train+=group[:250]
    dataset.rows=sorted(train,key=lambda r:r['image_id'])
    manifest_path=ROOT/'configs/severstal_manifest.json';manifest=json.loads(manifest_path.read_text())
    rows={r['image_id']:r for r in manifest['samples']}
    pilot=json.loads((ROOT/'configs/steel_pilot_selection.json').read_text())
    used=set(pilot['calibration']+pilot['evaluation'])
    development=[]
    for label in (0,1):
        group=[r for r in manifest['samples'] if r['split']=='val' and r['label']==label and r['image_id'] not in used]
        rng.shuffle(group);development+=group[:40]
    evaluation=[rows[i] for i in pilot['evaluation']];calibration=[rows[i] for i in pilot['calibration']]
    assert len(development)==80 and all(r['split']=='val' for r in development+evaluation+calibration)
    assert not {r['image_id'] for r in development}&used
    baseline_rows=list(dataset.rows)
    if args.coverage_balanced:
        dataset.rows=coverage_balanced_rows(manifest['samples'],baseline_rows)
    coverage_rows=list(dataset.rows)
    if args.expanded:
        dataset.rows=expanded_training_rows(manifest['samples'],coverage_rows)
    selection={'train':[r['image_id'] for r in dataset.rows],'checkpoint_validation':[r['image_id'] for r in development],
               'calibration':pilot['calibration'],'evaluation':pilot['evaluation']}
    if args.coverage_balanced:
        previous=json.loads((ROOT/'configs/steel_unet_selection.json').read_text())
        for key in ('checkpoint_validation','calibration','evaluation'):
            assert selection[key]==previous[key],f'Changed validation set: {key}'
        if not args.expanded:
            assert {r['image_id'] for r in dataset.rows if not r['label']}=={r['image_id'] for r in baseline_rows if not r['label']}
    (ROOT/f'configs/{experiment}_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    annotations=dataset.annotations
    def mask_for(row):
        mask=np.zeros((256,1600),dtype=np.uint8)
        for text in annotations[row['image_id']]:mask|=decode_rle(text)
        return mask
    print(f'Indexing supervised training tiles ({len(dataset.rows)} parent images)',flush=True)
    positive=[];negative=[];tile_classes={};class_annotations=defaultdict(dict)
    with (ROOT/manifest['annotation_csv']).open(encoding='utf-8-sig',newline='') as file:
        for row in csv.DictReader(file):
            if row['EncodedPixels'].strip():class_annotations[row['ImageId']][int(row['ClassId'])]=row['EncodedPixels']
    # Index masks directly: do not normalize every image tile just to learn its label.
    for parent,row in enumerate(dataset.rows):
        read_image(row)  # Validate source hash, decodability and dimensions once.
        class_masks={c:decode_rle(text) for c,text in class_annotations[row['image_id']].items()}
        for tile,(x,y,x2,y2) in enumerate(dataset.layout):
            i=parent*len(dataset.layout)+tile
            tile_classes[i]=[c for c,m in class_masks.items() if m[y:y2,x:x2].any()]
            (positive if tile_classes[i] else negative).append(i)
        if (parent+1)%1000==0:print(f'Indexed {parent+1}/{len(dataset.rows)} parents',flush=True)
    print(f'Tiles: {len(positive)} positive / {len(negative)} negative',flush=True)
    if not positive or not negative:raise ValueError('Both tile classes required')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    start=time.perf_counter();history=[];best=-1;best_state=None;class_draws=Counter()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    # Each round is 80 sampled updates, not one exhaustive pass over the dataset.
    skipped_steps=0
    for round_number in range(1,args.rounds+1):
        model.train();losses=[]
        for step in range(80):
            indices=rng.choices(positive,k=4)+rng.choices(negative,k=4)
            class_draws.update(c for i in indices for c in tile_classes[i])
            samples=[dataset[i] for i in indices]
            x=torch.stack([s['image'] for s in samples]);y=torch.stack([s['mask'] for s in samples])
            if rng.random()<.5:x=x.flip(-1);y=y.flip(-1)
            if rng.random()<.5:x=x.flip(-2);y=y.flip(-2)
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                logits=model(x);loss=segmentation_loss(logits,y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            previous_scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            skipped_steps+=int(scaler.get_scale()<previous_scale)
            losses.append(float(loss.item()))
        model.eval();tp=fp=fn=0
        for row in development:
            heat=predict(model,read_image(row),device);mask=mask_for(row)>0;p=heat>.5
            tp+=int((p&mask).sum());fp+=int((p&~mask).sum());fn+=int((~p&mask).sum())
        dice=2*tp/max(1,2*tp+fp+fn)
        history.append({'round':round_number,'updates':round_number*80,'training_loss':float(np.mean(losses)),
                        'checkpoint_validation_pooled_dice_at_05':dice})
        print(json.dumps(history[-1]),flush=True)
        if dice>best:
            best=dice;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_round=round_number
        (output/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        torch.save({'model':best_state,'best_round':best_round,'current_round':round_number},output/'best_training.pt')
    model.load_state_dict(best_state);model.eval()
    # Pixel threshold selected on checkpoint-development set, never on evaluation images.
    thresholds=np.arange(.1,.91,.1);counts=np.zeros((len(thresholds),3),dtype=np.int64)
    for row in development:
        heat=predict(model,read_image(row),device);mask=mask_for(row)>0
        for i,t in enumerate(thresholds):
            p=heat>t;counts[i]+=[int((p&mask).sum()),int((p&~mask).sum()),int((~p&mask).sum())]
    dice_values=2*counts[:,0]/np.maximum(1,2*counts[:,0]+counts[:,1]+counts[:,2])
    pixel_threshold=float(thresholds[dice_values.argmax()])
    cal_scores=[]
    for i,row in enumerate(calibration):
        cal_scores.append(float(predict(model,read_image(row),device).max()))
        if (i+1)%100==0:print(f'Normal calibration {i+1}/200',flush=True)
    image_threshold=validation_threshold(cal_scores)
    saved={'model':best_state,'image_threshold':image_threshold,'pixel_threshold':pixel_threshold,
           'best_round':best_round,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
           'architecture':('SmallUNet base16 GroupNorm, scratch' if args.encoder=='small' else
                           f'ResNet18UNet encoder={args.encoder}, all layers fine-tuned; BatchNorm encoder/GroupNorm decoder')+'; binary union of four classes'}
    torch.save(saved,output/'best.pt')
    scores=[];pixel_auc=[];aggregate=np.zeros(3,dtype=np.int64);per_defect_dice=[];preview=[];seen=set();pred_rows=[]
    for i,row in enumerate(evaluation):
        array=read_image(row);heat=predict(model,array,device);mask=mask_for(row)>0;p=heat>pixel_threshold
        score=float(heat.max());scores.append(score)
        a,b,c=int((p&mask).sum()),int((p&~mask).sum()),int((~p&mask).sum());aggregate+=[a,b,c]
        if mask.any():
            pixel_auc.append(float(roc_auc_score(mask.ravel(),heat.ravel())))
            per_defect_dice.append(2*a/max(1,2*a+b+c))
        key=tuple(row['classes'])
        if key not in seen and len(preview)<6:seen.add(key);preview.append((row,array,mask,heat,score))
        pred_rows.append({'image_id':row['image_id'],'label':row['label'],'score':score,'prediction':int(score>image_threshold)})
        if (i+1)%50==0:print(f'Development evaluation {i+1}/200',flush=True)
    labels=np.array([r['label'] for r in evaluation]);tn,fp,fn,tp=confusion_matrix(labels,np.array(scores)>image_threshold,labels=[0,1]).ravel()
    a,b,c=map(int,aggregate)
    report={'scope':'Supervised pilot; same evaluation images as anomaly pilot; NO test images',
            'architecture':saved['architecture'],'parameters':sum(p.numel() for p in model.parameters()),
            'experiment':experiment,'coverage_minimum_per_class':120 if args.expanded else (60 if args.coverage_balanced else None),
            'training_images_per_class':{str(c):sum(c in r['classes'] for r in dataset.rows) for c in (1,2,3,4)},
            'positive_tiles_per_class':{str(c):sum(c in cs for cs in tile_classes.values()) for c in (1,2,3,4)},
            'actual_sampled_tiles_per_class':{str(c):class_draws[c] for c in (1,2,3,4)},
            'original_training_images_retained':len({r['image_id'] for r in dataset.rows}&{r['image_id'] for r in baseline_rows}),
            'train_images':len(dataset.rows),'training_positive_tiles':len(positive),'training_negative_tiles':len(negative),
            'sampling':'Each batch 4 positive + 4 negative tiles, replacement; paired horizontal/vertical flips',
            'loss':'BCEWithLogits + per-sample soft Dice','optimizer':'AdamW lr=0.001 weight_decay=0.0001',
            'amp':device=='cuda','rounds':args.rounds,'total_updates':args.rounds*80,'checkpoint_validation_images':80,
            'optimizer_steps_skipped_by_amp':skipped_steps,'completed_optimizer_steps':args.rounds*80-skipped_steps,
            'budget_note':'total_updates is attempted optimization steps; AMP skipped steps are counted separately',
            'best_round':best_round,'history':history,'pixel_threshold':pixel_threshold,'image_threshold':image_threshold,
            'threshold_note':'Pixel threshold maximizes pooled Dice on checkpoint validation; image threshold normal calibration q95',
            'evaluation_normal_images':100,'evaluation_defect_images':100,'image_auroc':float(roc_auc_score(labels,scores)),
            'confusion':dict(zip(['tn','fp','fn','tp'],map(int,[tn,fp,fn,tp]))),
            'recall':float(tp/(tp+fn)),'false_positive_rate':float(fp/(fp+tn)),
            'pooled_pixel_dice_all_200_images':2*a/max(1,2*a+b+c),'pooled_pixel_iou_all_200_images':a/max(1,a+b+c),
            'mean_per_defect_image_dice':float(np.mean(per_defect_dice)),
            'mean_per_defect_image_pixel_auroc':float(np.mean(pixel_auc)),
            'metric_note':'Full 256x1600 reconstruction; pixel AUROC is per-defect-image mean, not pooled.',
            'wall_seconds':time.perf_counter()-start,'peak_torch_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
            'limitations':'Small development pilot, validation already inspected; supervision differs from normal-only baseline. Not production validated.'}
    restored=torch.load(output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(restored['model'])
    repeat=predict(model,read_image(evaluation[0]),device)
    np.testing.assert_allclose(repeat.max(),scores[0],rtol=1e-5,atol=1e-5);report['reload_check']='passed'
    canvas=Image.new('RGB',(1600,len(preview)*590),'white');draw=ImageDraw.Draw(canvas)
    for i,(row,array,mask,heat,score) in enumerate(preview):
        marked=array.copy();marked[mask]=(.5*marked[mask]+.5*np.array([255,0,0])).astype(np.uint8)
        overlay=(.6*array+.4*heat_rgb(heat,1)).astype(np.uint8)
        panels=[array,marked,overlay,np.repeat(((heat>pixel_threshold)*255).astype(np.uint8)[:,:,None],3,axis=2)]
        for j,panel in enumerate(panels):canvas.paste(Image.fromarray(panel).resize((1600,128)),(0,i*590+30+j*138))
        draw.text((5,i*590+5),f"{row['image_id']} classes={row['classes']} score={score:.4f}: original / annotation / UNet overlay / binary mask",fill='black')
    canvas.save(output/'validation_preview.png')
    with (output/'validation_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(pred_rows[0]));w.writeheader();w.writerows(pred_rows)
    text=json.dumps(report,indent=2);(output/'metrics.json').write_text(text,encoding='utf-8')
    (ROOT/f'reports/{experiment}_metrics.json').write_text(text,encoding='utf-8');print(text,flush=True)


if __name__=='__main__':main()
