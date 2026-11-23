"""Resource-bounded supervised pilot; development validation only, test untouched."""
import argparse,csv,hashlib,json,os,random,sys,time
if '--deterministic' in sys.argv:
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
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
from src.resnet_unet import ResNet18UNet,JointResNet18UNet
from src.reproducibility import configure,state_hash
from src.tile_sampling import ClassAwareSampler,POLICIES,SMALL_MAX_PIXELS,update_negative_flip_hash,reference_negative_flip_hash


@torch.inference_mode()
def predict_outputs(model,image,device):
    stitch=Stitcher(*image.shape[:2])
    classification=[]
    for batch,layout in tile_batches(image,device):
        if isinstance(model,JointResNet18UNet):
            logits,cls=model.forward_joint(batch);classification.extend(cls.sigmoid().cpu().tolist())
        else:logits=model(batch)
        maps=logits.sigmoid()[:,0].cpu().numpy()
        for heat,box in zip(maps,layout):stitch.add(heat,box)
    return stitch.finish(),max(classification) if classification else None


def predict(model,image,device):
    return predict_outputs(model,image,device)[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--image',type=Path)
    parser.add_argument('--coverage-balanced',action='store_true',help='Class coverage intervention with the same model and update budget')
    parser.add_argument('--encoder',choices=['small','random-resnet18','pretrained-resnet18'],default='small')
    parser.add_argument('--expanded',action='store_true',help='4000 train images including the coverage-balanced 500')
    parser.add_argument('--rounds',type=int,default=6,help='80 optimization attempts per round')
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--seed',type=int,default=42,help='Training seed; data selection stays fixed at 42')
    parser.add_argument('--classification-weight',type=float,default=None,help='Enable matched joint architecture; 0 is segmentation-only control, 0.5 is joint loss')
    parser.add_argument('--tile-sampling',choices=POLICIES,default='uniform')
    args=parser.parse_args()
    if args.encoder!='small' and not args.coverage_balanced:parser.error('ResNet18 comparisons require --coverage-balanced')
    if args.rounds<1:parser.error('rounds must be positive')
    if args.expanded and not args.coverage_balanced:parser.error('Expansion requires --coverage-balanced')
    if args.classification_weight is not None and (args.classification_weight<0 or not np.isfinite(args.classification_weight)):
        parser.error('classification-weight must be finite and nonnegative')
    if args.classification_weight is not None and args.encoder!='pretrained-resnet18':parser.error('Joint comparisons require pretrained-resnet18')
    if args.tile_sampling!='uniform' and (not args.expanded or not args.deterministic or not args.classification_weight):
        parser.error('Class-aware sampling requires expanded deterministic joint training')
    reproducibility=configure(args.seed,args.deterministic);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    experiment='steel_unet_balanced' if args.coverage_balanced else 'steel_unet'
    if args.encoder!='small':experiment+='_'+args.encoder.replace('-','_')
    if args.expanded:experiment+='_expanded'
    if args.rounds!=6:experiment+=f'_r{args.rounds}'
    if args.deterministic:experiment+=f'_det_s{args.seed}'
    elif args.seed!=42:experiment+=f'_s{args.seed}'
    if args.classification_weight is not None:experiment+=f'_cls{args.classification_weight:g}'.replace('.','p')
    if args.tile_sampling!='uniform':experiment+='_tile_'+args.tile_sampling.replace('-','_')
    output=ROOT/'runs'/experiment;output.mkdir(parents=True,exist_ok=True)
    encoder_model=JointResNet18UNet if args.classification_weight is not None else ResNet18UNet
    model=(SmallUNet() if args.encoder=='small' else encoder_model(
        ROOT/'weights/resnet18-f37072fd.pth' if args.encoder=='pretrained-resnet18' else None)).to(device)
    initial_state_sha256=state_hash(model.state_dict())
    if args.image:
        saved=torch.load(output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(saved['model']);model.eval()
        with Image.open(args.image) as image:array=np.asarray(image.convert('RGB')).copy()
        heat,classification_score=predict_outputs(model,array,device)
        Image.fromarray((.6*array+.4*heat_rgb(heat,1)).astype(np.uint8)).save(output/'single_overlay.png')
        Image.fromarray((heat>saved['pixel_threshold']).astype(np.uint8)*255).save(output/'single_mask.png')
        np.save(output/'single_probability.npy',heat)
        result={'score':float(heat.max()),'threshold':saved['image_threshold'],
                'prediction':int(heat.max()>saved['image_threshold'])}
        if classification_score is not None:
            result.update({'classification_score':classification_score,'classification_threshold':saved['classification_threshold'],
                           'classification_alarm':int(classification_score>saved['classification_threshold']),
                           'classification_head_trained':args.classification_weight>0,'mask_not_gated':True})
        print(json.dumps(result));return
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
    positive=[];negative=[];tile_classes={};foreground_pixels={};class_annotations=defaultdict(dict)
    with (ROOT/manifest['annotation_csv']).open(encoding='utf-8-sig',newline='') as file:
        for row in csv.DictReader(file):
            if row['EncodedPixels'].strip():class_annotations[row['ImageId']][int(row['ClassId'])]=row['EncodedPixels']
    # Index masks directly: do not normalize every image tile just to learn its label.
    for parent,row in enumerate(dataset.rows):
        read_image(row)  # Validate source hash, decodability and dimensions once.
        class_masks={c:decode_rle(text) for c,text in class_annotations[row['image_id']].items()}
        union=np.zeros((256,1600),dtype=bool)
        for m in class_masks.values():union|=m>0
        for tile,(x,y,x2,y2) in enumerate(dataset.layout):
            i=parent*len(dataset.layout)+tile
            tile_classes[i]=[c for c,m in class_masks.items() if m[y:y2,x:x2].any()]
            foreground_pixels[i]=int(union[y:y2,x:x2].sum())
            (positive if tile_classes[i] else negative).append(i)
        if (parent+1)%1000==0:print(f'Indexed {parent+1}/{len(dataset.rows)} parents',flush=True)
    print(f'Tiles: {len(positive)} positive / {len(negative)} negative',flush=True)
    if not positive or not negative:raise ValueError('Both tile classes required')
    tile_index={'tiles_per_parent':len(dataset.layout),'total_tiles':len(dataset),
                'positive':[{'index':i,'classes':tile_classes[i],'pixels':foreground_pixels[i]} for i in positive]}
    (output/'tile_sampling_index.json').write_text(json.dumps(tile_index),encoding='utf-8')
    sampler=ClassAwareSampler(tile_classes,foreground_pixels,args.tile_sampling,args.seed) if args.tile_sampling!='uniform' else None
    sampling_inventory=sampler.inventory() if sampler else {'policy':'uniform'}
    print(json.dumps({'sampling_inventory':sampling_inventory}),flush=True)
    negative_flip_digest=hashlib.sha256();drawn_positive=set();small_draws=0
    if args.deterministic:
        # Isolate training randomness from fixed data-selection shuffles.
        rng=random.Random(args.seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    start=time.perf_counter();history=[];best=-1;best_state=None;class_draws=Counter()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    # Each round is 80 sampled updates, not one exhaustive pass over the dataset.
    skipped_steps=0
    first_round_state_sha256=None
    for round_number in range(1,args.rounds+1):
        model.train();losses=[];seg_losses=[];cls_losses=[]
        for step in range(80):
            uniform_positive=rng.choices(positive,k=4);negative_indices=rng.choices(negative,k=4)
            positive_indices=sampler.draw() if sampler else uniform_positive
            indices=positive_indices+negative_indices
            drawn_positive.update(positive_indices);small_draws+=sum(foreground_pixels[i]<=SMALL_MAX_PIXELS for i in positive_indices)
            class_draws.update(c for i in indices for c in tile_classes[i])
            samples=[dataset[i] for i in indices]
            x=torch.stack([s['image'] for s in samples]);y=torch.stack([s['mask'] for s in samples])
            horizontal=rng.random()<.5;vertical=rng.random()<.5
            update_negative_flip_hash(negative_flip_digest,negative_indices,horizontal,vertical)
            if horizontal:x=x.flip(-1);y=y.flip(-1)
            if vertical:x=x.flip(-2);y=y.flip(-2)
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                if args.classification_weight is not None:
                    logits,cls=model.forward_joint(x)
                    seg_loss=segmentation_loss(logits,y)
                    cls_loss=torch.nn.functional.binary_cross_entropy_with_logits(cls,y.flatten(1).amax(1))
                    loss=seg_loss+args.classification_weight*cls_loss
                    seg_losses.append(float(seg_loss.item()));cls_losses.append(float(cls_loss.item()))
                else:
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
        if args.classification_weight is not None:
            history[-1].update({'segmentation_loss':float(np.mean(seg_losses)),'classification_loss':float(np.mean(cls_losses))})
        if round_number==1:first_round_state_sha256=state_hash(model.state_dict())
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
    cal_scores=[];cal_classification=[]
    for i,row in enumerate(calibration):
        heat,cls_score=predict_outputs(model,read_image(row),device)
        cal_scores.append(float(heat.max()))
        if cls_score is not None:cal_classification.append(cls_score)
        if (i+1)%100==0:print(f'Normal calibration {i+1}/200',flush=True)
    image_threshold=validation_threshold(cal_scores)
    classification_threshold=validation_threshold(cal_classification) if cal_classification else None
    saved={'model':best_state,'image_threshold':image_threshold,'pixel_threshold':pixel_threshold,
           'reproducibility':reproducibility,'model_state_sha256':state_hash(best_state),
           'classification_weight':args.classification_weight,'classification_threshold':classification_threshold,
           'tile_sampling':args.tile_sampling,
           'best_round':best_round,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
           'architecture':('SmallUNet base16 GroupNorm, scratch' if args.encoder=='small' else
                           f'ResNet18UNet encoder={args.encoder}, all layers fine-tuned; BatchNorm encoder/GroupNorm decoder')+'; binary union of four classes'}
    torch.save(saved,output/'best.pt')
    scores=[];pixel_auc=[];aggregate=np.zeros(3,dtype=np.int64);per_defect_dice=[];preview=[];seen=set();pred_rows=[]
    cls_scores=[];gated_counts=np.zeros(3,dtype=np.int64);normal_pixel_fp=0;gated_normal_fp=0;normal_nonempty=0;gated_normal_nonempty=0
    for i,row in enumerate(evaluation):
        array=read_image(row);heat,cls_score=predict_outputs(model,array,device);mask=mask_for(row)>0;p=heat>pixel_threshold
        score=float(heat.max());scores.append(score)
        a,b,c=int((p&mask).sum()),int((p&~mask).sum()),int((~p&mask).sum());aggregate+=[a,b,c]
        if not row['label']:
            normal_pixel_fp+=int(p.sum());normal_nonempty+=int(p.any())
        if cls_score is not None:
            cls_scores.append(cls_score)
            gated=p if cls_score>classification_threshold else np.zeros_like(p)
            gated_counts+=[int((gated&mask).sum()),int((gated&~mask).sum()),int((~gated&mask).sum())]
            if not row['label']:
                gated_normal_fp+=int(gated.sum());gated_normal_nonempty+=int(gated.any())
        if mask.any():
            pixel_auc.append(float(roc_auc_score(mask.ravel(),heat.ravel())))
            per_defect_dice.append(2*a/max(1,2*a+b+c))
        key=tuple(row['classes'])
        if key not in seen and len(preview)<6:seen.add(key);preview.append((row,array,mask,heat,score))
        pred_rows.append({'image_id':row['image_id'],'label':row['label'],'score':score,'prediction':int(score>image_threshold)})
        if cls_score is not None:
            pred_rows[-1].update({'classification_score':cls_score,'classification_alarm':int(cls_score>classification_threshold)})
        if (i+1)%50==0:print(f'Development evaluation {i+1}/200',flush=True)
    labels=np.array([r['label'] for r in evaluation]);tn,fp,fn,tp=confusion_matrix(labels,np.array(scores)>image_threshold,labels=[0,1]).ravel()
    a,b,c=map(int,aggregate)
    report={'scope':'Supervised pilot; same evaluation images as anomaly pilot; NO test images',
            'architecture':saved['architecture'],'parameters':sum(p.numel() for p in model.parameters()),
            'experiment':experiment,'coverage_minimum_per_class':120 if args.expanded else (60 if args.coverage_balanced else None),
            'reproducibility':reproducibility,'initial_state_sha256':initial_state_sha256,
            'first_80_steps_state_sha256':first_round_state_sha256,'best_state_sha256':state_hash(best_state),
            'training_rng_policy':'Dedicated Random(training seed) reset before optimizer' if args.deterministic else 'Historical selection-and-training RNG stream',
            'training_images_per_class':{str(c):sum(c in r['classes'] for r in dataset.rows) for c in (1,2,3,4)},
            'positive_tiles_per_class':{str(c):sum(c in cs for cs in tile_classes.values()) for c in (1,2,3,4)},
            'actual_sampled_tiles_per_class':{str(c):class_draws[c] for c in (1,2,3,4)},
            'original_training_images_retained':len({r['image_id'] for r in dataset.rows}&{r['image_id'] for r in baseline_rows}),
            'train_images':len(dataset.rows),'training_positive_tiles':len(positive),'training_negative_tiles':len(negative),
            'sampling':'Each batch 4 positive + 4 negative tiles, replacement; paired horizontal/vertical flips',
            'tile_sampling':sampling_inventory,'actual_small_positive_tile_draws':small_draws,
            'unique_positive_tiles_drawn':len(drawn_positive),
            'unique_positive_parents_drawn':len({i//len(dataset.layout) for i in drawn_positive}),
            'negative_and_flip_sequence_sha256':negative_flip_digest.hexdigest(),
            'loss':'BCEWithLogits + per-sample soft Dice','optimizer':'AdamW lr=0.001 weight_decay=0.0001',
            'classification_weight':args.classification_weight,'classification_target':'Tile mask nonempty; image score=max of 12 tile-head probabilities' if cls_scores else None,
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
    report['normal_masks']={'nonempty_of_100':normal_nonempty,'normal_pixel_fpr':normal_pixel_fp/(100*256*1600)}
    if args.deterministic:
        expected=reference_negative_flip_hash(positive,negative,args.seed,args.rounds*80)
        assert negative_flip_digest.hexdigest()==expected
        report['negative_and_flip_sequence_matches_uniform_reference']=True
    if cls_scores:
        ct,cfp,cfn,ctp=confusion_matrix(labels,np.array(cls_scores)>classification_threshold,labels=[0,1]).ravel()
        ga,gb,gc=map(int,gated_counts)
        report['classification_metrics']={'head_trained':args.classification_weight>0,'threshold':classification_threshold,
            'image_auroc':float(roc_auc_score(labels,cls_scores)),
            'confusion':dict(zip(['tn','fp','fn','tp'],map(int,[ct,cfp,cfn,ctp]))),
            'recall':float(ctp/(ctp+cfn)),'false_positive_rate':float(cfp/(cfp+ct)),
            'gated_masks_exploratory':{'pooled_dice':2*ga/max(1,2*ga+gb+gc),'normal_nonempty_of_100':gated_normal_nonempty,
                                       'normal_pixel_fpr':gated_normal_fp/(100*256*1600)},
            'gating_warning':'Optional analysis only: suppressing masks can hide defects; raw masks remain default output.'}
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
