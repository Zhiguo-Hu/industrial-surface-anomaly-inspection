"""Choose cleanup/score using development only; calibrate normal q95; evaluate fixed pilot."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from sklearn.metrics import roc_auc_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.unet import SmallUNet
from src.resnet_unet import ResNet18UNet
from src.region_scoring import clean_mask,region_scores,pooled_dice
from src.segmentation_diagnostics import mask_statistics
from src.baseline import validation_threshold
from scripts.train_steel_unet import predict
from scripts.run_steel_pilot import read_image
from scripts.audit_severstal import decode_rle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['small','pretrained'],default='pretrained')
    parser.add_argument('--image',type=Path)
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    name='steel_unet' if args.model=='small' else 'steel_unet_balanced_pretrained_resnet18'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint=ROOT/f'runs/{name}/best.pt';checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved=torch.load(checkpoint,map_location=device,weights_only=True)
    model=(SmallUNet() if args.model=='small' else ResNet18UNet()).to(device)
    model.load_state_dict(saved['model']);model.eval()
    output=ROOT/f'runs/{name}_regions';output.mkdir(parents=True,exist_ok=True)
    if args.image:
        rule=json.loads((output/'selected_rule.json').read_text())
        if rule['checkpoint_sha256']!=checkpoint_sha:raise ValueError('Rule/checkpoint mismatch')
        with Image.open(args.image) as image:array=np.asarray(image.convert('RGB')).copy()
        heat=predict(model,array,device)
        scores,mask=region_scores(heat,rule['pixel_threshold'],rule['minimum_area'])
        result={'scores':scores,'selected_score':rule['score_name'],'threshold':rule['image_threshold'],
                'alarm':int(scores[rule['score_name']]>rule['image_threshold']),
                'mask_is_not_gated_by_alarm':True}
        Image.fromarray(mask.astype(np.uint8)*255).save(output/'single_mask.png')
        (output/'single_prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2));return
    path=ROOT/'configs/severstal_manifest.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==saved['manifest_sha256']
    manifest=json.loads(path.read_text());byid={r['image_id']:r for r in manifest['samples']}
    selection=json.loads((ROOT/f'configs/{name}_selection.json').read_text())
    groups={key:[byid[i] for i in selection[key]] for key in ['checkpoint_validation','calibration','evaluation']}
    assert all(r['split']=='val' for rows in groups.values() for r in rows)
    ids=[r['image_id'] for rows in groups.values() for r in rows]
    assert len(ids)==len(set(ids))
    assert all(r['label']==0 for r in groups['calibration'])
    annotation=ROOT/manifest['annotation_csv']
    assert hashlib.sha256(annotation.read_bytes()).hexdigest()==manifest['source_csv_sha256']
    annotations={}
    with annotation.open(encoding='utf-8-sig',newline='') as file:
        for r in csv.DictReader(file):
            if r['EncodedPixels'].strip():annotations.setdefault(r['ImageId'],[]).append(r['EncodedPixels'])
    def target(row):
        mask=np.zeros((256,1600),dtype=bool)
        for text in annotations.get(row['image_id'],[]):mask|=decode_rle(text)>0
        return mask
    dev=[]
    for row in groups['checkpoint_validation']:
        dev.append((row,predict(model,read_image(row),device),target(row)))
    grid=[0,16,64,256];dice_by_area={}
    for area in grid:
        totals=np.zeros(3,dtype=np.int64)
        for row,heat,mask in dev:
            p=clean_mask(heat,saved['pixel_threshold'],area)
            totals+=[int((p&mask).sum()),int((p&~mask).sum()),int((~p&mask).sum())]
        dice_by_area[area]=pooled_dice(totals)
    minimum_area=max(grid,key=lambda a:(dice_by_area[a],-a))
    dev_scores=[region_scores(h,saved['pixel_threshold'],minimum_area)[0] for _,h,_ in dev]
    score_names=list(dev_scores[0])
    labels=[r['label'] for r,_,_ in dev]
    dev_auc={key:float(roc_auc_score(labels,[s[key] for s in dev_scores])) for key in score_names}
    chosen=max(score_names,key=lambda key:dev_auc[key])
    del dev
    print(f'{name}: dev cleanup area={minimum_area}, score={chosen}; now normal calibration',flush=True)
    cal=[]
    for i,row in enumerate(groups['calibration']):
        heat=predict(model,read_image(row),device)
        cal.append(region_scores(heat,saved['pixel_threshold'],minimum_area)[0])
        if (i+1)%100==0:print(f'{name} calibration {i+1}/200',flush=True)
    thresholds={key:validation_threshold([s[key] for s in cal]) for key in score_names}
    rule={'model':name,'checkpoint_sha256':checkpoint_sha,'pixel_threshold':saved['pixel_threshold'],
          'minimum_area':minimum_area,'connectivity':8,'score_name':chosen,'image_threshold':thresholds[chosen],
          'rule_selection':'Cleanup maximizes pooled Dice on 80 checkpoint-validation images; score maximizes image AUROC on the same 80; thresholds are normal-calibration q95',
          'selection_warning':'Reuses checkpoint-selection development data; exploratory, not independent validation'}
    (output/'selected_rule.json').write_text(json.dumps(rule,indent=2),encoding='utf-8')
    (ROOT/f'configs/{name}_region_rule.json').write_text(json.dumps(rule,indent=2),encoding='utf-8')
    # Rules and thresholds are persisted before reading evaluation images.
    records=[];metrics_counts={k:np.zeros(3,dtype=np.int64) for k in ['raw','cleaned']}
    normal_fp={k:0 for k in ['raw','cleaned']};normal_nonempty={k:0 for k in ['raw','cleaned']}
    preview=[]
    for i,row in enumerate(groups['evaluation']):
        array=read_image(row);heat=predict(model,array,device);mask=target(row)
        scores,cleaned=region_scores(heat,saved['pixel_threshold'],minimum_area);raw=heat>saved['pixel_threshold']
        for key,p in [('raw',raw),('cleaned',cleaned)]:
            m=mask_statistics(p,mask);metrics_counts[key]+=[m['tp_pixels'],m['fp_pixels'],m['fn_pixels']]
            if not row['label']:normal_fp[key]+=m['fp_pixels'];normal_nonempty[key]+=int(p.any())
        records.append({'image_id':row['image_id'],'label':row['label'],**scores})
        if len(preview)<4 and (i<2 or i in [100,101]):preview.append((row,array,mask,raw,cleaned,scores))
        if (i+1)%50==0:print(f'{name} evaluation {i+1}/200',flush=True)
    labels=np.array([r['label'] for r in records]);evaluation={}
    for key in score_names:
        scores=np.array([r[key] for r in records]);tn,fp,fn,tp=confusion_matrix(labels,scores>thresholds[key],labels=[0,1]).ravel()
        evaluation[key]={'auroc':float(roc_auc_score(labels,scores)),'threshold':thresholds[key],
                         'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    previous=json.loads((ROOT/f'reports/{name}_metrics.json').read_text())
    np.testing.assert_allclose(evaluation['max_probability']['auroc'],previous['image_auroc'])
    for key in ['tn','fp','fn','tp']:assert evaluation['max_probability'][key]==previous['confusion'][key]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checkpoint_sha
    report={'scope':'Frozen models; development-only selection/calibration and original evaluation subset. No test images.',
            'model':name,'rule':rule,'cleanup_dev_dice_grid':dice_by_area,'score_dev_auroc':dev_auc,
            'evaluation':evaluation,'selected_on_dev':chosen,
            'masks':{k:{'pooled_dice':pooled_dice(v),'normal_nonempty_of_100':normal_nonempty[k],
                        'normal_pixel_fpr':normal_fp[k]/(100*256*1600)} for k,v in metrics_counts.items()},
            'checks':{'original_max_score_reproduced':True,'weights_unchanged':True,
                      'rules_saved_before_evaluation':True,'calibration_dev_eval_disjoint':True},
            'caveats':['No evaluation-based winner selection; selected_on_dev is frozen before evaluation.',
                       'Normal q95 is not a guarantee of 5% false alarms on new samples.',
                       'Connectivity/area filtering can remove real small defects.',
                       'Masks are not gated by alarms. Low alarm rate cannot establish clean masks.']}
    with (output/'evaluation_scores.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    canvas=Image.new('RGB',(1600,450*len(preview)),'white');draw=ImageDraw.Draw(canvas)
    for i,(row,array,mask,raw,cleaned,scores) in enumerate(preview):
        annotation=array.copy();annotation[mask]=(.5*annotation[mask]+.5*np.array([255,0,0])).astype(np.uint8)
        for j,panel in enumerate([annotation,np.repeat(raw[:,:,None],3,2).astype(np.uint8)*255,np.repeat(cleaned[:,:,None],3,2).astype(np.uint8)*255]):
            canvas.paste(Image.fromarray(panel).resize((1600,128)),(0,i*450+30+j*138))
        draw.text((5,i*450+5),f"{row['image_id']} GT={row['label']} selected alarm={int(scores[chosen]>thresholds[chosen])}: annotation / raw mask / cleaned mask",fill='black')
    canvas.save(output/'region_preview.png')
    text=json.dumps(report,indent=2);(ROOT/f'reports/{name}_region_metrics.json').write_text(text,encoding='utf-8');print(text,flush=True)


if __name__=='__main__':main()
