"""Learn compact decision fusion on development only; calibrate on separate normals."""
import argparse,csv,hashlib,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved,normalize_images
from src.strip_inference import MASK_FEATURE_NAMES,feature_names,mask_score_features,fused_score,pyramid_logits,public_policy
from src.strip_cache import CachedStrips
from src.inspection_protocol import partition,development_metrics,operating_metrics
from src.reproducibility import configure,state_hash

OUTPUT=ROOT/'runs/strip_inspector';CACHE=ROOT/'runs/strip_training_cache'


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-name',default='strip_inspector')
    parser.add_argument('--checkpoint',default='best.pt')
    parser.add_argument('--include-pyramid',action='store_true')
    parser.add_argument('--include-nonlinear',action='store_true')
    parser.add_argument('--include-raw-head',action='store_true')
    parser.add_argument('--reuse-features',action='store_true');args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:parser.error('Invalid run-name')
    if Path(args.checkpoint).name!=args.checkpoint:parser.error('Invalid checkpoint name')
    OUTPUT=ROOT/'runs'/args.run_name
    configure(42,True);torch.set_num_threads(4)
    meta=json.loads((CACHE/'metadata.json').read_text());rows=meta['rows'];selection=partition(rows)
    saved=torch.load(OUTPUT/args.checkpoint,map_location='cpu',weights_only=True)
    assert saved['manifest_sha256']==meta['manifest_sha256']==hashlib.sha256((ROOT/'configs/severstal_manifest.json').read_bytes()).hexdigest()
    assert saved['model_sha256']==state_hash(saved['model'])
    indices=selection['development']+selection['calibration']+selection['qualification']
    semantic=any(k.startswith('auxiliary.') for k in saved['model']);previous=None
    if args.reuse_features:
        cached=np.load(OUTPUT/'fusion_features.npz',allow_pickle=False)
        assert str(cached['model_sha256'])==saved['model_sha256']
        assert cached['feature_names'].tolist()==feature_names(semantic,args.include_pyramid)
        ids=cached['image_ids'].tolist();labels=cached['labels']
        assert ids==[rows[i]['image_id'] for i in indices]
        np.testing.assert_array_equal(labels,[rows[i]['label'] for i in indices])
        features={False:cached['single'],True:cached['flip4']}
        if 'pixel_counts' in cached:
            pixel_records=[dict(zip(('tp','gt','predicted','pixels'),map(int,values))) for values in cached['pixel_counts']]
        else:
            # Legacy cache: masks are unchanged by score fusion; reuse audited localization.
            previous=json.loads((OUTPUT/'qualification.json').read_text())
            assert previous['model_sha256']==saved['model_sha256']
            packed=np.load(CACHE/'masks.npy',mmap_mode='r');pixel_records=[]
            for i in indices:pixel_records.append({'gt':int(np.bitwise_count(packed[i]).sum()),'pixels':256*1600})
    else:
        model=inspector_from_saved(saved).cuda().eval()
        loader=DataLoader(CachedStrips(CACHE,indices),batch_size=8,num_workers=2,pin_memory=True)
        features={False:[],True:[]};ids=[];labels=[];pixel_records=[]
        with torch.inference_mode():
            for batch,(x,target,positions) in enumerate(loader):
                x=normalize_images(x.cuda(non_blocking=True));mask,original,aux=model.forward_joint(x,return_aux=True);logits=[original];auxlogits=[aux] if semantic else []
                first=mask_score_features(original,mask)
                if semantic:first=torch.cat([first,aux],1)
                features[False].append(first.cpu().numpy())
                for dimensions in [(-1,),(-2,),(-2,-1)]:
                    pooled=model.features(x.flip(dimensions));logits.append(model.detector(pooled).squeeze(1))
                    if semantic:auxlogits.append(model.auxiliary(pooled))
                averaged=mask_score_features(torch.stack(logits).mean(0),mask)
                if semantic:averaged=torch.cat([averaged,torch.stack(auxlogits).mean(0)],1)
                if args.include_pyramid:
                    cropped=pyramid_logits(model,x)
                    features[False][-1]=torch.cat([first,cropped],1).cpu().numpy()
                    averaged=torch.cat([averaged,cropped],1)
                features[True].append(averaged.cpu().numpy())
                truth=target.cuda(non_blocking=True).bool();predicted=mask>0
                counts=torch.stack([(truth&predicted).sum((1,2,3)),truth.sum((1,2,3)),predicted.sum((1,2,3))],1).cpu().tolist()
                pixel_records.extend({'tp':int(tp),'gt':int(gt),'predicted':int(pred),'pixels':int(mask.shape[-2]*mask.shape[-1])} for tp,gt,pred in counts)
                ids.extend(rows[i]['image_id'] for i in positions.tolist());labels.extend(rows[i]['label'] for i in positions.tolist())
                if (batch+1)%40==0:print(f'Fusion features {len(ids)}/{len(indices)}',flush=True)
        features={k:np.concatenate(v) for k,v in features.items()};labels=np.array(labels)
        # Private, checkpoint-bound features allow replay without repeated GPU work.
        np.savez_compressed(OUTPUT/'fusion_features.npz',single=features[False],flip4=features[True],
                            labels=labels,image_ids=np.array(ids),model_sha256=np.array(saved['model_sha256']),
                            pixel_counts=np.array([[r[k] for k in ('tp','gt','predicted','pixels')] for r in pixel_records]),
                            feature_names=np.array(feature_names(semantic,args.include_pyramid)))
    ndev=len(selection['development']);ncal=len(selection['calibration']);options=[]
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    full_names=feature_names(semantic,args.include_pyramid)
    for flip4,use_semantic,pyramid in [(f,s,p) for f in (False,True) for s in ((False,True) if semantic else (False,))
                                     for p in ((False,True) if args.include_pyramid else (False,))]:
        names=feature_names(use_semantic,pyramid)
        x=features[flip4][:,[full_names.index(name) for name in names]];dev=x[:ndev];y=labels[:ndev]
        models=[('linear',strength,LogisticRegression(C=strength,max_iter=2000)) for strength in (.01,.1,1.)]
        if args.include_nonlinear:models.extend(('rbf_svm',strength,SVC(C=strength,kernel='rbf',gamma='scale')) for strength in (.1,1.,10.))
        for method,strength,estimator in models:
            learner=make_pipeline(StandardScaler(),estimator)
            oof=cross_val_predict(learner,dev,y,cv=splitter,method='decision_function')
            metric=development_metrics(y,oof);learner.fit(dev,y)
            scaler,decision=learner.steps[0][1],learner.steps[1][1]
            spec={'feature_names':names,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                  'method':method,'intercept':float(decision.intercept_[0])}
            if method=='linear':spec['coefficient']=decision.coef_[0].tolist()
            else:spec.update(support_vectors=decision.support_vectors_.tolist(),dual_coefficient=decision.dual_coef_[0].tolist(),gamma=float(decision._gamma))
            actual=fused_score(torch.from_numpy(x),spec).numpy()
            np.testing.assert_allclose(actual,learner.decision_function(x),rtol=1e-5,atol=1e-5)
            options.append({'flip4':flip4,'semantic':use_semantic,'pyramid':pyramid,'method':method,'regularization_C':strength,
                            'oof_development':metric,'selection_metric':metric,'fusion':spec,'scores':actual})
    if args.include_raw_head:
        for flip4 in (False,True):
            spec={'method':'linear','feature_names':MASK_FEATURE_NAMES,'mean':[0.]*9,'scale':[1.]*9,'coefficient':[1.]+[0.]*8,'intercept':0.}
            scores=features[flip4][:,0].astype(float);metric=development_metrics(labels[:ndev],scores[:ndev])
            options.append({'flip4':flip4,'semantic':False,'pyramid':False,'method':'raw_frozen_head','regularization_C':None,
                            'selection_metric':metric,'metric_scope':'Direct development score of frozen CNN; no learned decision fusion, not OOF backbone validation',
                            'fusion':spec,'scores':scores})
    chosen=max(options,key=lambda r:(r['selection_metric']['recall'],r['selection_metric']['auroc']))
    # Quantile choice remains development-only after the OOF model choice.
    thresholds=[]
    for q in (.95,.975):
        threshold=float(np.quantile(chosen['scores'][ndev:ndev+ncal],q,method='higher'))
        metric=operating_metrics(labels[:ndev],chosen['scores'][:ndev],threshold)
        thresholds.append({'q':q,'threshold':threshold,'development':metric})
    eligible=[t for t in thresholds if t['development']['fpr']<=.05] or thresholds
    selected=max(eligible,key=lambda t:(t['development']['recall'],-t['development']['fpr']))
    policy={'flip4':chosen['flip4'],'normal_calibration_quantile':selected['q'],'threshold':selected['threshold'],'fusion':chosen['fusion']}
    qmetrics=operating_metrics(labels[ndev+ncal:],chosen['scores'][ndev+ncal:],policy['threshold'])
    diagnostic=development_metrics(labels[ndev+ncal:],chosen['scores'][ndev+ncal:])
    qstart=ndev+ncal;qpred=chosen['scores'][qstart:]>policy['threshold'];qlabel=labels[qstart:]
    qrows=[rows[i] for i in selection['qualification']];qpixel=pixel_records[qstart:]
    class_metrics={}
    for c in (1,2,3,4):
        group=np.array([c in r['classes'] for r in qrows]);total=int(group.sum());found=int(qpred[group].sum())
        class_metrics[str(c)]={'defective_images':total,'detected':found,'recall':found/max(1,total)}
    if previous is not None:localization=previous['qualification_localization']
    else:
        positives=[r for r in qpixel if r['gt']];normals=[r for r in qpixel if not r['gt']]
        tp=sum(r['tp'] for r in qpixel);gt=sum(r['gt'] for r in qpixel);pred=sum(r['predicted'] for r in qpixel)
        localization={'threshold':.5,'ungated':True,'global_dice':2*tp/max(1,gt+pred),'pixel_recall':tp/max(1,gt),
                      'mean_positive_image_dice':float(np.mean([2*r['tp']/max(1,r['gt']+r['predicted']) for r in positives])),
                      'normal_pixel_fpr':sum(r['predicted'] for r in normals)/max(1,sum(r['pixels'] for r in normals))}
    area_metrics={}
    for lower,upper in [(0,.001),(.001,.01),(.01,.05),(.05,1.00001)]:
        group=np.array([r['gt']>0 and lower<=r['gt']/r['pixels']<upper for r in qpixel]);total=int(group.sum());found=int(qpred[group].sum())
        area_metrics[f'{lower:g}-{upper:g}']={'defective_images':total,'detected':found,'recall':found/max(1,total)}
    passed=qmetrics['recall']>=.95 and qmetrics['fpr']<=.05
    frozen={**saved,'inspection_policy':policy};temporary=OUTPUT/'candidate.tmp';torch.save(frozen,temporary);temporary.replace(OUTPUT/'candidate.pt')
    report={'status':'qualification_passed' if passed else 'target_not_met','target':{'recall_min':.95,'fpr_max':.05},
            'model_sha256':saved['model_sha256'],'source_epoch':saved['epoch'],'source_steps':saved['steps'],
            'source_run':args.run_name,'source_checkpoint':args.checkpoint,'selected_oof_development':chosen.get('oof_development'),
            'selected_development_metric':chosen['selection_metric'],'selected_method':chosen['method'],
            'selection_candidates':[{k:v for k,v in o.items() if k not in ['scores','fusion']} for o in options],
            'selected_policy':policy,'threshold_candidates':thresholds,'qualification':qmetrics,'test_used':False,
            'qualification_per_class':class_metrics,'qualification_by_defect_area':area_metrics,'qualification_localization':localization,
            'diagnostic_only_qualification_roc_at_5pct_fpr':{'not_applied_to_policy':True,**diagnostic},
            'inference_precision':'float32 backbone / float64 decision fusion','scope':'Meta model trained on 500 development images, 5-fold OOF selects method/C/view/crop features; calibration400normal; qualification986 is validation, final test untouched'}
    (OUTPUT/'qualification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    public={**report,'selected_policy':public_policy(policy)}
    (ROOT/'reports/strip_fusion_qualification.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    (ROOT/'reports'/f'{args.run_name}_qualification.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    with (OUTPUT/'qualification_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','label','score','prediction']);writer.writeheader()
        for name,label,score in zip(ids[ndev+ncal:],labels[ndev+ncal:],chosen['scores'][ndev+ncal:]):
            writer.writerow({'image_id':name,'label':int(label),'score':float(score),'prediction':int(score>policy['threshold'])})
    print(json.dumps(public,indent=2),flush=True)


if __name__=='__main__':main()
