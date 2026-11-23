"""Learn compact decision fusion on development only; calibrate on separate normals."""
import argparse,csv,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved,normalize_images
from src.strip_inference import feature_names,mask_score_features,fused_score,pyramid_logits
from src.strip_cache import CachedStrips
from src.inspection_protocol import partition,development_metrics,operating_metrics
from src.reproducibility import configure,state_hash

OUTPUT=ROOT/'runs/strip_inspector';CACHE=ROOT/'runs/strip_training_cache'


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-name',default='strip_inspector')
    parser.add_argument('--include-pyramid',action='store_true');args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:parser.error('Invalid run-name')
    OUTPUT=ROOT/'runs'/args.run_name
    configure(42,True);torch.set_num_threads(4)
    meta=json.loads((CACHE/'metadata.json').read_text());rows=meta['rows'];selection=partition(rows)
    saved=torch.load(OUTPUT/'best.pt',map_location='cpu',weights_only=True)
    assert saved['model_sha256']==state_hash(saved['model'])
    model=inspector_from_saved(saved).cuda().eval()
    indices=selection['development']+selection['calibration']+selection['qualification']
    loader=DataLoader(CachedStrips(CACHE,indices),batch_size=8,num_workers=2,pin_memory=True)
    features={False:[],True:[]};ids=[];labels=[];semantic=hasattr(model,'auxiliary')
    with torch.inference_mode():
        for batch,(x,_,positions) in enumerate(loader):
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
            ids.extend(rows[i]['image_id'] for i in positions.tolist());labels.extend(rows[i]['label'] for i in positions.tolist())
            if (batch+1)%40==0:print(f'Fusion features {len(ids)}/{len(indices)}',flush=True)
    features={k:np.concatenate(v) for k,v in features.items()};labels=np.array(labels)
    ndev=len(selection['development']);ncal=len(selection['calibration']);options=[]
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for flip4,pyramid in [(f,p) for f in (False,True) for p in ((False,True) if args.include_pyramid else (False,))]:
        names=feature_names(semantic,pyramid)
        x=features[flip4][:,:len(names)];dev=x[:ndev];y=labels[:ndev]
        for strength in (.01,.1,1.):
            learner=make_pipeline(StandardScaler(),LogisticRegression(C=strength,max_iter=2000))
            oof=cross_val_predict(learner,dev,y,cv=splitter,method='decision_function')
            metric=development_metrics(y,oof);learner.fit(dev,y)
            scaler,linear=learner.steps[0][1],learner.steps[1][1]
            spec={'feature_names':names,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                  'coefficient':linear.coef_[0].tolist(),'intercept':float(linear.intercept_[0])}
            actual=fused_score(torch.from_numpy(x),spec).numpy()
            np.testing.assert_allclose(actual,learner.decision_function(x),rtol=1e-5,atol=1e-5)
            options.append({'flip4':flip4,'pyramid':pyramid,'regularization_C':strength,'oof_development':metric,'fusion':spec,'scores':actual})
    chosen=max(options,key=lambda r:(r['oof_development']['recall'],r['oof_development']['auroc']))
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
    passed=qmetrics['recall']>=.95 and qmetrics['fpr']<=.05
    frozen={**saved,'inspection_policy':policy};temporary=OUTPUT/'candidate.tmp';torch.save(frozen,temporary);temporary.replace(OUTPUT/'candidate.pt')
    report={'status':'qualification_passed' if passed else 'target_not_met','target':{'recall_min':.95,'fpr_max':.05},
            'model_sha256':saved['model_sha256'],'source_epoch':saved['epoch'],'source_steps':saved['steps'],
            'selection_candidates':[{k:v for k,v in o.items() if k not in ['scores','fusion']} for o in options],
            'selected_policy':policy,'threshold_candidates':thresholds,'qualification':qmetrics,'test_used':False,
            'inference_precision':'float32','scope':'Meta model trained on 500 development images, 5-fold OOF selects C/view/crop features; calibration400normal; qualification986 is validation, final test untouched'}
    (OUTPUT/'qualification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'reports/strip_fusion_qualification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUTPUT/'qualification_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','label','score','prediction']);writer.writeheader()
        for name,label,score in zip(ids[ndev+ncal:],labels[ndev+ncal:],chosen['scores'][ndev+ncal:]):
            writer.writerow({'image_id':name,'label':int(label),'score':float(score),'prediction':int(score>policy['threshold'])})
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
