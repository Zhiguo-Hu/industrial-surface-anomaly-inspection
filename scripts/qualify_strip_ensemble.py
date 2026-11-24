"""Development-only linear fusion of two frozen stages; no new network training."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved
from src.strip_ensemble import StripEnsemble
from src.strip_inference import feature_names,ensemble_feature_names,fused_score,public_policy
from src.inspection_protocol import partition,development_metrics,operating_metrics
from src.reproducibility import state_hash


def main():
    torch.set_num_threads(4);runs=['strip_inspector','strip_inspector_refined']
    sources=[torch.load(ROOT/'runs'/r/'candidate.pt',map_location='cpu',weights_only=True) for r in runs]
    cached=[np.load(ROOT/'runs'/r/'fusion_features.npz',allow_pickle=False) for r in runs]
    meta=json.loads((ROOT/'runs/strip_training_cache/metadata.json').read_text());rows=meta['rows'];selection=partition(rows)
    indices=selection['development']+selection['calibration']+selection['qualification']
    ids=[rows[i]['image_id'] for i in indices];labels=np.array([rows[i]['label'] for i in indices])
    manifest_sha=hashlib.sha256((ROOT/'configs/severstal_manifest.json').read_bytes()).hexdigest()
    for source,cache in zip(sources,cached):
        assert source['model_sha256']==state_hash(source['model'])==str(cache['model_sha256'])
        assert source['manifest_sha256']==meta['manifest_sha256']==manifest_sha
        assert cache['image_ids'].tolist()==ids;np.testing.assert_array_equal(cache['labels'],labels)
    ndev=len(selection['development']);ncal=len(selection['calibration']);start=ndev+ncal;options=[]
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for flip4 in (False,True):
        for semantic in (False,True):
            for pyramid in (False,True):
                names=[feature_names(False,pyramid),feature_names(semantic,pyramid)]
                x=np.concatenate([c['flip4' if flip4 else 'single'][:,[c['feature_names'].tolist().index(n) for n in ns]] for c,ns in zip(cached,names)],1)
                for strength in (.01,.1,1.):
                    learner=make_pipeline(StandardScaler(),LogisticRegression(C=strength,max_iter=2000))
                    oof=cross_val_predict(learner,x[:ndev],labels[:ndev],cv=splitter,method='decision_function')
                    metric=development_metrics(labels[:ndev],oof);learner.fit(x[:ndev],labels[:ndev]);scaler,linear=learner.steps[0][1],learner.steps[1][1]
                    spec={'method':'linear','feature_names':ensemble_feature_names(*names),'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
                          'coefficient':linear.coef_[0].tolist(),'intercept':float(linear.intercept_[0])}
                    scores=fused_score(torch.from_numpy(x),spec).numpy();np.testing.assert_allclose(scores,learner.decision_function(x),rtol=1e-5,atol=1e-5)
                    options.append({'flip4':flip4,'semantic':semantic,'pyramid':pyramid,'regularization_C':strength,
                                    'oof_development':metric,'fusion':spec,'scores':scores})
    chosen=max(options,key=lambda r:(r['oof_development']['recall'],r['oof_development']['auroc']));thresholds=[]
    for q in (.95,.975):
        t=float(np.quantile(chosen['scores'][ndev:start],q,method='higher'))
        thresholds.append({'q':q,'threshold':t,'development':operating_metrics(labels[:ndev],chosen['scores'][:ndev],t)})
    eligible=[r for r in thresholds if r['development']['fpr']<=.05] or thresholds
    selected=max(eligible,key=lambda r:(r['development']['recall'],-r['development']['fpr']))
    policy={'flip4':chosen['flip4'],'normal_calibration_quantile':selected['q'],'threshold':selected['threshold'],'fusion':chosen['fusion']}
    scores=chosen['scores'][start:];qpred=scores>policy['threshold'];metrics=operating_metrics(labels[start:],scores,policy['threshold'])
    qrows=[rows[i] for i in selection['qualification']];per_class={}
    for c in (1,2,3,4):
        group=np.array([c in r['classes'] for r in qrows]);total=int(group.sum());found=int(qpred[group].sum())
        per_class[str(c)]={'defective_images':total,'detected':found,'recall':found/max(1,total)}
    packed=np.load(ROOT/'runs/strip_training_cache/masks.npy',mmap_mode='r')
    area=np.array([int(np.bitwise_count(packed[i]).sum())/(256*1600) for i in selection['qualification']]);by_area={}
    for lower,upper in ((0,.001),(.001,.01),(.01,.05),(.05,1.00001)):
        group=(area>0)&(area>=lower)&(area<upper);total=int(group.sum());found=int(qpred[group].sum())
        by_area[f'{lower:g}-{upper:g}']={'defective_images':total,'detected':found,'recall':found/max(1,total)}
    model=StripEnsemble([inspector_from_saved(s) for s in sources]);state=model.state_dict();sha=state_hash(state)
    output=ROOT/'runs/strip_inspector_ensemble';output.mkdir(exist_ok=True)
    member_info=[{'run':r,'model_sha256':s['model_sha256'],'epoch':s['epoch']} for r,s in zip(runs,sources)]
    saved={'model_kind':'strip_ensemble','model':state,'model_sha256':sha,'manifest_sha256':manifest_sha,
           'inspection_policy':policy,'source_members':member_info,'epoch':0,'steps':0,'phase':'two-frozen-stages'}
    temp=output/'candidate.tmp';torch.save(saved,temp);temp.replace(output/'candidate.pt')
    template=json.loads((ROOT/'runs/strip_inspector_refined/qualification.json').read_text())
    report={'status':'qualification_passed' if metrics['recall']>=.95 and metrics['fpr']<=.05 else 'target_not_met',
            'target':{'recall_min':.95,'fpr_max':.05},'model_sha256':sha,'source_run':output.name,'source_members':member_info,
            'source_epoch':0,'source_steps':0,'selected_oof_development':chosen['oof_development'],'selected_policy':policy,
            'selection_candidates':[{k:v for k,v in r.items() if k not in ('scores','fusion')} for r in options],
            'threshold_candidates':thresholds,'qualification':metrics,'qualification_per_class':per_class,
            'qualification_by_defect_area':by_area,'qualification_localization':template['qualification_localization'],
            'diagnostic_only_qualification_roc_at_5pct_fpr':{'not_applied_to_policy':True,**development_metrics(labels[start:],scores)},
            'test_used':False,'scope':'Two frozen stages, linear meta model trained on development500 only; 5-fold OOF selects features/C/view; calibration400 normal; reused qualification986; refined member mask retained; no test access'}
    (output/'qualification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    public={**report,'selected_policy':public_policy(policy)}
    (ROOT/'reports/strip_ensemble_qualification.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    np.savez_compressed(output/'fusion_features.npz',single=np.concatenate([c['single'] for c in cached],1),
                        flip4=np.concatenate([c['flip4'] for c in cached],1),labels=labels,image_ids=np.array(ids),
                        model_sha256=np.array(sha),feature_names=np.array(ensemble_feature_names(*[c['feature_names'].tolist() for c in cached])))
    with (output/'qualification_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','label','score','prediction']);writer.writeheader()
        for name,label,score in zip(ids[start:],labels[start:],scores):writer.writerow({'image_id':name,'label':int(label),'score':float(score),'prediction':int(score>policy['threshold'])})
    print(json.dumps(public,indent=2))


if __name__=='__main__':main()
