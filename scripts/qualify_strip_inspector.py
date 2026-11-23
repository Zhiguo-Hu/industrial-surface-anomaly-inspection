"""Freeze a candidate; choose TTA/threshold on development+normal calibration only."""
import argparse,csv,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved,normalize_images
from src.strip_cache import CachedStrips
from src.inspection_protocol import partition,operating_metrics
from src.reproducibility import configure,state_hash

OUTPUT=ROOT/'runs/strip_inspector';CACHE=ROOT/'runs/strip_training_cache'


@torch.inference_mode()
def paired_view_scores(model,indices,rows):
    loader=DataLoader(CachedStrips(CACHE,indices),batch_size=16,num_workers=2,pin_memory=True)
    single=[];averaged=[];labels=[];ids=[]
    views=[(),(-1,),(-2,),(-2,-1)]
    for x,_,positions in loader:
        x=normalize_images(x.cuda(non_blocking=True));logits=[]
        for dimensions in views:
            image=x.flip(dimensions) if dimensions else x
            output=model.detector(model.features(image)).squeeze(1)
            logits.append(output.float())
        single.extend(logits[0].cpu().tolist());averaged.extend(torch.stack(logits).mean(0).cpu().tolist())
        ids.extend(rows[i]['image_id'] for i in positions.tolist());labels.extend(rows[i]['label'] for i in positions.tolist())
    return np.array(labels),{False:np.array(single),True:np.array(averaged)},ids


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',default='best.pt')
    parser.add_argument('--run-name',default='strip_inspector');args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:parser.error('Invalid run-name')
    OUTPUT=ROOT/'runs'/args.run_name
    configure(42,True);torch.set_num_threads(4)
    meta=json.loads((CACHE/'metadata.json').read_text());rows=meta['rows'];selection=partition(rows)
    expected_policies=[{'flip4':f,'normal_calibration_quantile':q} for f in [False,True] for q in [.95,.975]]
    protocol=json.loads((ROOT/'configs/strip_inspection_protocol.json').read_text())
    assert protocol['candidate_policies']==expected_policies
    saved=torch.load(OUTPUT/args.checkpoint,map_location='cpu',weights_only=True)
    assert saved['manifest_sha256']==meta['manifest_sha256']
    model=inspector_from_saved(saved).cuda().eval()
    assert state_hash(saved['model'])==saved['model_sha256']
    all_indices=selection['development']+selection['calibration']+selection['qualification']
    all_labels,all_scores,all_ids=paired_view_scores(model,all_indices,rows)
    ndev=len(selection['development']);ncal=len(selection['calibration'])
    candidates=[]
    for flip4 in (False,True):
        labels=all_labels[:ndev];dev=all_scores[flip4][:ndev];cal=all_scores[flip4][ndev:ndev+ncal]
        for quantile in (.95,.975):
            threshold=float(np.quantile(cal,quantile,method='higher'))
            metric=operating_metrics(labels,dev,threshold)
            candidates.append({'flip4':flip4,'normal_calibration_quantile':quantile,'threshold':threshold,'development':metric})
    eligible=[c for c in candidates if c['development']['fpr']<=.05]
    if not eligible:eligible=candidates
    chosen=max(eligible,key=lambda c:(c['development']['recall'],-c['development']['fpr'],c['development']['auroc']))
    labels=all_labels[ndev+ncal:];scores=all_scores[chosen['flip4']][ndev+ncal:];ids=all_ids[ndev+ncal:]
    metrics=operating_metrics(labels,scores,chosen['threshold'])
    passed=metrics['recall']>=.95 and metrics['fpr']<=.05
    frozen={**saved,'inspection_policy':chosen,'model_sha256':state_hash(saved['model'])}
    temporary=OUTPUT/'candidate.tmp';torch.save(frozen,temporary);temporary.replace(OUTPUT/'candidate.pt')
    report={'status':'qualification_passed' if passed else 'target_not_met','target':{'recall_min':.95,'fpr_max':.05},
            'model_sha256':frozen['model_sha256'],'source_epoch':saved.get('epoch'),'source_steps':saved.get('steps'),
            'selection_candidates':candidates,'selected_policy':chosen,'qualification':metrics,
            'inference_precision':'float32','test_used':False,'scope':'Reused validation partitions, not independent final test; no operating-policy selection using qualification labels'}
    (ROOT/'reports/strip_qualification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUTPUT/'qualification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUTPUT/'qualification_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','label','score','prediction']);writer.writeheader()
        for name,label,score in zip(ids,labels,scores):writer.writerow({'image_id':name,'label':int(label),'score':float(score),'prediction':int(score>chosen['threshold'])})
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
