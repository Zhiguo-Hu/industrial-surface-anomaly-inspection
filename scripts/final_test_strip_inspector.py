"""One frozen candidate on the original never-used test split; do not retune on it."""
import argparse,csv,hashlib,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved,normalize_images
from src.strip_inference import MASK_FEATURE_NAMES,decision_features,fused_score,public_policy
from src.inspection_protocol import operating_metrics,frozen_test_fingerprint,verify_test_identity
from src.reproducibility import configure,state_hash
from src.localization_metrics import localization_metrics
from scripts.audit_severstal import decode_rle

OUTPUT=ROOT/'runs/strip_inspector'


class FinalImages(Dataset):
    def __init__(self,rows,annotations):self.rows=rows;self.annotations=annotations
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];path=ROOT/r['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        with Image.open(path) as im:array=np.asarray(im.convert('RGB')).copy()
        mask=np.zeros((256,1600),dtype=np.uint8)
        for text in self.annotations.get(r['image_id'],[]):mask|=decode_rle(text)
        assert int(mask.any())==r['label']
        return torch.from_numpy(array.transpose(2,0,1).copy()),r['label'],i,torch.from_numpy(mask.copy()).unsqueeze(0)


def wilson(success,total):
    z=1.959963984540054;p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den;margin=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [float(center-margin),float(center+margin)]


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-name',default='strip_inspector');args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:parser.error('Invalid run-name')
    OUTPUT=ROOT/'runs'/args.run_name
    configure(42,True);torch.set_num_threads(4)
    qualification=json.loads((OUTPUT/'qualification.json').read_text())
    if qualification['status']!='qualification_passed':raise ValueError('Do not unlock final test before validation qualification passes')
    saved=torch.load(OUTPUT/'candidate.pt',map_location='cpu',weights_only=True)
    assert state_hash(saved['model'])==qualification['model_sha256']==saved['model_sha256']
    policy=saved['inspection_policy'];manifest_path=ROOT/'configs/severstal_manifest.json'
    if policy!=qualification['selected_policy']:raise ValueError('Candidate policy differs from qualification policy')
    if saved.get('pixel_threshold',.5)!=policy.get('pixel_threshold',.5):raise ValueError('Mask threshold differs from frozen inference policy')
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==saved['manifest_sha256']
    manifest=json.loads(manifest_path.read_text());rows=[r for r in manifest['samples'] if r['split']=='test']
    assert len(rows)==1882
    fingerprint=frozen_test_fingerprint(saved['model_sha256'],policy)
    access=ROOT/'reports/strip_final_test_access.json'
    if access.exists():verify_test_identity(json.loads(access.read_text())['frozen_fingerprint'],fingerprint)
    else:access.write_text(json.dumps({'frozen_fingerprint':fingerprint,'model_sha256':saved['model_sha256'],'policy':public_policy(policy),'test_images':1882},indent=2),encoding='utf-8')
    protocol_path=ROOT/'configs/strip_inspection_protocol.json';protocol=json.loads(protocol_path.read_text())
    protocol['test_used']=True;protocol['frozen_test_fingerprint']=fingerprint
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    annotation=ROOT/manifest['annotation_csv'];assert hashlib.sha256(annotation.read_bytes()).hexdigest()==manifest['source_csv_sha256']
    annotations={};wanted={r['image_id'] for r in rows}
    with annotation.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['ImageId'] in wanted and row['EncodedPixels'].strip():annotations.setdefault(row['ImageId'],[]).append(row['EncodedPixels'])
    model=inspector_from_saved(saved).cuda().eval()
    loader=DataLoader(FinalImages(rows,annotations),batch_size=8,num_workers=2,pin_memory=True)
    predictions=[];pixel_counts=[];pixel_threshold=saved.get('pixel_threshold',.5)
    with torch.inference_mode():
        for batch,(x,labels,indices,target) in enumerate(loader):
            x=normalize_images(x.cuda(non_blocking=True));fusion=policy.get('fusion')
            features,mask=decision_features(model,x,policy['flip4'],fusion['feature_names'] if fusion else MASK_FEATURE_NAMES)
            scores=(fused_score(features,fusion) if fusion else features[:,0]).cpu().tolist()
            truth=target.cuda().bool();pred=mask.float().sigmoid()>pixel_threshold
            pixel_counts.extend(torch.stack([(truth&pred).sum((1,2,3)),truth.sum((1,2,3)),pred.sum((1,2,3))],1).cpu().tolist())
            for i,label,score in zip(indices.tolist(),labels.tolist(),scores):
                predictions.append({'image_id':rows[i]['image_id'],'label':label,'score':score,'prediction':int(score>policy['threshold'])})
            if (batch+1)%20==0:print(f'Frozen final test {len(predictions)}/1882',flush=True)
    metrics=operating_metrics([r['label'] for r in predictions],[r['score'] for r in predictions],policy['threshold'])
    passed=metrics['recall']>=.95 and metrics['fpr']<=.05
    per_class={}
    for c in (1,2,3,4):
        group=[p for p,r in zip(predictions,rows) if c in r['classes']];found=sum(p['prediction'] for p in group)
        per_class[str(c)]={'defective_images':len(group),'detected':found,'recall':found/max(1,len(group))}
    report={'status':'dataset_acceptance_passed' if passed else 'dataset_acceptance_failed','target':{'recall_min':.95,'fpr_max':.05},
            'model_sha256':saved['model_sha256'],'frozen_fingerprint':fingerprint,'policy':public_policy(policy),'metrics':metrics,
            'recall_wilson_95':wilson(metrics['tp'],metrics['tp']+metrics['fn']),
            'fpr_wilson_95':wilson(metrics['fp'],metrics['fp']+metrics['tn']),
            'per_class':per_class,'localization':localization_metrics(pixel_counts,pixel_threshold),
            'scope':'One frozen candidate on original test; empirical image-alarm targets. Localization measured at development-selected threshold with no separate production acceptance target; production-camera validation remains separate'}
    (ROOT/'reports/strip_final_test.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (OUTPUT/'final_test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    if passed:
        deployment=OUTPUT/'deployment';deployment.mkdir(exist_ok=True)
        torch.save({**saved,'acceptance':report,'image_size_wh':[1600,256],'precision':'float32'},deployment/'model.pt')
        (ROOT/'configs/strip_deployment_current.json').write_text(json.dumps({'model':(deployment/'model.pt').relative_to(ROOT).as_posix(),
                                                                          'model_sha256':saved['model_sha256'],'frozen_fingerprint':fingerprint},indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
