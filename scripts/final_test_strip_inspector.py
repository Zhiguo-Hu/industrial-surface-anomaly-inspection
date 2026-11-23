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
from src.strip_inference import inspection_scores
from src.inspection_protocol import operating_metrics,frozen_test_fingerprint,verify_test_identity
from src.reproducibility import configure,state_hash

OUTPUT=ROOT/'runs/strip_inspector'


class FinalImages(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];path=ROOT/r['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        with Image.open(path) as im:array=np.asarray(im.convert('RGB')).copy()
        return torch.from_numpy(array.transpose(2,0,1).copy()),r['label'],i


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
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==saved['manifest_sha256']
    manifest=json.loads(manifest_path.read_text());rows=[r for r in manifest['samples'] if r['split']=='test']
    assert len(rows)==1882
    fingerprint=frozen_test_fingerprint(saved['model_sha256'],policy)
    access=ROOT/'reports/strip_final_test_access.json'
    if access.exists():verify_test_identity(json.loads(access.read_text())['frozen_fingerprint'],fingerprint)
    else:access.write_text(json.dumps({'frozen_fingerprint':fingerprint,'model_sha256':saved['model_sha256'],'policy':policy,'test_images':1882},indent=2),encoding='utf-8')
    protocol_path=ROOT/'configs/strip_inspection_protocol.json';protocol=json.loads(protocol_path.read_text())
    protocol['test_used']=True;protocol['frozen_test_fingerprint']=fingerprint
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    model=inspector_from_saved(saved).cuda().eval()
    loader=DataLoader(FinalImages(rows),batch_size=16,num_workers=2,pin_memory=True)
    predictions=[]
    with torch.inference_mode():
        for batch,(x,labels,indices) in enumerate(loader):
            x=normalize_images(x.cuda(non_blocking=True));scores=inspection_scores(model,x,policy).cpu().tolist()
            for i,label,score in zip(indices.tolist(),labels.tolist(),scores):
                predictions.append({'image_id':rows[i]['image_id'],'label':label,'score':score,'prediction':int(score>policy['threshold'])})
            if (batch+1)%20==0:print(f'Frozen final test {len(predictions)}/1882',flush=True)
    metrics=operating_metrics([r['label'] for r in predictions],[r['score'] for r in predictions],policy['threshold'])
    passed=metrics['recall']>=.95 and metrics['fpr']<=.05
    report={'status':'dataset_acceptance_passed' if passed else 'dataset_acceptance_failed','target':{'recall_min':.95,'fpr_max':.05},
            'model_sha256':saved['model_sha256'],'frozen_fingerprint':fingerprint,'policy':policy,'metrics':metrics,
            'recall_wilson_95':wilson(metrics['tp'],metrics['tp']+metrics['fn']),
            'fpr_wilson_95':wilson(metrics['fp'],metrics['fp']+metrics['tn']),
            'scope':'One frozen candidate on original test; empirical overall image-alarm targets only, not production-camera or pixel-localization acceptance'}
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
