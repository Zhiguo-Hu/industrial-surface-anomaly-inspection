"""Choose output mask threshold on development only; retain classifier weights/scores."""
import argparse,hashlib,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved,normalize_images
from src.strip_cache import CachedStrips
from src.inspection_protocol import partition
from src.localization_metrics import localization_metrics
from src.reproducibility import configure,state_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-name',default='steel_efficientnet_v2_s');args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:parser.error('Invalid run name')
    configure(42,True);torch.set_num_threads(4);output=ROOT/'runs'/args.run_name;cache=ROOT/'runs/strip_training_cache'
    meta=json.loads((cache/'metadata.json').read_text());rows=meta['rows'];selection=partition(rows)
    saved=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
    assert state_hash(saved['model'])==saved['model_sha256']
    assert saved['manifest_sha256']==meta['manifest_sha256']==hashlib.sha256((ROOT/'configs/severstal_manifest.json').read_bytes()).hexdigest()
    model=inspector_from_saved(saved).cuda().eval();indices=selection['development']+selection['qualification'];grid=np.arange(1,10)/10
    records=[];loader=DataLoader(CachedStrips(cache,indices),batch_size=8,num_workers=2,pin_memory=True)
    with torch.inference_mode():
        for batch,(x,target,_) in enumerate(loader):
            mask,_=model.forward_joint(normalize_images(x.cuda()));probability=mask.float().sigmoid();truth=target.cuda().bool();gt=truth.sum((1,2,3));counts=[]
            for t in grid:
                pred=probability>float(t)
                counts.append(torch.stack([(truth&pred).sum((1,2,3)),gt,pred.sum((1,2,3))],1))
            records.append(torch.stack(counts,1).cpu().numpy())
            if (batch+1)%60==0:print(f'Mask calibration {min((batch+1)*8,len(indices))}/{len(indices)}',flush=True)
    records=np.concatenate(records);n=len(selection['development'])
    candidates=[localization_metrics(records[:n,i],t) for i,t in enumerate(grid)]
    selected=max(range(len(grid)),key=lambda i:(candidates[i]['global_dice'],-grid[i]));threshold=float(grid[selected])
    qualification=localization_metrics(records[n:,selected],threshold)
    report={'model_sha256':saved['model_sha256'],'source_epoch':saved['epoch'],'selection':'Max development500 global Dice, lower threshold breaks ties',
            'development_candidates':candidates,'pixel_threshold':threshold,'development':candidates[selected],
            'qualification':qualification,'qualification_at_0p5':localization_metrics(records[n:,4],.5),'test_used':False,
            'scope':'Output mask threshold only; weights and image-alarm feature statistics/scores unchanged. Qualification is reused validation.'}
    torch.save({**saved,'pixel_threshold':threshold,'mask_calibration':report},output/'localization_selected.pt')
    (output/'mask_calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'reports'/f'{args.run_name}_mask_calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
