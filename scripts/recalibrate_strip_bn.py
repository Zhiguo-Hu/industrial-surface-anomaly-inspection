"""Compare unchanged weights with train-only BatchNorm recalibration on development."""
import argparse,copy,hashlib,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved,normalize_images
from src.strip_inference import image_logits
from src.strip_cache import CachedStrips
from src.inspection_protocol import partition,development_metrics
from src.reproducibility import configure,state_hash


@torch.inference_mode()
def development(model,loader,rows):
    model.eval();labels=[];scores=[]
    for x,_,indices in loader:
        scores.extend(image_logits(model,normalize_images(x.cuda())).cpu().tolist())
        labels.extend(rows[i]['label'] for i in indices.tolist())
    return development_metrics(labels,scores)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-name',default='strip_inspector_refined');args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:parser.error('Invalid run name')
    configure(42,True);torch.set_num_threads(4)
    output=ROOT/'runs'/args.run_name;cache=ROOT/'runs/strip_training_cache'
    meta=json.loads((cache/'metadata.json').read_text());rows=meta['rows'];selection=partition(rows)
    saved=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
    assert saved['model_sha256']==state_hash(saved['model'])
    assert saved['manifest_sha256']==meta['manifest_sha256']==hashlib.sha256((ROOT/'configs/severstal_manifest.json').read_bytes()).hexdigest()
    model=inspector_from_saved(saved).cuda().eval()
    parameter_hash=state_hash(dict(model.named_parameters()))
    devloader=DataLoader(CachedStrips(cache,selection['development']),batch_size=16,num_workers=2,pin_memory=True,persistent_workers=True)
    original=development(model,devloader,rows);print('Original '+json.dumps(original),flush=True)
    layers=[module for module in model.encoder.modules() if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)]
    for layer in layers:layer.reset_running_stats();layer.momentum=None;layer.train()
    loader=DataLoader(CachedStrips(cache,selection['train']),batch_size=16,shuffle=True,
                      generator=torch.Generator().manual_seed(42),num_workers=2,pin_memory=True)
    count=0
    with torch.inference_mode():
        for x,_,_ in loader:
            model.features(normalize_images(x.cuda()));count+=len(x)
            if count%1600==0:print(f'Train-only BN {count}/{len(selection["train"])}',flush=True)
    assert count==len(selection['train'])==8800
    assert parameter_hash==state_hash(dict(model.named_parameters())),'Recalibration changed learned parameters'
    recalibrated=development(model,devloader,rows)
    use_recalibrated=(recalibrated['recall'],recalibrated['auroc'])>(original['recall'],original['auroc'])
    selected=copy.deepcopy(saved)
    if use_recalibrated:
        selected['model']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        selected['model_sha256']=state_hash(selected['model']);selected['development']=recalibrated
        selected['bn_recalibration']={'source_model_sha256':saved['model_sha256'],'train_images':count,'learned_parameters_unchanged':True}
    temporary=output/'inference_selected.tmp';torch.save(selected,temporary);temporary.replace(output/'inference_selected.pt')
    report={'source_model_sha256':saved['model_sha256'],'selected_model_sha256':selected['model_sha256'],
            'source_epoch':saved['epoch'],'source_run':args.run_name,'original_development':original,
            'recalibrated_development':recalibrated,'selected':'recalibrated' if use_recalibrated else 'original',
            'train_images':count,'learned_parameters_unchanged':True,'test_used':False,
            'scope':'BN running moments only, all train images at native resolution without augmentation; selected on development500, no qualification or test labels'}
    (output/'bn_recalibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'reports'/f'{args.run_name}_bn_recalibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
