"""Fast all-parent frozen-feature head pretraining before full-strip fine tuning."""
import json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_cache import CachedStrips
from src.strip_inspector import StripInspector,normalize_images
from src.inspection_protocol import partition,development_metrics
from src.reproducibility import configure,state_hash

OUTPUT=ROOT/'runs/strip_inspector';CACHE=ROOT/'runs/strip_training_cache'
SOURCE=ROOT/'runs/steel_unet_balanced_pretrained_resnet18_expanded_r30_det_s42_cls0p5/best.pt'


def main():
    runtime=configure(42,True);torch.set_num_threads(4);OUTPUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((CACHE/'metadata.json').read_text());rows=meta['rows'];selection=partition(rows)
    protocol={'seed':42,'recall_target':.95,'fpr_target':.05,
              'selection':{k:[rows[i]['image_id'] for i in v] for k,v in selection.items()},
              'calibration_quantile':.975,'test_used':False,
              'candidate_policies':[{'flip4':f,'normal_calibration_quantile':q} for f in [False,True] for q in [.95,.975]],
              'scope':'New full-strip protocol; reused validation set, test locked until qualification passes'}
    (ROOT/'configs/strip_inspection_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    source=torch.load(SOURCE,map_location='cpu',weights_only=True);model=StripInspector().cuda();model.warm_start(source)
    feature_path=OUTPUT/'frozen_features.pt'
    if feature_path.exists():
        cached=torch.load(feature_path,map_location='cpu',weights_only=True)
        assert cached['source_sha256']==source['model_state_sha256'] and cached['manifest_sha256']==meta['manifest_sha256']
        features=cached['features']
    else:
        loader=DataLoader(CachedStrips(CACHE,range(len(rows))),batch_size=16,num_workers=2,pin_memory=True)
        features=torch.empty(len(rows),1792);model.eval()
        with torch.inference_mode():
            for batch,(x,_,indices) in enumerate(loader):
                x=normalize_images(x.cuda(non_blocking=True))
                features[indices]=model.features(x).cpu()
                if (batch+1)%50==0:print(f'Whole-strip feature extraction {min((batch+1)*16,len(rows))}/{len(rows)}',flush=True)
        assert torch.isfinite(features).all()
        torch.save({'features':features,'source_sha256':source['model_state_sha256'],'manifest_sha256':meta['manifest_sha256']},feature_path)
    features=features.cuda();labels=torch.tensor([r['label'] for r in rows],dtype=torch.float32,device='cuda')
    train=torch.tensor(selection['train'],device='cuda');dev=torch.tensor(selection['development'],device='cuda')
    optimizer=torch.optim.AdamW(model.detector.parameters(),lr=3e-4,weight_decay=.01)
    best=(-1,-1);history=[]
    for epoch in range(1,101):
        model.detector.train();losses=[]
        order=train[torch.randperm(len(train),device='cuda')]
        for indices in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            logits=model.detector(features[indices]).squeeze(1)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels[indices])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.detector.parameters(),5);optimizer.step();losses.append(loss.item())
        model.eval()
        with torch.inference_mode():scores=model.detector(features[dev]).sigmoid().squeeze(1).cpu().numpy()
        metrics=development_metrics(labels[dev].cpu().numpy(),scores)
        history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),**metrics})
        key=(metrics['recall'],metrics['auroc'])
        if key>best:
            best=key;torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                'epoch':epoch,'development':metrics,'runtime':runtime,'manifest_sha256':meta['manifest_sha256'],
                                'model_sha256':state_hash(model.state_dict()),'phase':'frozen-feature-head'},OUTPUT/'head_warmup.pt')
        if epoch%10==0:print(json.dumps(history[-1]),flush=True)
    (OUTPUT/'head_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(json.dumps({'best_development_recall_at5pct_fpr':best[0],'corresponding_auroc':best[1]}),flush=True)


if __name__=='__main__':main()
