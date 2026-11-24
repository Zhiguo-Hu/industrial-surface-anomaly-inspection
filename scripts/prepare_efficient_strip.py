"""Profile BF16 training, extract frozen train/val features and warm task heads."""
import copy,json,os,sys,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.efficient_strip import EfficientStripInspector,OFFICIAL_SHA256
from src.strip_inspector import normalize_images,mask_objective
from src.strip_cache import CachedStrips
from src.inspection_protocol import partition,development_metrics
from src.reproducibility import configure,state_hash

OUT=ROOT/'runs/steel_efficientnet_v2_s';CACHE=ROOT/'runs/strip_training_cache'
WEIGHTS=ROOT/'weights/efficientnet_v2_s-dd5fe13b.pth'


def profile_batch(batch_size,rows):
    model=EfficientStripInspector(WEIGHTS).cuda().train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    data=CachedStrips(CACHE,list(range(batch_size)));samples=[data[i] for i in range(batch_size)]
    x=normalize_images(torch.stack([a for a,_,_ in samples]).cuda());target=torch.stack([b for _,b,_ in samples]).cuda().float()
    torch.cuda.reset_peak_memory_stats();times=[]
    for step in range(3):
        torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            encoded,skips=model.encode(x);pooled=model.pool_features(encoded,skips)
            logit=model.detector(pooled).squeeze(1).float();labels=target.flatten(1).any(1).float()
            mask=model.decode(encoded[:batch_size//2],tuple(s[:batch_size//2] for s in skips),x.shape[-2:])
            mask_loss,_=mask_objective(mask,target[:batch_size//2])
            aux=model.auxiliary(pooled).float();cls=x.new_tensor([[float(c in rows[i]['classes']) for c in (1,2,3,4)] for i in range(batch_size)])
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,labels)+.5*mask_loss+.2*torch.nn.functional.binary_cross_entropy_with_logits(aux,cls)
        assert torch.isfinite(loss);loss.backward()
        assert torch.isfinite(model.out.weight.grad).all() and model.detector[-1].weight.grad.abs().sum()>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
    return {'micro_batch':batch_size,'accumulation':16//batch_size,'effective_batch':16,
            'micro_step_seconds_warmed':float(np.mean(times[1:])),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'precision':'bfloat16','finite_loss_and_gradients':True}


def main():
    runtime=configure(42,True);torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    assert torch.cuda.is_bf16_supported(),'Hardware BF16 required for this run'
    meta=json.loads((CACHE/'metadata.json').read_text());rows=meta['rows'];selection=partition(rows)
    assert len(rows)==10686 and not any(r['split']=='test' for r in rows)
    attempts=[];profile=None
    for size in (8,4):
        try:
            result=profile_batch(size,rows);attempts.append(result)
            if result['peak_allocated_mib']<5200:profile=result;break
        except torch.cuda.OutOfMemoryError:
            attempts.append({'micro_batch':size,'result':'capacity_probe_reduced_batch'})
        finally:torch.cuda.empty_cache()
    if profile is None:raise RuntimeError('No profiled configuration within GPU memory budget')
    profile={**profile,'attempts':attempts,'gpu':torch.cuda.get_device_name(0),'pretrained_sha256':OFFICIAL_SHA256}
    (ROOT/'reports/efficientnet_compute_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    (OUT/'compute_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8');print('PROFILE '+json.dumps(profile),flush=True)
    configure(42,True);model=EfficientStripInspector(WEIGHTS).cuda().eval();source_sha=state_hash(model.encoder.state_dict())
    feature_path=OUT/'frozen_features.pt'
    if feature_path.exists():
        saved=torch.load(feature_path,map_location='cpu',weights_only=True)
        assert saved['manifest_sha256']==meta['manifest_sha256'] and saved['encoder_sha256']==source_sha
        features=saved['features']
    else:
        features=torch.empty(len(rows),3008,dtype=torch.float32)
        loader=DataLoader(CachedStrips(CACHE,range(len(rows))),batch_size=16,num_workers=2,pin_memory=True)
        with torch.inference_mode():
            for batch,(x,_,positions) in enumerate(loader):
                with torch.autocast('cuda',dtype=torch.bfloat16):pooled=model.features(normalize_images(x.cuda(non_blocking=True)))
                features[positions]=pooled.float().cpu()
                if (batch+1)%80==0:print(f'Frozen features {(batch+1)*16}/{len(rows)}',flush=True)
        assert torch.isfinite(features).all()
        torch.save({'features':features,'manifest_sha256':meta['manifest_sha256'],'encoder_sha256':source_sha,'test_used':False},feature_path)
    train=torch.tensor(selection['train'],device='cuda');dev=torch.tensor(selection['development'],device='cuda')
    labels=torch.tensor([r['label'] for r in rows],dtype=torch.float32,device='cuda')
    classes=torch.tensor([[float(c in r['classes']) for c in (1,2,3,4)] for r in rows],device='cuda')
    counts=classes[train].sum(0);positive_weight=((len(train)-counts)/counts).clamp(max=20)
    x=features.cuda();optimizer=torch.optim.AdamW(list(model.detector.parameters())+list(model.auxiliary.parameters()),lr=5e-4,weight_decay=1e-3)
    best=(-1.,-1.);history=[]
    for epoch in range(1,101):
        model.detector.train();model.auxiliary.train();order=train[torch.randperm(len(train),device='cuda')]
        for positions in order.split(256):
            pooled=x[positions];binary=model.detector(pooled).squeeze(1);aux=model.auxiliary(pooled)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(binary,labels[positions])+.2*torch.nn.functional.binary_cross_entropy_with_logits(aux,classes[positions],pos_weight=positive_weight)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(model.detector.parameters())+list(model.auxiliary.parameters()),5);optimizer.step()
        model.eval()
        with torch.inference_mode():scores=model.detector(x[dev]).squeeze(1).cpu().numpy()
        metric=development_metrics(labels[dev].cpu().numpy(),scores);history.append({'epoch':epoch,**metric})
        key=(metric['recall'],metric['auroc'])
        if key>best:
            best=key;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'model':state,'model_kind':'efficientnet_v2_s','model_sha256':state_hash(state),'manifest_sha256':meta['manifest_sha256'],
                        'epoch':epoch,'steps':0,'development':metric,'runtime':runtime,'pretrained_sha256':OFFICIAL_SHA256,'phase':'frozen-encoder-head-warmup'},OUT/'head_warmup.pt')
        if epoch%10==0:print('HEAD '+json.dumps(history[-1]),flush=True)
    (OUT/'head_warmup_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print('PREPARATION_COMPLETE '+json.dumps({'best_recall':best[0],'best_auroc':best[1],'micro_batch':profile['micro_batch'],'test_used':False}),flush=True)


if __name__=='__main__':main()
