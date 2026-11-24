"""Native-resolution BF16 joint training with exact update-boundary pause/resume."""
import argparse,copy,hashlib,json,math,os,random,sys,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved,normalize_images,mask_objective
from src.strip_cache import CachedStrips
from src.inspection_protocol import partition,development_metrics
from src.reproducibility import configure,state_hash

OUT=ROOT/'runs/steel_efficientnet_v2_s';CACHE=ROOT/'runs/strip_training_cache'


def atomic_save(value,path):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


@torch.inference_mode()
def evaluate(model,loader,rows):
    model.eval();labels=[];scores=[]
    for x,_,positions in loader:
        value=model.detector(model.features(normalize_images(x.cuda(non_blocking=True)))).squeeze(1)
        scores.extend(value.cpu().tolist());labels.extend(rows[i]['label'] for i in positions.tolist())
    return development_metrics(labels,scores)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.epochs<1:parser.error('Positive epoch count required')
    if (OUT/'last.pt').exists() and not args.resume:parser.error('Training state already exists; use --resume')
    if (OUT/'pause_requested').exists() and not args.resume:parser.error('Paused run requires --resume')
    if args.resume:(OUT/'pause_requested').unlink(missing_ok=True)
    runtime=configure(42,True);torch.set_num_threads(4);assert torch.cuda.is_bf16_supported()
    meta=json.loads((CACHE/'metadata.json').read_text());rows=meta['rows'];selection=partition(rows)
    assert len(selection['train'])==8800 and not any(r['split']=='test' for r in rows)
    assert meta['manifest_sha256']==hashlib.sha256((ROOT/'configs/severstal_manifest.json').read_bytes()).hexdigest()
    profile=json.loads((OUT/'compute_profile.json').read_text());micro=profile['micro_batch'];accumulation=profile['accumulation']
    assert micro*accumulation==16 and 8800%16==0
    warm=torch.load(OUT/'head_warmup.pt',map_location='cpu',weights_only=True)
    assert warm['model_sha256']==state_hash(warm['model']) and warm['manifest_sha256']==meta['manifest_sha256']
    model=inspector_from_saved(warm).cuda();ema=copy.deepcopy(model).eval().requires_grad_(False)
    decoder=[p for name in ('d4','d3','d2','d1','out') for p in getattr(model,name).parameters()]
    heads=list(model.detector.parameters())+list(model.auxiliary.parameters())
    rates=[1e-4,4e-4,3e-4]
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':rates[0]},
                                {'params':decoder,'lr':rates[1]},{'params':heads,'lr':rates[2]}],weight_decay=1e-4)
    classes=torch.tensor([[float(c in r['classes']) for c in (1,2,3,4)] for r in rows],device='cuda')
    counts=classes[selection['train']].sum(0);weights=((8800-counts)/counts).clamp(max=20)
    config={'training_schema':1,'epochs':args.epochs,'micro_batch':micro,'accumulation':accumulation,'precision':'bfloat16',
            'encoder_lr':rates[0],'decoder_lr':rates[1],'head_lr':rates[2],'mask_loss_weight':.5,'class_loss_weight':.2,
            'class_positive_weights':weights.cpu().tolist(),'native_size_wh':[1600,256],'seed':42,
            'warm_model_sha256':warm['model_sha256'],'manifest_sha256':meta['manifest_sha256'],
            'early_qualification_gate':{'minimum_epoch':3,'consecutive_epochs':2,'development_recall_min':.97,'development_auroc_min':.99,'maximum_attempts':1}}
    epoch_start=1;skip_batches=0;steps=0;history=[];best=(-1.,-1.)
    if args.resume:
        saved=torch.load(OUT/'last.pt',map_location='cpu',weights_only=True);assert saved['training_config']==config
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);optimizer.load_state_dict(saved['optimizer'])
        epoch_start=saved['epoch'];skip_batches=saved['next_batch'];steps=saved['steps'];history=saved['history'];best=tuple(saved['best'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);random.setstate(saved['python_rng'])
    (OUT/'training_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    devloader=DataLoader(CachedStrips(CACHE,selection['development']),batch_size=8,num_workers=2,pin_memory=True,persistent_workers=True,
                         generator=torch.Generator().manual_seed(9001))
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();total_steps=args.epochs*550

    def checkpoint(epoch,next_batch):
        atomic_save({'training_schema':1,'training_config':config,'model':model.state_dict(),'ema':ema.state_dict(),
                     'optimizer':optimizer.state_dict(),'epoch':epoch,'next_batch':next_batch,'steps':steps,'history':history,'best':best,
                     'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()},OUT/'last.pt')

    for epoch in range(epoch_start,args.epochs+1):
        loader=DataLoader(CachedStrips(CACHE,selection['train']),batch_size=micro,shuffle=True,
                          generator=torch.Generator().manual_seed(1000+epoch),num_workers=2,pin_memory=True)
        model.train();losses=[];optimizer.zero_grad(set_to_none=True)
        for batch,(x,y,positions) in enumerate(loader):
            if epoch==epoch_start and batch<skip_batches:continue
            x=x.cuda(non_blocking=True).float();y=y.cuda(non_blocking=True).float()
            if random.random()<.5:x=x.flip(-1);y=y.flip(-1)
            if random.random()<.5:x=x.flip(-2);y=y.flip(-2)
            contrast=x.new_empty((len(x),1,1,1)).uniform_(.9,1.1);brightness=x.new_empty((len(x),1,1,1)).uniform_(-8,8)
            x=normalize_images(((x-127.5)*contrast+127.5+brightness).clamp_(0,255))
            ratio=.1+.9*(1+math.cos(math.pi*min(1,steps/total_steps)))/2;warmup=min(1,(steps+1)/100)
            for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*ratio*warmup
            with torch.autocast('cuda',dtype=torch.bfloat16):
                encoded,skips=model.encode(x);pooled=model.pool_features(encoded,skips)
                image_logit=model.detector(pooled).squeeze(1).float();aux=model.auxiliary(pooled).float()
                classification=torch.nn.functional.binary_cross_entropy_with_logits(image_logit,y.flatten(1).any(1).float())
                class_loss=torch.nn.functional.binary_cross_entropy_with_logits(aux,classes[positions.cuda()],pos_weight=weights)
                n=max(1,len(x)//2);mask=model.decode(encoded[:n],tuple(s[:n] for s in skips),x.shape[-2:])
                mask_loss,_=mask_objective(mask,y[:n]);loss=classification+.2*class_loss+.5*mask_loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; previous atomic checkpoint is intact')
            (loss/accumulation).backward();losses.append(float(loss.item()))
            if (batch+1)%accumulation:continue
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
            with torch.no_grad():
                source=model.state_dict()
                for name,target in ema.state_dict().items():
                    if target.is_floating_point():target.lerp_(source[name],.005)
                    else:target.copy_(source[name])
            if steps%50==0:
                status={'stage':'training','epoch':epoch,'epochs':args.epochs,'micro_batch':batch+1,'micro_batches':len(loader),
                        'steps':steps,'loss_last50_micro_batches':float(np.mean(losses[-50:])),
                        'elapsed_seconds':time.perf_counter()-started,'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20}
                (OUT/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(status),flush=True)
            if steps%100==0:checkpoint(epoch,batch+1)
            if (OUT/'pause_requested').exists():
                checkpoint(epoch,batch+1)
                (OUT/'progress.json').write_text(json.dumps({'stage':'paused_by_user','epoch':epoch,'next_batch':batch+1,'steps':steps}),encoding='utf-8')
                print('PAUSED_AT_SAVED_UPDATE_BOUNDARY',flush=True);return
        metric=evaluate(ema,devloader,rows);history.append({'epoch':epoch,'steps':steps,'training_loss':float(np.mean(losses)),**metric})
        key=(metric['recall'],metric['auroc'])
        if key>best:
            best=key;state={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()}
            atomic_save({'model_kind':'efficientnet_v2_s','model':state,'model_sha256':state_hash(state),'epoch':epoch,'steps':steps,
                         'manifest_sha256':meta['manifest_sha256'],'development':metric,'training_config':config,'runtime':runtime,
                         'pretrained_sha256':warm['pretrained_sha256'],'phase':'native-full-strip-EfficientNetV2S-EMA'},OUT/'best.pt')
        (OUT/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print('EPOCH '+json.dumps(history[-1]),flush=True);checkpoint(epoch+1,0);skip_batches=0
        if epoch>=3 and not (OUT/'early_qualification_checked').exists() and len(history)>=2:
            if all(r['recall']>=.97 and r['auroc']>=.99 for r in history[-2:]):
                result={'stage':'early_qualification_ready','completed_epochs':epoch,'planned_max_epochs':args.epochs,'steps':steps,
                        'best_development_recall':best[0],'best_development_auroc':best[1],'test_used':False}
                (OUT/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True);return
    result={'stage':'training_complete','epochs':args.epochs,'steps':steps,'best_development_recall':best[0],
            'best_development_auroc':best[1],'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20,'test_used':False}
    (OUT/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'training_complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
