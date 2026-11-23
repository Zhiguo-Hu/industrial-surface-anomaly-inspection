"""Full-data/full-strip refinement; image operating-point selection, mask auxiliaries."""
import argparse,copy,json,math,os,random,sys,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import StripInspector,mask_objective,normalize_images
from src.strip_cache import CachedStrips,guided_crop
from src.inspection_protocol import partition,development_metrics
from src.reproducibility import configure,state_hash

OUTPUT=ROOT/'runs/strip_inspector';CACHE=ROOT/'runs/strip_training_cache'


def atomic_save(value,path):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


@torch.inference_mode()
def evaluate(model,loader,rows):
    model.eval();scores=[];labels=[];records=[]
    for x,_,indices in loader:
        x=normalize_images(x.cuda(non_blocking=True))
        with torch.autocast('cuda',dtype=torch.float16):logits=model.detector(model.features(x)).squeeze(1)
        values=logits.float().sigmoid().cpu().tolist()
        for i,score in zip(indices.tolist(),values):
            labels.append(rows[i]['label']);scores.append(score);records.append({'image_id':rows[i]['image_id'],'label':rows[i]['label'],'score':score})
    return development_metrics(labels,scores),records


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=12);parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--run-name',default='strip_inspector')
    parser.add_argument('--warm-checkpoint',type=Path);parser.add_argument('--crop-width',type=int,default=0)
    parser.add_argument('--crop-probability',type=float,default=.5);parser.add_argument('--mask-every',type=int,default=4)
    parser.add_argument('--auxiliary-classes',action='store_true')
    parser.add_argument('--encoder-lr',type=float,default=7e-5);parser.add_argument('--decoder-lr',type=float,default=2e-4)
    parser.add_argument('--detector-lr',type=float,default=2e-4);args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:parser.error('run-name must be a simple directory name')
    if args.mask_every<1 or not 0<=args.crop_probability<=1:parser.error('Invalid auxiliary/crop schedule')
    if args.crop_width and not 32<=args.crop_width<=1600:parser.error('Invalid crop width')
    OUTPUT=ROOT/'runs'/args.run_name
    runtime=configure(42,True);torch.set_num_threads(4);OUTPUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((CACHE/'metadata.json').read_text());rows=meta['rows'];selection=partition(rows)
    warm_path=args.warm_checkpoint or OUTPUT/'head_warmup.pt'
    model=StripInspector(auxiliary_classes=args.auxiliary_classes).cuda();warm=torch.load(warm_path,map_location='cpu',weights_only=True)
    model.warm_start(warm);ema=copy.deepcopy(model).eval().requires_grad_(False)
    decoder=list(model.d4.parameters())+list(model.d3.parameters())+list(model.d2.parameters())+list(model.d1.parameters())+list(model.out.parameters())
    heads=list(model.detector.parameters())+(list(model.auxiliary.parameters()) if args.auxiliary_classes else [])
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':args.encoder_lr},
                                {'params':decoder,'lr':args.decoder_lr},{'params':heads,'lr':args.detector_lr}],weight_decay=1e-4)
    base_rates=[args.encoder_lr,args.decoder_lr,args.detector_lr];scaler=torch.amp.GradScaler('cuda')
    run_config={k:v for k,v in vars(args).items() if k not in ['warm_checkpoint','resume']}
    run_config['warm_model_sha256']=state_hash(warm['model'])
    assert run_config['warm_model_sha256']==warm.get('model_sha256',warm.get('model_state_sha256'))
    counts=np.array([sum(c in rows[i]['classes'] for i in selection['train']) for c in (1,2,3,4)])
    class_weights=torch.tensor(np.minimum(5,np.sqrt((len(selection['train'])-counts)/np.maximum(1,counts))),device='cuda',dtype=torch.float32)
    run_config['auxiliary_positive_weights']=class_weights.cpu().tolist()
    (OUTPUT/'training_config.json').write_text(json.dumps(run_config,indent=2),encoding='utf-8')
    devloader=DataLoader(CachedStrips(CACHE,selection['development']),batch_size=16,num_workers=2,pin_memory=True,persistent_workers=True)
    epoch_start=1;skip_batches=0;steps=0;skipped=0;best=(-1.,-1.);history=[]
    if args.resume:
        saved=torch.load(OUTPUT/'last.pt',map_location='cpu',weights_only=True)
        assert saved.get('training_schema')==2,'Incompatible loss-version checkpoint'
        if 'training_config' in saved:assert saved['training_config']==run_config
        assert saved['manifest_sha256']==meta['manifest_sha256']
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        epoch_start=saved['epoch'];skip_batches=saved['next_batch'];steps=saved['steps'];skipped=saved['skipped'];best=tuple(saved['best']);history=saved['history']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);random.setstate(saved['python_rng'])
    total_steps=math.ceil(len(selection['train'])/args.batch_size)*args.epochs;start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epoch_start,args.epochs+1):
        generator=torch.Generator().manual_seed(1000+epoch)
        loader=DataLoader(CachedStrips(CACHE,selection['train']),batch_size=args.batch_size,shuffle=True,
                          generator=generator,num_workers=2,pin_memory=True,persistent_workers=False)
        model.train();losses=[]
        for batch_number,(x,y,positions) in enumerate(loader):
            if epoch==epoch_start and batch_number<skip_batches:continue
            cropped=bool(args.crop_width and random.random()<args.crop_probability)
            if cropped:x,y=guided_crop(x,y,args.crop_width,random)
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True).float();x=x.float()
            if random.random()<.5:x=x.flip(-1);y=y.flip(-1)
            if random.random()<.5:x=x.flip(-2);y=y.flip(-2)
            contrast=x.new_empty((len(x),1,1,1)).uniform_(.85,1.15);brightness=x.new_empty((len(x),1,1,1)).uniform_(-12,12)
            x=((x-127.5)*contrast+127.5+brightness).clamp_(0,255);x=normalize_images(x)
            ratio=.1+.9*(1+math.cos(math.pi*min(1,steps/max(1,total_steps))))/2
            warmup=min(1,(steps+1)/100)
            for group,rate in zip(optimizer.param_groups,base_rates):group['lr']=rate*ratio*warmup
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                encoded,skips=model.encode(x);pooled=model.pool_features(encoded,skips);cls=model.detector(pooled).squeeze(1)
                labels=y.flatten(1).any(1).float()
                cls_loss=torch.nn.functional.binary_cross_entropy_with_logits(cls.float(),labels)
                loss=cls_loss
                if args.auxiliary_classes and not cropped:
                    target=x.new_tensor([[float(c in rows[i]['classes']) for c in (1,2,3,4)] for i in positions.tolist()])
                    auxiliary_logits=model.auxiliary(pooled).float()
                    loss=loss+.2*torch.nn.functional.binary_cross_entropy_with_logits(auxiliary_logits,target,pos_weight=class_weights)
                if steps%args.mask_every==0:
                    n=min(8,len(x));mask=model.decode(encoded[:n],tuple(s[:n] for s in skips),x.shape[-2:])
                    auxiliary,_=mask_objective(mask,y[:n]);loss=loss+auxiliary
            if not torch.isfinite(loss):raise ValueError('Non-finite strip training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();overflow=scaler.get_scale()<old_scale
            skipped+=int(overflow);steps+=1
            if not overflow:
                with torch.no_grad():
                    source_state=model.state_dict()
                    for name,target in ema.state_dict().items():
                        value=source_state[name]
                        if target.is_floating_point():target.lerp_(value,.005)
                        else:target.copy_(value)
            losses.append(float(loss.item()))
            if steps%50==0:
                status={'stage':'training','epoch':epoch,'batch':batch_number+1,'batches':len(loader),'steps':steps,
                        'loss_last50':float(np.mean(losses[-50:])),'amp_skips':skipped,
                        'elapsed_seconds':time.perf_counter()-start}
                (OUTPUT/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(status),flush=True)
            if steps%200==0:
                atomic_save({'training_schema':2,'training_config':run_config,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                             'epoch':epoch,'next_batch':batch_number+1,'steps':steps,'skipped':skipped,'best':best,'history':history,
                             'manifest_sha256':meta['manifest_sha256'],'torch_rng':torch.get_rng_state(),
                             'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()},OUTPUT/'last.pt')
        metrics,records=evaluate(ema,devloader,rows)
        history.append({'epoch':epoch,'steps':steps,'mean_training_loss':float(np.mean(losses)),**metrics})
        key=(metrics['recall'],metrics['auroc'])
        if key>best:
            best=key;atomic_save({'training_schema':2,'training_config':run_config,'model':{k:v.cpu().clone() for k,v in ema.state_dict().items()},'epoch':epoch,'steps':steps,
                                 'development':metrics,'manifest_sha256':meta['manifest_sha256'],'runtime':runtime,
                                 'model_sha256':state_hash(ema.state_dict()),'phase':'full-strip-EMA'},OUTPUT/'best.pt')
            (OUTPUT/'best_development_predictions.json').write_text(json.dumps(records),encoding='utf-8')
        (OUTPUT/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print('EPOCH '+json.dumps(history[-1]),flush=True)
        atomic_save({'training_schema':2,'training_config':run_config,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                     'epoch':epoch+1,'next_batch':0,'steps':steps,'skipped':skipped,'best':best,'history':history,
                     'manifest_sha256':meta['manifest_sha256'],'torch_rng':torch.get_rng_state(),
                     'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()},OUTPUT/'last.pt')
        skip_batches=0
    print(json.dumps({'stage':'training_complete','best_development_recall':best[0],'best_development_auroc':best[1],
                      'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)


if __name__=='__main__':main()
