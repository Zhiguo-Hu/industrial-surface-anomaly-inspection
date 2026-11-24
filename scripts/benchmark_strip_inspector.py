"""Measure warmed single-image inference on a development image, never final test."""
import argparse,json,os,sys,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved
from src.strip_inference import inspect_image,public_policy
from src.inspection_protocol import partition
from src.reproducibility import configure,state_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-name',default='strip_inspector_refined')
    parser.add_argument('--iterations',type=int,default=20);args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name or args.iterations<5:parser.error('Invalid run or iteration count')
    configure(42,True);torch.set_num_threads(4)
    output=ROOT/'runs'/args.run_name;saved=torch.load(output/'candidate.pt',map_location='cpu',weights_only=True)
    assert state_hash(saved['model'])==saved['model_sha256']
    model=inspector_from_saved(saved).cuda().eval();policy=saved['inspection_policy']
    cache=ROOT/'runs/strip_training_cache';meta=json.loads((cache/'metadata.json').read_text())
    index=partition(meta['rows'])['development'][0]
    array=np.array(np.load(cache/'images.npy',mmap_mode='r')[index],copy=True)
    for _ in range(5):inspect_image(model,array,'cuda',policy['flip4'],policy.get('fusion'))
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(args.iterations):
        start=time.perf_counter();inspect_image(model,array,'cuda',policy['flip4'],policy.get('fusion'));torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
    report={'model_sha256':saved['model_sha256'],'gpu':torch.cuda.get_device_name(0),'precision':'float32',
            'batch_size':1,'image_size_wh':[1600,256],'warmup':5,'iterations':args.iterations,
            'latency_ms_median':float(np.median(times)),'latency_ms_p95':float(np.quantile(times,.95)),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'policy':public_policy(policy),
            'scope':'In-memory RGB -> preprocessing/transfer -> frozen scoring -> CPU score and raw mask; excludes JPEG decode, disk writes and camera. One development image; no test access.'}
    (ROOT/'reports/strip_inference_benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
