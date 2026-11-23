"""Local inference with a frozen full-strip candidate or accepted deployment artifact."""
import argparse,json,os,sys,time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inspector import inspector_from_saved
from src.strip_inference import inspect_image
from src.reproducibility import configure,state_hash
from src.steel import heat_rgb


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--image',required=True,type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/strip_inspector/predictions');args=parser.parse_args()
    configure(42,True);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.model is None:
        pointer=ROOT/'configs/strip_deployment_current.json'
        args.model=ROOT/json.loads(pointer.read_text())['model'] if pointer.exists() else ROOT/'runs/strip_inspector/deployment/model.pt'
    saved=torch.load(args.model,map_location='cpu',weights_only=True)
    if state_hash(saved['model'])!=saved['model_sha256']:raise ValueError('Model integrity check failed')
    model=inspector_from_saved(saved).to(device).eval();policy=saved['inspection_policy']
    with Image.open(args.image) as im:array=np.asarray(im.convert('RGB')).copy()
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter();score,probability=inspect_image(model,array,device,policy['flip4'],policy.get('fusion'))
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start;args.output_dir.mkdir(parents=True,exist_ok=True)
    Image.fromarray((.6*array+.4*heat_rgb(probability,1)).astype(np.uint8)).save(args.output_dir/'overlay.png')
    Image.fromarray((probability>saved.get('pixel_threshold',.5)).astype(np.uint8)*255).save(args.output_dir/'mask.png')
    np.save(args.output_dir/'pixel_scores.npy',probability)
    result={'image':str(args.image.resolve()),'prediction':int(score>policy['threshold']),'score':score,
            'threshold':policy['threshold'],'inference_seconds_cold':elapsed,'device':device,
            'dataset_acceptance':saved.get('acceptance',{}).get('status','not_final_tested'),
            'pixel_mask_ungated':True,'score_is_calibrated_probability':False}
    (args.output_dir/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
