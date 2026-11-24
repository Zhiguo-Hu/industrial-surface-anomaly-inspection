"""Independent-process score reload and output checks on two development images."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.strip_inference import fused_score


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-name',default='strip_inspector_refined');args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name:parser.error('Invalid run name')
    output=ROOT/'runs'/args.run_name;torch.set_num_threads(2)
    saved=torch.load(output/'candidate.pt',map_location='cpu',weights_only=True);policy=saved['inspection_policy']
    meta=json.loads((ROOT/'runs/strip_training_cache/metadata.json').read_text());lookup={r['image_id']:r for r in meta['rows']}
    features=np.load(output/'fusion_features.npz',allow_pickle=False)
    assert str(features['model_sha256'])==saved['model_sha256']
    all_names=features['feature_names'].tolist();columns=[all_names.index(n) for n in policy['fusion']['feature_names']]
    matrix=features['flip4' if policy['flip4'] else 'single'][:,columns];checks=[]
    for position in (0,200):
        image_id=str(features['image_ids'][position]);row=lookup[image_id];assert row['split']=='val'
        target=output/f'reload_check_{position}'
        completed=subprocess.run([sys.executable,str(ROOT/'scripts/inspect_steel_strip.py'),'--image',str(ROOT/row['image']),
                                  '--model',str(output/'candidate.pt'),'--output-dir',str(target)],cwd=ROOT,capture_output=True,text=True)
        if completed.returncode:raise RuntimeError(completed.stderr)
        prediction=json.loads((target/'prediction.json').read_text())
        expected=float(fused_score(torch.from_numpy(matrix[position:position+1]),policy['fusion']).item())
        difference=abs(prediction['score']-expected)
        assert difference<.001,('Independent score discrepancy',difference)
        assert prediction['prediction']==int(expected>policy['threshold'])
        scores=np.load(target/'pixel_scores.npy',allow_pickle=False)
        assert scores.shape==(256,1600) and np.isfinite(scores).all() and scores.min()>=0 and scores.max()<=1
        with Image.open(target/'overlay.png') as overlay:assert overlay.size==(1600,256)
        with Image.open(target/'mask.png') as mask:assert mask.size==(1600,256) and set(np.unique(mask)).issubset({0,255})
        checks.append({'image_id':image_id,'label':row['label'],'prediction':prediction['prediction'],
                       'score':prediction['score'],'cached_qualification_score':expected,'absolute_score_difference':difference,
                       'finite_full_resolution_outputs':True})
    assert [r['label'] for r in checks]==[0,1]
    report={'status':'passed','model_sha256':saved['model_sha256'],'checks':checks,
            'scope':'Independent CLI processes on one normal and one defective development image; scores compared to batch8 FP32 qualification features; no test access'}
    (ROOT/'reports/strip_artifact_reload.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
