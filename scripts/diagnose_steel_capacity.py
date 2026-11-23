"""Single-variable validation ablation: 2048 -> 8192 memory features; never deploys changes."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.baseline import load_model,validation_threshold
from src.patchcore import LocalFeatures,coreset_indices
from src.steel import candidates,infer
from scripts.run_steel_pilot import read_image


def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=LocalFeatures(load_model(ROOT/'weights/resnet18-f37072fd.pth',device)).to(device).eval()
    saved=torch.load(ROOT/'runs/steel_pilot/bank.pt',map_location='cpu',weights_only=True)
    manifest=json.loads((ROOT/'configs/severstal_manifest.json').read_text())
    rows={r['image_id']:r for r in manifest['samples']}
    selected=json.loads((ROOT/'configs/steel_pilot_selection.json').read_text())
    assert all(rows[i]['split']=='train' and rows[i]['label']==0 for i in selected['train'])
    assert all(rows[i]['split']=='val' for k in ['calibration','evaluation'] for i in selected[k])
    generator=torch.Generator().manual_seed(42);parts=[]
    for i,name in enumerate(selected['train']):
        parts.append(candidates(model,read_image(rows[name]),device,generator))
        if (i+1)%100==0:print(f'Candidate rebuild {i+1}/300',flush=True)
    pool=torch.cat(parts).to(device);del parts
    indices=coreset_indices(pool,8192)
    bank=pool[indices].clone();del pool
    # The expanded bank must have the exact original bank as its prefix.
    torch.testing.assert_close(bank[:2048].cpu(),saved['bank'],rtol=0,atol=0)
    cal=[];scores=[]
    for group,target in [('calibration',cal),('evaluation',scores)]:
        for i,name in enumerate(selected[group]):
            score,_=infer(model,bank,read_image(rows[name]),device);target.append(score)
            if (i+1)%50==0:print(f'Expanded {group} {i+1}/200',flush=True)
    threshold=validation_threshold(cal)
    labels=np.array([rows[i]['label'] for i in selected['evaluation']])
    tn,fp,fn,tp=confusion_matrix(labels,np.array(scores)>threshold,labels=[0,1]).ravel()
    result={'scope':'Exploratory validation-only capacity ablation; original deployed pilot unchanged',
            'changed_variable':'2048 to 8192 coreset vectors; candidate pool, selection order, data, model, map and scoring unchanged',
            'exact_original_bank_prefix_check':'passed', 'bank_shape':list(bank.shape),
            'threshold':threshold,'image_auroc':float(roc_auc_score(labels,scores)),
            'confusion':dict(zip(['tn','fp','fn','tp'],map(int,[tn,fp,fn,tp]))),
            'recall':float(tp/(tp+fn)),'false_positive_rate':float(fp/(fp+tn)),
            'test_images_accessed':False,'original_bank_modified':False}
    text=json.dumps(result,indent=2)
    (ROOT/'reports/steel_capacity_ablation.json').write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':main()
