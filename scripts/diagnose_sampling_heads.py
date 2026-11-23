"""Inspect head activations on a fixed first-ten-per-label development subset."""
import csv,json,os,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.compare_tile_sampling import BASE,NAMES,read
from scripts.train_steel_unet import predict_outputs
from scripts.run_steel_pilot import read_image
from src.resnet_unet import JointResNet18UNet
from src.reproducibility import configure


def main():
    configure(42,True);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest=read('configs/severstal_manifest.json');rows={r['image_id']:r for r in manifest['samples']}
    selected=read(f'configs/{BASE}_selection.json')['evaluation']
    chosen=[i for label in (0,1) for i in [n for n in selected if rows[n]['label']==label][:10]]
    result={}
    for policy,name in NAMES.items():
        saved=torch.load(ROOT/f'runs/{name}/best.pt',map_location=device,weights_only=True)
        model=JointResNet18UNet().to(device);model.load_state_dict(saved['model']);model.eval()
        stats={'tile_rows':0,'all_zero_hidden_rows':0,'nonzero_activations':0,'hidden_values':0,
               'maximum_pre_relu':None};ever=torch.zeros(128,dtype=torch.bool,device=device)
        def before(module,inputs,output):
            value=float(output.max().item());stats['maximum_pre_relu']=value if stats['maximum_pre_relu'] is None else max(value,stats['maximum_pre_relu'])
        def after(module,inputs,output):
            nonzero=output>0;ever.logical_or_(nonzero.any(0))
            stats['tile_rows']+=output.shape[0];stats['all_zero_hidden_rows']+=int((~nonzero.any(1)).sum().item())
            stats['nonzero_activations']+=int(nonzero.sum().item());stats['hidden_values']+=output.numel()
        hooks=[model.classifier[0].register_forward_hook(before),model.classifier[1].register_forward_hook(after)]
        with (ROOT/f'runs/{name}/validation_predictions.csv').open(encoding='utf-8-sig',newline='') as f:pred={r['image_id']:r for r in csv.DictReader(f)}
        for i in chosen:
            _,score=predict_outputs(model,read_image(rows[i]),device)
            np.testing.assert_allclose(score,float(pred[i]['classification_score']),rtol=1e-5,atol=1e-5)
        for hook in hooks:hook.remove()
        stats.update({'ever_active_units_of_128':int(ever.sum().item()),
                      'final_linear_weight_nonzero':int(model.classifier[2].weight.count_nonzero().item()),
                      'final_bias':float(model.classifier[2].bias.item()),
                      'sigmoid_final_bias':float(model.classifier[2].bias.sigmoid().item())})
        result[policy]=stats;del saved,model
    report={'scope':'Saved best checkpoints; fixed 10 normal + 10 defective evaluation images, no retraining/no test',
            'image_ids':chosen,'results':result,
            'limits':'Activation observations apply to these 20 images/240 tiles, not every training sample or proof of when/why units became inactive.'}
    (ROOT/'reports/steel_sampling_head_diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
