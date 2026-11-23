"""Self-contained normal steel memory: frozen domain encoder and auditable vectors."""
import torch
from torch import nn
from torchvision.models import resnet18
from src.patchcore import LocalFeatures,coreset_indices
from src.reproducibility import state_hash
from src.steel import tile_batches


def normal_training_rows(rows):
    selected=sorted([r for r in rows if r['split']=='train' and r['label']==0],key=lambda r:r['image_id'])
    if not selected or any(r['classes'] for r in selected):raise ValueError('Normal train identities/labels inconsistent')
    if len({r['image_id'] for r in selected})!=len(selected) or len({r['sha256'] for r in selected})!=len(selected):
        raise ValueError('Duplicate normal training identity/content')
    return selected


def domain_extractor(checkpoint):
    if state_hash(checkpoint['model'])!=checkpoint['model_state_sha256']:raise ValueError('Source checkpoint checksum mismatch')
    backbone=resnet18(weights=None);backbone.fc=nn.Identity()
    backbone.load_state_dict({k[len('encoder.'):]:v for k,v in checkpoint['model'].items() if k.startswith('encoder.')},strict=True)
    return LocalFeatures(backbone).eval()


def standalone_extractor(state):
    model=LocalFeatures(resnet18(weights=None));model.load_state_dict(state,strict=True)
    return model.eval()


def sample_grid(features,image_index,tile_offset,generator,per_tile=32):
    n,c,h,w=features.shape
    if not 0<per_tile<=h*w:raise ValueError('Invalid feature sampling size')
    values=[];references=[]
    for j,tile in enumerate(features.permute(0,2,3,1).flatten(1,2)):
        chosen=torch.randperm(h*w,generator=generator)[:per_tile]
        values.append(tile[chosen.to(tile.device)].detach().cpu())
        references.extend([[image_index,tile_offset+j,int(p)//w,int(p)%w] for p in chosen])
    return torch.cat(values).contiguous(),torch.tensor(references,dtype=torch.int64)


@torch.inference_mode()
def image_candidates(model,image,device,image_index,per_tile=32,seed=42):
    generator=torch.Generator().manual_seed(seed+image_index);parts=[];references=[];offset=0
    for batch,layout in tile_batches(image,device):
        f=model(batch)
        if not torch.isfinite(f).all():raise ValueError('Non-finite normal features')
        values,refs=sample_grid(f,image_index,offset,generator,per_tile)
        parts.append(values);references.append(refs);offset+=len(layout)
    return torch.cat(parts),torch.cat(references)


@torch.inference_mode()
def select_vectors(features,references,count,seed,device):
    if len(features)!=len(references):raise ValueError('Feature provenance length mismatch')
    chosen=coreset_indices(features.to(device),min(count,len(features)),seed=seed).cpu()
    return features[chosen].clone(),references[chosen].clone()


def validate_bank(saved,rows,manifest_hash=None):
    if saved.get('schema')!=1 or not saved.get('complete'):raise ValueError('Memory bank is not complete')
    expected=normal_training_rows(rows)
    if saved['normal_image_ids']!=[r['image_id'] for r in expected]:raise ValueError('Memory sources are not the canonical normal train set')
    if manifest_hash is not None and saved['manifest_sha256']!=manifest_hash:raise ValueError('Dataset manifest changed')
    bank,refs=saved['bank'],saved['provenance']
    if bank.ndim!=2 or bank.shape[1]!=384 or not len(bank) or not torch.isfinite(bank).all():raise ValueError('Invalid memory feature matrix')
    if refs.shape!=(len(bank),4) or refs.dtype!=torch.int64:raise ValueError('Invalid memory provenance')
    if (refs<0).any() or (refs[:,0]>=len(expected)).any() or (refs[:,1]>=12).any() or (refs[:,2:]>=32).any():
        raise ValueError('Memory provenance out of bounds')
    if state_hash({'bank':bank})!=saved['bank_sha256']:raise ValueError('Memory tensor checksum mismatch')
    if state_hash({'provenance':refs})!=saved['provenance_sha256']:raise ValueError('Memory provenance checksum mismatch')
    if state_hash(saved['feature_model'])!=saved['feature_model_sha256']:raise ValueError('Feature model checksum mismatch')
    for key in ['threshold','visual_scale']:
        if not torch.isfinite(torch.tensor(saved[key])) or saved[key]<0:raise ValueError('Invalid calibrated memory metadata')
    if saved['visual_scale']==0:raise ValueError('Visualization scale must be positive')
    return True
