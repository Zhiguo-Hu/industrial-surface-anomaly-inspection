"""Steel pilot selections, bounded candidate extraction, and full-strip inference."""
import random
import numpy as np
import torch
from torchvision.transforms.functional import normalize, to_tensor
from src.patchcore import anomaly_maps, patch_distances
from src.tiling import boxes, crop_tile, Stitcher


def pilot_selection(rows, seed=42):
    rng=random.Random(seed)
    groups={}
    for name,split,label in [('train','train',0),('normal_val','val',0),('defect_val','val',1)]:
        group=sorted([r for r in rows if r['split']==split and r['label']==label],key=lambda r:r['image_id'])
        rng.shuffle(group);groups[name]=group
    if len(groups['train'])<300 or len(groups['normal_val'])<300 or len(groups['defect_val'])<100:
        raise ValueError('Insufficient data for fixed pilot')
    return {'train':groups['train'][:300], 'calibration':groups['normal_val'][:200],
            'evaluation':groups['normal_val'][200:300]+groups['defect_val'][:100]}


def tile_batches(image,device):
    layout=boxes(image.shape[0],image.shape[1])
    for offset in range(0,len(layout),8):
        group=layout[offset:offset+8]
        batch=torch.stack([normalize(to_tensor(crop_tile(image,b)),[0.485,0.456,0.406],
                                     [0.229,0.224,0.225]) for b in group]).to(device)
        yield batch,group


@torch.inference_mode()
def candidates(model,image,device,generator,per_tile=32):
    parts=[]
    for batch,_ in tile_batches(image,device):
        f=model(batch).permute(0,2,3,1).flatten(1,2)
        for tile in f:
            index=torch.randperm(len(tile),generator=generator)[:per_tile].to(device)
            parts.append(tile[index].cpu())
    return torch.cat(parts)


@torch.inference_mode()
def infer(model,bank,image,device):
    stitch=Stitcher(*image.shape[:2])
    for batch,layout in tile_batches(image,device):
        f=model(batch);n,c,h,w=f.shape
        distances=patch_distances(f.permute(0,2,3,1).reshape(-1,c),bank)
        _,maps=anomaly_maps(distances,n,h,w)
        for m,b in zip(maps.cpu().numpy(),layout):stitch.add(m,b)
    result=stitch.finish()
    return float(result.max()),result


def heat_rgb(heat,scale):
    values=np.clip(heat/scale,0,1)
    colors=np.array([[15,25,110],[0,180,220],[255,220,0],[220,0,0]])
    return np.stack([np.interp(values,[0,.33,.66,1],colors[:,i]) for i in range(3)],axis=-1).astype(np.uint8)
