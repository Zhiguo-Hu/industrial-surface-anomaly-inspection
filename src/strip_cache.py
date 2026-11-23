"""Disk-backed RGB and bit-packed masks; no repeated JPEG decode/hash in training."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class CachedStrips(Dataset):
    def __init__(self,folder,indices):
        self.folder=Path(folder);self.indices=list(indices);self.images=None;self.masks=None
        meta=json.loads((self.folder/'metadata.json').read_text())
        if not meta['complete']:raise ValueError('Strip cache is incomplete')
        self.width=meta['width']

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        if self.images is None:
            self.images=np.load(self.folder/'images.npy',mmap_mode='r')
            self.masks=np.load(self.folder/'masks.npy',mmap_mode='r')
        i=self.indices[index]
        image=torch.from_numpy(np.array(self.images[i].transpose(2,0,1),copy=True))
        mask=torch.from_numpy(np.unpackbits(self.masks[i],axis=-1,count=self.width).copy()).unsqueeze(0)
        return image,mask,i


def guided_crop(images,masks,width,rng):
    original=images.shape[-1]
    if not 1<=width<=original:raise ValueError('Invalid guided-crop width')
    cropped_images=[];cropped_masks=[]
    for image,mask in zip(images,masks):
        columns=torch.where(mask[0].bool().any(0))[0]
        if len(columns):
            anchor=int(columns[rng.randrange(len(columns))]);left=max(0,anchor-width+1);right=min(anchor,original-width)
            start=rng.randint(left,right)
        else:start=rng.randint(0,original-width)
        cropped_images.append(image[...,start:start+width]);cropped_masks.append(mask[...,start:start+width])
    return torch.stack(cropped_images),torch.stack(cropped_masks)
