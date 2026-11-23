"""On-demand tiles; never split tiles independently of their source image."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize, to_tensor

from scripts.audit_severstal import decode_rle
from src.tiling import boxes, crop_tile


class SeverstalTiles(Dataset):
    def __init__(self, root, split='train', normal_only=False, size=256, stride=128):
        if split not in {'train','val','test'}:
            raise ValueError('Unknown split')
        self.root=Path(root)
        manifest=json.loads((self.root/'configs/severstal_manifest.json').read_text(encoding='utf-8'))
        annotation=self.root/manifest['annotation_csv']
        if hashlib.sha256(annotation.read_bytes()).hexdigest()!=manifest['source_csv_sha256']:
            raise ValueError('Annotation changed since fixed split')
        self.rows=[r for r in manifest['samples'] if r['split']==split and (not normal_only or r['label']==0)]
        if not self.rows:raise ValueError('No selected samples')
        self.annotations=defaultdict(list)
        with annotation.open(encoding='utf-8-sig',newline='') as file:
            for r in csv.DictReader(file):
                if r['EncodedPixels'].strip():self.annotations[r['ImageId']].append(r['EncodedPixels'])
        self.size,self.stride=size,stride
        self.layout=boxes(256,1600,size,stride)
        self.split=split
        self._cache_index=None

    def __len__(self):
        return len(self.rows)*len(self.layout)

    def __getitem__(self,index):
        if index<0 or index>=len(self):raise IndexError(index)
        image_index,tile_index=divmod(index,len(self.layout))
        row=self.rows[image_index]
        if self._cache_index!=image_index:
            path=self.root/row['image']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError(f'Image changed: {path}')
            with Image.open(path) as image:
                if image.size!=(1600,256):raise ValueError('Unexpected Severstal size')
                self._image=np.asarray(image.convert('RGB')).copy()
            self._mask=np.zeros((256,1600),dtype=np.uint8)
            for text in self.annotations[row['image_id']]:self._mask|=decode_rle(text)
            self._cache_index=image_index
        box=self.layout[tile_index]
        image=crop_tile(self._image,box,self.size)
        mask=crop_tile(self._mask,box,self.size,is_mask=True)
        return {'image':normalize(to_tensor(image),[0.485,0.456,0.406],[0.229,0.224,0.225]),
                'mask':torch.from_numpy(mask.copy()).unsqueeze(0).float(),
                'box':torch.tensor(box), 'image_id':row['image_id'], 'split':row['split'],
                'image_label':row['label'], 'tile_label':int(mask.any()),
                'valid_hw':torch.tensor([box[3]-box[1],box[2]-box[0]])}
