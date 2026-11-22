"""Frozen ResNet18 global embeddings and nearest-normal-neighbor scoring."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18


def preprocessing(size):
    # Preserve the full field of view; no center crop. Bottle originals are square.
    return transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class ImageDataset(Dataset):
    def __init__(self, root, rows, size):
        self.root, self.rows = Path(root), rows
        self.transform = preprocessing(size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]['image']) as image:
            return self.transform(image.convert('RGB'))


def load_model(weights, device):
    digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
    if digest != 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec':
        raise ValueError('Official ResNet18 weights checksum mismatch')
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.fc = nn.Identity()
    model.requires_grad_(False)
    return model.to(device).eval()


@torch.inference_mode()
def extract(model, loader, device):
    batches = []
    for batch in loader:
        features = model(batch.to(device))
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        if not torch.isfinite(features).all():
            raise ValueError('Non-finite model features')
        batches.append(features.cpu())
    if not batches:
        raise ValueError('Empty feature dataset')
    return torch.cat(batches)


def nearest_scores(query, bank, chunk_size=64):
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError('Expected matching 2-D feature matrices')
    if not len(query) or not len(bank) or chunk_size < 1:
        raise ValueError('Empty features or invalid chunk size')
    if not torch.isfinite(query).all() or not torch.isfinite(bank).all():
        raise ValueError('Non-finite features')
    # CPU distance computation is sufficient for the small 167-vector bank.
    query, bank = query.cpu().float(), bank.cpu().float()
    scores, indices = [], []
    for chunk in query.split(chunk_size):
        values, index = torch.cdist(chunk, bank, p=2,
                                   compute_mode='donot_use_mm_for_euclid_dist').min(dim=1)
        scores.append(values)
        indices.append(index)
    return torch.cat(scores).numpy(), torch.cat(indices).numpy()


def validation_threshold(scores, quantile=0.95):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected finite nonempty validation scores')
    return float(np.quantile(scores, quantile, method='linear'))


def validate_manifest(rows):
    if any(r['split'] not in {'train', 'val', 'test'} for r in rows):
        raise ValueError('Unknown split')
    for key in ['image', 'sha256']:
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate {key} across manifest')
    for split in ['train', 'val', 'test']:
        subset = [r for r in rows if r['split'] == split]
        if not subset:
            raise ValueError(f'Empty split: {split}')
        if split != 'test' and any(r['label'] != 0 or r['official_split'] != 'train' for r in subset):
            raise ValueError('Train/validation must use official normal training images only')
        if split == 'test' and any(r['official_split'] != 'test' for r in subset):
            raise ValueError('Test must use official test images only')
