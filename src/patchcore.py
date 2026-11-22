"""Lightweight PatchCore-style model; deviations documented in the stage report."""
import torch
from torch import nn
from torch.nn import functional as F


class LocalFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                  backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        a = self.layer2(self.stem(x))
        b = self.layer3(a)
        a = F.avg_pool2d(a, 3, stride=1, padding=1)
        b = F.avg_pool2d(b, 3, stride=1, padding=1)
        b = F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([a, b], dim=1)


@torch.inference_mode()
def coreset_indices(features, count, seed=42, projection_dim=32):
    if not 0 < count <= len(features):
        raise ValueError('Invalid coreset size')
    generator = torch.Generator(device=features.device).manual_seed(seed)
    projection = torch.randn(features.shape[1], projection_dim,
                             generator=generator, device=features.device) / projection_dim**0.5
    projected = features @ projection
    minimum = torch.full((len(features),), float('inf'), device=features.device)
    selected = torch.zeros(len(features), dtype=torch.bool, device=features.device)
    index = int(torch.randint(len(features), (1,), generator=generator, device=features.device).item())
    indices = []
    for _ in range(count):
        indices.append(index)
        selected[index] = True
        distances = (projected-projected[index]).square().sum(dim=1)
        minimum = torch.minimum(minimum, distances)
        index = int(minimum.masked_fill(selected, -1).argmax().item())
    return torch.tensor(indices, device=features.device)


@torch.inference_mode()
def patch_distances(query, bank, query_chunk=256, bank_chunk=2048):
    if not len(bank) or query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError('Invalid query/bank')
    parts = []
    for q in query.split(query_chunk):
        best = torch.full((len(q),), float('inf'), device=q.device)
        for b in bank.split(bank_chunk):
            best = torch.minimum(best, torch.cdist(q, b).amin(dim=1))
        parts.append(best)
    return torch.cat(parts)


def anomaly_maps(distances, batch, height, width, output_size=256):
    low = distances.reshape(batch, 1, height, width)
    maps = F.interpolate(low, size=(output_size, output_size), mode='bilinear', align_corners=False)
    axis = torch.arange(-12, 13, device=maps.device, dtype=maps.dtype)
    kernel = torch.exp(-axis.square()/(2*4**2))
    kernel = kernel/kernel.sum()
    maps = F.conv2d(F.pad(maps, (12,12,0,0), mode='replicate'), kernel.view(1,1,1,-1))
    maps = F.conv2d(F.pad(maps, (0,0,12,12), mode='replicate'), kernel.view(1,1,-1,1))
    return low.flatten(1).amax(dim=1), maps[:,0]
