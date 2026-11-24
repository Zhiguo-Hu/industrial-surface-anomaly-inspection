"""Native-resolution EfficientNetV2-S inspection with a small U-Net decoder."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_v2_s
from src.unet import block

OFFICIAL_SHA256='dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74'


class EfficientStripInspector(nn.Module):
    def __init__(self,weights_path=None):
        super().__init__()
        backbone=efficientnet_v2_s(weights=None,stochastic_depth_prob=.1)
        if weights_path is not None:
            path=Path(weights_path)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=OFFICIAL_SHA256:raise ValueError('Official backbone checksum mismatch')
            backbone.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        self.encoder=backbone.features
        self.d4=block(1280+160,128);self.d3=block(128+64,64)
        self.d2=block(64+48,32);self.d1=block(32+24,16);self.out=nn.Conv2d(16,1,1)
        self.detector=nn.Sequential(nn.LayerNorm(3008),nn.Linear(3008,256),nn.SiLU(),nn.Dropout(.15),nn.Linear(256,1))
        self.auxiliary=nn.Sequential(nn.LayerNorm(3008),nn.Linear(3008,4))

    def encode(self,x):
        skips=[]
        for index,layer in enumerate(self.encoder):
            x=layer(x)
            if index in (1,2,3,5):skips.append(x)
        return x,tuple(skips)

    def pool_features(self,encoded,skips):
        return torch.cat([v for f in (skips[2],skips[3],encoded) for v in (f.mean((-2,-1)),f.amax((-2,-1)))],1)

    def features(self,x):
        encoded,skips=self.encode(x);return self.pool_features(encoded,skips)

    def decode(self,x,skips,size):
        for skip,decoder in zip(reversed(skips),(self.d4,self.d3,self.d2,self.d1)):
            x=decoder(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        return F.interpolate(self.out(x),size=size,mode='bilinear',align_corners=False)

    def forward_joint(self,x,return_aux=False):
        encoded,skips=self.encode(x);pooled=self.pool_features(encoded,skips)
        mask=self.decode(encoded,skips,x.shape[-2:]);logit=self.detector(pooled).squeeze(1)
        return (mask,logit,self.auxiliary(pooled)) if return_aux else (mask,logit)

    def forward(self,x):return self.forward_joint(x)[0]
