"""ResNet18 U-Net with matched random/pretrained encoder variants."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from src.unet import block


class ResNet18UNet(nn.Module):
    def __init__(self,weights_path=None):
        super().__init__()
        self.encoder=resnet18(weights=None)
        self.encoder.fc=nn.Identity()
        self.d4=block(512+256,128);self.d3=block(128+128,64)
        self.d2=block(64+64,32);self.d1=block(32+64,16)
        self.out=nn.Conv2d(16,1,1)
        # Load AFTER constructing every layer, preserving identical decoder initialization.
        if weights_path:
            digest=hashlib.sha256(weights_path.read_bytes()).hexdigest()
            if digest!='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec':
                raise ValueError('ResNet18 weight checksum mismatch')
            state=torch.load(weights_path,map_location='cpu',weights_only=True)
            state.pop('fc.weight');state.pop('fc.bias')
            self.encoder.load_state_dict(state,strict=True)

    def encode(self,x):
        e=self.encoder
        a=e.relu(e.bn1(e.conv1(x)))
        b=e.layer1(e.maxpool(a));c=e.layer2(b);d=e.layer3(c);x=e.layer4(d)
        return x,(a,b,c,d)

    def decode(self,x,skips,size):
        a,b,c,d=skips
        for skip,decoder in [(d,self.d4),(c,self.d3),(b,self.d2),(a,self.d1)]:
            x=decoder(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        return F.interpolate(self.out(x),size=size,mode='bilinear',align_corners=False)

    def forward(self,x):
        encoded,skips=self.encode(x)
        return self.decode(encoded,skips,x.shape[-2:])


class JointResNet18UNet(ResNet18UNet):
    """Tile classification from shared encoder; masks remain ungated by default."""
    def __init__(self,weights_path=None):
        super().__init__(weights_path)
        # Initialize last to preserve all segmentation initialization.
        self.classifier=nn.Sequential(nn.Linear(1024,128),nn.ReLU(),nn.Linear(128,1))

    def forward_joint(self,x):
        encoded,skips=self.encode(x)
        pooled=torch.cat([encoded.mean(dim=(-2,-1)),encoded.amax(dim=(-2,-1))],dim=1)
        classification=self.classifier(pooled).squeeze(1)
        return self.decode(encoded,skips,x.shape[-2:]),classification
