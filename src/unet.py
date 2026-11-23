"""Small U-Net trained from scratch for binary surface-defect segmentation."""
import torch
from torch import nn
from torch.nn import functional as F


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU())


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1=block(3,16);self.e2=block(16,32);self.e3=block(32,64);self.middle=block(64,128)
        self.d3=block(192,64);self.d2=block(96,32);self.d1=block(48,16);self.out=nn.Conv2d(16,1,1)

    def forward(self,x):
        a=self.e1(x);b=self.e2(F.max_pool2d(a,2));c=self.e3(F.max_pool2d(b,2))
        x=self.middle(F.max_pool2d(c,2))
        for skip,decoder in [(c,self.d3),(b,self.d2),(a,self.d1)]:
            x=decoder(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],dim=1))
        return self.out(x)


def segmentation_loss(logits,mask):
    bce=F.binary_cross_entropy_with_logits(logits,mask)
    probability=logits.sigmoid()
    intersection=(probability*mask).sum((1,2,3))
    dice=(2*intersection+1)/(probability.sum((1,2,3))+mask.sum((1,2,3))+1)
    return bce+(1-dice).mean()
