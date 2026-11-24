"""Full-strip supervised detector with stable multiscale image head."""
import torch
from torch import nn
from torch.nn import functional as F
from src.resnet_unet import ResNet18UNet


class StripInspector(ResNet18UNet):
    def __init__(self,auxiliary_classes=False):
        super().__init__()
        self.detector=nn.Sequential(nn.LayerNorm(1792),nn.Linear(1792,256),nn.SiLU(),nn.Dropout(.1),nn.Linear(256,1))
        if auxiliary_classes:self.auxiliary=nn.Sequential(nn.LayerNorm(1792),nn.Linear(1792,4))

    def forward_joint(self,x,return_aux=False):
        encoded,skips=self.encode(x)
        pooled=self.pool_features(encoded,skips)
        mask=self.decode(encoded,skips,x.shape[-2:]);classification=self.detector(pooled).squeeze(1)
        if return_aux:return mask,classification,self.auxiliary(pooled) if hasattr(self,'auxiliary') else None
        return mask,classification

    def pool_features(self,encoded,skips):
        scales=[skips[2],skips[3],encoded]
        return torch.cat([v for f in scales for v in [f.mean((-2,-1)),f.amax((-2,-1))]],1)

    def features(self,x):
        encoded,skips=self.encode(x)
        return self.pool_features(encoded,skips)

    def warm_start(self,saved):
        state={k:v for k,v in saved['model'].items() if not k.startswith('classifier.')}
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith(('detector.','auxiliary.')) for k in missing):raise ValueError('Incompatible warm-start model')


def inspector_from_saved(saved):
    if saved.get('model_kind')=='strip_ensemble':
        from src.strip_ensemble import StripEnsemble
        states=[{k[len(f'members.{i}.'):]:v for k,v in saved['model'].items() if k.startswith(f'members.{i}.')} for i in (0,1)]
        if sum(map(len,states))!=len(saved['model']) or not all(states):raise ValueError('Invalid ensemble state')
        return StripEnsemble([inspector_from_saved({'model':state}) for state in states])
    model=StripInspector(auxiliary_classes=any(k.startswith('auxiliary.') for k in saved['model']))
    model.load_state_dict(saved['model'],strict=True)
    return model


def mask_objective(segmentation,mask):
    # Full strips exceed half-precision reduction range; always reduce in FP32.
    logits=segmentation.float();target=mask.float();probability=logits.sigmoid()
    positive=target.flatten(1).any(1)
    bce=F.binary_cross_entropy_with_logits(logits,target)
    intersection=(probability*target).sum((1,2,3))
    dice=(2*intersection+1)/(probability.sum((1,2,3))+target.sum((1,2,3))+1)
    dice_loss=(1-dice)[positive].mean() if positive.any() else probability.sum()*0
    return .5*bce+.5*dice_loss,{'pixel_bce':bce.detach(),'positive_dice':dice_loss.detach()}


def strip_loss(segmentation,classification,mask):
    auxiliary,parts=mask_objective(segmentation,mask)
    classification_loss=F.binary_cross_entropy_with_logits(classification.float(),mask.flatten(1).any(1).float())
    return classification_loss+auxiliary,{'classification':classification_loss.detach(),**parts}


def normalize_images(x):
    x=x.float()/255
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1)
    std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    return (x-mean)/std
