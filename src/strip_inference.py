"""Shared FP32 deployment path: same full-image score as qualification."""
import hashlib,json
import numpy as np
import torch
from src.strip_inspector import normalize_images

MASK_FEATURE_NAMES=['image_logit','mask_max_logit','top41_mean_logit','top410_mean_logit','top4096_mean_logit',
                    'mask_area_above_0p1','mask_area_above_0p3','mask_area_above_0p5','mask_area_above_0p7']
SEMANTIC_FEATURE_NAMES=MASK_FEATURE_NAMES+[f'class{i}_logit' for i in (1,2,3,4)]
PYRAMID_FEATURE_NAMES=['crop800_max_logit','crop800_mean_logit','crop800_min_logit']


def feature_names(semantic=False,pyramid=False):
    return (SEMANTIC_FEATURE_NAMES if semantic else MASK_FEATURE_NAMES)+(PYRAMID_FEATURE_NAMES if pyramid else [])


def ensemble_feature_names(first,second):
    return [f'member{i}/{name}' for i,names in enumerate((first,second)) for name in names]


def public_policy(policy):
    """Publish method/provenance, retaining fitted fusion parameters in local artifacts."""
    result={k:v for k,v in policy.items() if k!='fusion'}
    if policy.get('fusion') is not None:
        fusion=policy['fusion'];result['fusion']={'method':fusion.get('method','linear'),'feature_names':fusion['feature_names'],
            'parameters_sha256':hashlib.sha256(json.dumps(fusion,sort_keys=True).encode()).hexdigest(),
            'parameters_saved_with_local_candidate':True}
        if 'support_vectors' in fusion:result['fusion']['support_vector_count']=len(fusion['support_vectors'])
    return result


@torch.inference_mode()
def pyramid_logits(model,x):
    """Three overlapping native-resolution windows; every source column is covered."""
    if x.shape[-1]!=1600:raise ValueError('Pyramid requires width1600')
    values=torch.stack([image_logits(model,x[...,start:start+800]) for start in (0,400,800)],1)
    return torch.stack([values.max(1).values,values.mean(1),values.min(1).values],1)


def mask_score_features(logit,segmentation):
    flat=segmentation.float().flatten(1);top=flat.topk(min(4096,flat.shape[1]),dim=1).values
    values=[logit[:,None],top[:,:1],top[:,:min(41,top.shape[1])].mean(1,keepdim=True),
            top[:,:min(410,top.shape[1])].mean(1,keepdim=True),top.mean(1,keepdim=True)]
    for p in [.1,.3,.5,.7]:values.append((flat>np.log(p/(1-p))).float().mean(1,keepdim=True))
    return torch.cat(values,1)


def fused_score(features,spec):
    schemas=[feature_names(s,p) for s in (False,True) for p in (False,True)]
    supported=schemas+[ensemble_feature_names(a,b) for a in schemas for b in schemas]
    if spec['feature_names'] not in supported:raise ValueError('Unsupported inspection feature schema')
    if features.shape[1]!=len(spec['feature_names']):raise ValueError('Feature count does not match fusion schema')
    x=features.double();mean=x.new_tensor(spec['mean']);scale=x.new_tensor(spec['scale'])
    x=(x-mean)/scale
    method=spec.get('method','linear')
    if method=='linear':return x@x.new_tensor(spec['coefficient'])+spec['intercept']
    if method=='rbf_svm':
        support=x.new_tensor(spec['support_vectors']);distance=(x[:,None,:]-support[None,:,:]).square().sum(-1)
        return torch.exp(-spec['gamma']*distance)@x.new_tensor(spec['dual_coefficient'])+spec['intercept']
    raise ValueError('Unsupported decision fusion method')


@torch.inference_mode()
def image_logits(model,x,flip4=False):
    views=[(),(-1,),(-2,),(-2,-1)] if flip4 else [()]
    values=[model.detector(model.features(x.flip(d) if d else x)).squeeze(1) for d in views]
    return torch.stack(values).mean(0)


@torch.inference_mode()
def inspection_scores(model,x,policy):
    if policy.get('fusion') is None:return image_logits(model,x,policy['flip4'])
    features,_=decision_features(model,x,policy['flip4'],policy['fusion']['feature_names'])
    return fused_score(features,policy['fusion'])


@torch.inference_mode()
def decision_features(model,x,flip4,names):
    if hasattr(model,'members'):
        values=[]
        for i,member in enumerate(model.members):
            member_names=[name[len(f'member{i}/'):] for name in names if name.startswith(f'member{i}/')]
            features,mask=decision_features(member,x,flip4,member_names);values.append(features)
        return torch.cat(values,1),mask
    semantic='class1_logit' in names;pyramid='crop800_max_logit' in names
    if semantic:mask,original,aux=model.forward_joint(x,return_aux=True);auxvalues=[aux]
    else:mask,original=model.forward_joint(x)
    values=[original]
    if flip4:
        for dimensions in [(-1,),(-2,),(-2,-1)]:
            features=model.features(x.flip(dimensions));values.append(model.detector(features).squeeze(1))
            if semantic:auxvalues.append(model.auxiliary(features))
    logit=torch.stack(values).mean(0)
    features=mask_score_features(logit,mask)
    if semantic:features=torch.cat([features,torch.stack(auxvalues).mean(0)],1)
    if pyramid:features=torch.cat([features,pyramid_logits(model,x)],1)
    return features,mask


@torch.inference_mode()
def inspect_image(model,array,device,flip4=False,fusion=None):
    if array.shape!=(256,1600,3):raise ValueError('This candidate requires RGB steel strips of width1600, height256')
    x=normalize_images(torch.from_numpy(np.array(array.transpose(2,0,1),copy=True))[None].to(device))
    features,mask=decision_features(model,x,flip4,fusion['feature_names'] if fusion else MASK_FEATURE_NAMES)
    score=fused_score(features,fusion) if fusion else features[:,0]
    return float(score.item()),mask[0,0].float().sigmoid().cpu().numpy()
