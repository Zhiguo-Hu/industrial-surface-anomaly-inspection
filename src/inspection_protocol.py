"""Locked development/calibration/qualification partitions and operating metrics."""
import random
import hashlib,json
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix


def partition(rows):
    rng=random.Random(42);groups={}
    for label in (0,1):
        group=sorted([i for i,r in enumerate(rows) if r['split']=='val' and r['label']==label],key=lambda i:rows[i]['image_id'])
        rng.shuffle(group);groups[label]=group
    selection={'train':[i for i,r in enumerate(rows) if r['split']=='train'],
               'development':groups[0][:200]+groups[1][:300],
               'calibration':groups[0][200:600],
               'qualification':groups[0][600:]+groups[1][300:]}
    sets=[set(v) for v in selection.values()]
    if any(a&b for j,a in enumerate(sets) for b in sets[j+1:]):raise ValueError('Inspection partition leakage')
    if any(rows[i]['split']=='test' for v in selection.values() for i in v):raise ValueError('Test data in development protocol')
    return selection


def operating_metrics(labels,scores,threshold):
    labels=np.asarray(labels);scores=np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all() or not np.isfinite(threshold):raise ValueError('Non-finite prediction')
    tn,fp,fn,tp=confusion_matrix(labels,scores>threshold,labels=[0,1]).ravel()
    return {'auroc':float(roc_auc_score(labels,scores)),'threshold':float(threshold),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'recall':float(tp/max(1,tp+fn)),'fpr':float(fp/max(1,fp+tn))}


def development_metrics(labels,scores):
    labels=np.asarray(labels);scores=np.asarray(scores)
    threshold=float(np.quantile(scores[labels==0],.95,method='higher'))
    return operating_metrics(labels,scores,threshold)


def frozen_test_fingerprint(model_sha256,policy):
    return hashlib.sha256(json.dumps({'model':model_sha256,'flip4':policy['flip4'],'threshold':policy['threshold'],
                                    'fusion':policy.get('fusion')},sort_keys=True).encode()).hexdigest()


def verify_test_identity(previous,requested):
    if previous is not None and previous!=requested:raise ValueError('Final test already belongs to a different frozen candidate')
    return True
