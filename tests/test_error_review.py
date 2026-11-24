import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.review_frozen_errors import frozen_maps
from src.error_review import case_evidence, patch_start, select_errors
from src.reproducibility import state_hash
from src.strip_inspector import normalize_images


class TinyInspector(nn.Module):
    """CPU fixture with both deep and shallow classifier branches."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Conv2d(3,4,3,padding=1)
        self.detector=nn.Linear(8,1)
        self.mask=nn.Conv2d(4,1,1)

    def encode(self,x):
        shallow=self.encoder(x).relu()
        return F.avg_pool2d(shallow,2),(shallow,)

    def pool_features(self,encoded,skips):
        return torch.cat([encoded.mean((-2,-1)),skips[0].mean((-2,-1))],1)

    def decode(self,encoded,skips,size):
        return F.interpolate(self.mask(encoded),size=size,mode='bilinear',align_corners=False)


class ErrorReviewTests(unittest.TestCase):
    def test_archived_decision_boundary_and_corruption(self):
        rows=[{'label':1,'prediction':0,'score':0}, {'label':0,'prediction':1,'score':2},
              {'label':0,'prediction':0,'score':-1}, {'label':1,'prediction':1,'score':1}]
        errors=select_errors(rows,0)
        self.assertEqual([(r['kind'],r['score']) for r in errors],[('FN',0.),('FP',2.)])
        for row in ({'label':0,'prediction':1,'score':0}, {'label':1,'prediction':0,'score':float('nan')}):
            with self.assertRaises(ValueError):select_errors([row],0)

    def test_overlap_distinguishes_no_alarm_from_no_localization(self):
        truth=np.zeros((16,32),bool);truth[4:8,10:14]=True
        probability=truth.astype(np.float32)
        e=case_evidence(np.full((16,32,3),100,np.uint8),truth,probability,probability,
                        [-2,-1,1,2],-1,0,.4)
        self.assertEqual(e['pixel_dice'],1.)
        self.assertEqual(e['pixel_recall'],1.)
        self.assertIn('localized_overlap_without_image_alarm',e['descriptive_flags'])
        normal=case_evidence(np.zeros((16,32,3),np.uint8),np.zeros_like(truth),probability,
                             probability,[1],1,0,.4)
        self.assertIsNone(normal['pixel_dice'])
        self.assertIsNone(normal['pixel_recall'])

    def test_native_crop_preserves_targets_at_both_ends(self):
        for column,expected in ((0,0),(1599,1344)):
            truth=np.zeros((256,1600),bool);truth[100,column]=True
            start=patch_start(truth,np.zeros(truth.shape),np.zeros(truth.shape))
            self.assertEqual(start,expected)
            self.assertTrue(truth[:,start:start+256].any())

    def test_attribution_replay_preserves_frozen_model_and_four_view_score(self):
        torch.manual_seed(42)
        model=TinyInspector().eval().requires_grad_(False)
        array=np.random.default_rng(42).integers(0,256,(16,24,3),dtype=np.uint8)
        before=state_hash(model.state_dict())
        score,probability,cam,views=frozen_maps(model,array,{'flip4':True},'cpu')
        x=normalize_images(torch.from_numpy(array.transpose(2,0,1).copy())[None])
        expected=[]
        with torch.no_grad():
            for dims in ((),(-1,),(-2,),(-2,-1)):
                encoded,skips=model.encode(x.flip(dims) if dims else x)
                expected.append(model.detector(model.pool_features(encoded,skips)).item())
        np.testing.assert_allclose(views,expected,atol=1e-6)
        self.assertAlmostEqual(score,float(np.mean(expected)),places=6)
        self.assertEqual(probability.shape,(16,24));self.assertEqual(cam.shape,(16,24))
        self.assertTrue(np.isfinite(cam).all())
        self.assertTrue((cam>=0).all() and (cam<=1).all())
        self.assertEqual(state_hash(model.state_dict()),before)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))


if __name__=='__main__':unittest.main()
