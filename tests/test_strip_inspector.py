import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from src.strip_inspector import StripInspector,strip_loss,normalize_images,inspector_from_saved
from src.strip_cache import CachedStrips,guided_crop
import random
from src.inspection_protocol import partition,operating_metrics,development_metrics,frozen_test_fingerprint,verify_test_identity
from src.strip_inference import MASK_FEATURE_NAMES,mask_score_features,fused_score,inspect_image,inspection_scores,feature_names,pyramid_logits,ensemble_feature_names,decision_features
from src.strip_ensemble import StripEnsemble
from src.efficient_strip import EfficientStripInspector


class StripInspectorTests(unittest.TestCase):
    def test_joint_shape_and_both_gradients(self):
        torch.set_num_threads(2);model=StripInspector()
        mask,cls=model.forward_joint(torch.randn(2,3,64,128))
        target=torch.zeros_like(mask);target[0,:,:,10:20]=1
        loss,_=strip_loss(mask,cls,target);loss.backward()
        self.assertEqual(mask.shape,(2,1,64,128));self.assertEqual(cls.shape,(2,))
        self.assertTrue(torch.isfinite(loss));self.assertGreater(model.detector[-1].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.out.weight.grad.abs().sum().item(),0)

    def test_full_strip_half_precision_reductions_stay_finite(self):
        logits=torch.zeros(2,1,256,1600,dtype=torch.float16,requires_grad=True)
        target=torch.ones_like(logits);target[1]=0
        loss,parts=strip_loss(logits,torch.zeros(2,requires_grad=True),target);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(parts['positive_dice'].item(),0)

    def test_auxiliary_does_not_duplicate_classification_gradient(self):
        logits=torch.zeros(2,1,8,16,requires_grad=True);cls=torch.tensor([.2,-.4],requires_grad=True)
        target=torch.zeros_like(logits);target[0,:,:,4:6]=1
        loss,_=strip_loss(logits,cls,target)
        actual=torch.autograd.grad(loss,cls)[0]
        expected=(cls.sigmoid()-torch.tensor([1.,0.]))/2
        torch.testing.assert_close(actual,expected)

    def test_cache_roundtrip_preserves_colors_masks_and_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);image=np.arange(2*8*16*3,dtype=np.uint8).reshape(2,8,16,3)
            mask=np.zeros((2,8,16),dtype=np.uint8);mask[1,:,3:7]=1
            np.save(p/'images.npy',image);np.save(p/'masks.npy',np.packbits(mask,axis=-1))
            (p/'metadata.json').write_text(json.dumps({'complete':True,'width':16}))
            x,y,i=CachedStrips(p,[1])[0]
            np.testing.assert_array_equal(x.numpy(),image[1].transpose(2,0,1));np.testing.assert_array_equal(y[0].numpy(),mask[1])
            self.assertEqual(i,1);self.assertTrue(torch.isfinite(normalize_images(x[None])).all())

    def test_protocol_excludes_test_and_keeps_partitions_disjoint(self):
        rows=[{'image_id':str(i),'label':int(i>=885),'split':'val'} for i in range(1886)]
        rows+=[{'image_id':'train','label':1,'split':'train'},{'image_id':'test','label':1,'split':'test'}]
        p=partition(rows);groups=[set(v) for v in p.values()]
        self.assertTrue(all(not a&b for j,a in enumerate(groups) for b in groups[j+1:]))
        self.assertEqual(len(p['development']),500);self.assertEqual(len(p['calibration']),400)
        self.assertFalse(any(rows[i]['split']=='test' for g in groups for i in g))

    def test_constant_scores_do_not_pass_recall_gate(self):
        metric=development_metrics([0,0,1,1],[.5,.5,.5,.5])
        self.assertEqual(metric['recall'],0);self.assertEqual(metric['auroc'],.5)
        good=operating_metrics([0,0,1,1],[.1,.2,.8,.9],.5)
        self.assertEqual(good['recall'],1);self.assertEqual(good['fpr'],0)

    def test_final_test_identity_rejects_model_and_threshold_changes(self):
        a=frozen_test_fingerprint('model-a',{'flip4':False,'threshold':1.})
        self.assertTrue(verify_test_identity(None,a));self.assertTrue(verify_test_identity(a,a))
        for model,threshold in [('model-b',1.),('model-a',2.)]:
            b=frozen_test_fingerprint(model,{'flip4':False,'threshold':threshold})
            with self.assertRaises(ValueError):verify_test_identity(a,b)
        b=frozen_test_fingerprint('model-a',{'flip4':False,'threshold':1.,'fusion':{'coefficient':[1.]}})
        with self.assertRaises(ValueError):verify_test_identity(a,b)
        b=frozen_test_fingerprint('model-a',{'flip4':False,'threshold':1.,'pixel_threshold':.3})
        with self.assertRaises(ValueError):verify_test_identity(a,b)

    def test_mask_statistics_and_exported_linear_decision(self):
        mask=torch.zeros(2,1,8,16);mask[0,0,0,0]=5
        features=mask_score_features(torch.tensor([1.,-2.]),mask)
        self.assertEqual(features.shape,(2,9));self.assertEqual(features[0,1].item(),5)
        self.assertAlmostEqual(features[0,7].item(),1/128)
        spec={'feature_names':MASK_FEATURE_NAMES,'mean':[0.]*9,'scale':[1.]*9,
              'coefficient':list(range(1,10)),'intercept':.2}
        expected=features.numpy().astype(np.float64)@np.arange(1,10)+.2
        np.testing.assert_allclose(fused_score(features,spec).numpy(),expected,rtol=1e-12)

    def test_exported_rbf_fusion_matches_fitted_sklearn(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        rng=np.random.default_rng(42);x=rng.normal(size=(70,9));y=(x[:,0]*x[:,1]>0).astype(int)
        learner=make_pipeline(StandardScaler(),SVC(C=1.7));learner.fit(x,y)
        scaler,decision=learner.steps[0][1],learner.steps[1][1]
        spec={'feature_names':MASK_FEATURE_NAMES,'method':'rbf_svm','mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
              'support_vectors':decision.support_vectors_.tolist(),'dual_coefficient':decision.dual_coef_[0].tolist(),
              'gamma':float(decision._gamma),'intercept':float(decision.intercept_[0])}
        unseen=rng.normal(size=(12,9));actual=fused_score(torch.from_numpy(unseen),spec).numpy()
        np.testing.assert_allclose(actual,learner.decision_function(unseen),rtol=1e-10,atol=1e-10)
        with self.assertRaises(ValueError):fused_score(torch.from_numpy(unseen),{**spec,'method':'unknown'})

    def test_deployment_and_qualification_score_paths_agree(self):
        class Corners(torch.nn.Module):
            def __init__(self):
                super().__init__();self.detector=torch.nn.Linear(3,1)
            def features(self,x):return x[:,:,0,0]
            def forward_joint(self,x):return x[:,:1],self.detector(self.features(x)).squeeze(1)
        model=Corners().eval();array=np.zeros((256,1600,3),dtype=np.uint8);array[:,:800]=150
        x=normalize_images(torch.from_numpy(array.transpose(2,0,1).copy())[None])
        spec={'feature_names':MASK_FEATURE_NAMES,'mean':[0.]*9,'scale':[1.]*9,'coefficient':[.1]*9,'intercept':.2}
        score,mask=inspect_image(model,array,'cpu',True,spec)
        expected=inspection_scores(model,x,{'flip4':True,'fusion':spec})
        self.assertAlmostEqual(score,expected.item(),places=6);self.assertEqual(mask.shape,(256,1600))
        spec={**spec,'feature_names':feature_names(False,True),'mean':[0.]*12,'scale':[1.]*12,'coefficient':[.1]*12}
        score,_=inspect_image(model,array,'cpu',True,spec)
        expected=inspection_scores(model,x,{'flip4':True,'fusion':spec})
        self.assertAlmostEqual(score,expected.item(),places=6)

    def test_pyramid_preserves_native_window_positions(self):
        class Widths(torch.nn.Module):
            def features(self,x):
                assert x.shape[-1]==800
                return x[:,0,0,:1]
            def detector(self,x):return x
        x=torch.arange(1600.).view(1,1,1,1600)
        torch.testing.assert_close(pyramid_logits(Widths(),x),torch.tensor([[800.,400.,0.]]))
        with self.assertRaises(ValueError):pyramid_logits(Widths(),x[...,:800])

    def test_semantic_pyramid_batch_and_single_image_decisions_agree(self):
        class SemanticCorners(torch.nn.Module):
            def __init__(self):
                super().__init__();self.detector=torch.nn.Linear(3,1);self.auxiliary=torch.nn.Linear(3,4)
            def features(self,x):return x[:,:,0,0]
            def forward_joint(self,x,return_aux=False):
                p=self.features(x);out=(x[:,:1],self.detector(p).squeeze(1))
                return (*out,self.auxiliary(p)) if return_aux else out
        model=SemanticCorners().eval();array=np.zeros((256,1600,3),dtype=np.uint8)
        array[:64,:400]=255;array[-64:,800:]=120
        names=feature_names(True,True);n=len(names)
        self.assertEqual(n,16)
        spec={'feature_names':names,'mean':[.2]*n,'scale':[1.3]*n,'coefficient':[i/10 for i in range(n)],'intercept':-.3}
        score,mask=inspect_image(model,array,'cpu',True,spec)
        x=normalize_images(torch.from_numpy(array.transpose(2,0,1).copy())[None]).repeat(2,1,1,1)
        batch=inspection_scores(model,x,{'flip4':True,'fusion':spec})
        np.testing.assert_allclose(batch.numpy(),[score,score],rtol=1e-6,atol=1e-6)
        self.assertEqual(mask.shape,(256,1600))

    def test_guided_crop_keeps_foreground_and_image_mask_pairing(self):
        image=torch.arange(16,dtype=torch.uint8).view(1,1,1,16).expand(3,3,8,16).clone()
        mask=torch.zeros(3,1,8,16,dtype=torch.uint8);mask[0,0,3,0]=1;mask[1,0,4,15]=1
        x,y=guided_crop(image,mask,8,random.Random(42))
        self.assertEqual(x.shape,(3,3,8,8));self.assertTrue(y[0].any());self.assertTrue(y[1].any());self.assertFalse(y[2].any())
        for i in range(3):
            start=int(x[i,0,0,0]);torch.testing.assert_close(y[i],mask[i,...,start:start+8])

    def test_auxiliary_model_roundtrip_and_gradients(self):
        model=StripInspector(auxiliary_classes=True)
        mask,cls,aux=model.forward_joint(torch.randn(2,3,64,128),return_aux=True)
        self.assertEqual(aux.shape,(2,4));aux.square().mean().backward()
        self.assertGreater(model.auxiliary[-1].weight.grad.abs().sum().item(),0)
        restored=inspector_from_saved({'model':model.state_dict()})
        self.assertTrue(hasattr(restored,'auxiliary'))
        torch.testing.assert_close(restored.auxiliary[-1].weight,model.auxiliary[-1].weight)

    def test_ensemble_reload_and_refined_mask_preservation(self):
        model=StripEnsemble([StripInspector(),StripInspector(auxiliary_classes=True)]).eval()
        restored=inspector_from_saved({'model_kind':'strip_ensemble','model':model.state_dict()}).eval()
        names=ensemble_feature_names(MASK_FEATURE_NAMES,feature_names(True,False));x=torch.randn(2,3,32,64)
        features,mask=decision_features(restored,x,False,names)
        self.assertEqual(features.shape,(2,22))
        with torch.inference_mode():expected_mask=restored.members[1].forward_joint(x)[0]
        torch.testing.assert_close(mask,expected_mask)
        spec={'feature_names':names,'mean':[0.]*22,'scale':[1.]*22,'coefficient':[.1]*22,'intercept':.3}
        actual=inspection_scores(restored,x,{'flip4':False,'fusion':spec})
        torch.testing.assert_close(actual,.1*features.double().sum(1)+.3)
        invalid={**model.state_dict(),'unclaimed.weight':torch.ones(1)}
        with self.assertRaises(ValueError):inspector_from_saved({'model_kind':'strip_ensemble','model':invalid})

    def test_efficient_joint_shapes_gradients_and_implicit_reload(self):
        model=EfficientStripInspector().eval();x=torch.randn(2,3,64,128)
        mask,logit,aux=model.forward_joint(x,return_aux=True)
        self.assertEqual(mask.shape,(2,1,64,128));self.assertEqual(aux.shape,(2,4))
        self.assertEqual(model.features(x).shape,(2,3008))
        target=torch.zeros_like(mask);target[0,:,:,20:24]=1
        loss,_=strip_loss(mask,logit,target);loss=loss+.2*aux.square().mean();loss.backward()
        self.assertTrue(torch.isfinite(model.encoder[0][0].weight.grad).all())
        self.assertGreater(model.encoder[0][0].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.out.weight.grad.abs().sum().item(),0)
        restored=inspector_from_saved({'model':model.state_dict()}).eval()
        with torch.inference_mode():actual=restored.forward_joint(x,return_aux=True)
        for a,b in zip(actual,(mask,logit,aux)):torch.testing.assert_close(a,b)

    def test_efficient_pretraining_checksum_rejects_nonofficial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.pth';path.write_bytes(b'not official pretrained weights')
            with self.assertRaises(ValueError):EfficientStripInspector(path)
