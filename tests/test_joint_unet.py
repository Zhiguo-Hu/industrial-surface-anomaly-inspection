import unittest
import torch
from src.resnet_unet import ResNet18UNet,JointResNet18UNet
from src.unet import segmentation_loss


class JointUNetTests(unittest.TestCase):
    def test_segmentation_initialization_and_forward_are_preserved(self):
        torch.set_num_threads(2)
        torch.manual_seed(42);original=ResNet18UNet().eval()
        torch.manual_seed(42);joint=JointResNet18UNet().eval()
        for k,v in original.state_dict().items():torch.testing.assert_close(v,joint.state_dict()[k],rtol=0,atol=0)
        x=torch.randn(2,3,65,97)
        with torch.no_grad():
            mask,cls=joint.forward_joint(x)
            torch.testing.assert_close(mask,original(x),rtol=0,atol=0)
            self.assertEqual(cls.shape,(2,))

    def test_joint_loss_backpropagates_to_both_branches(self):
        model=JointResNet18UNet();x=torch.randn(2,3,64,64)
        y=torch.zeros(2,1,64,64);y[0,:,10:20,10:20]=1
        mask,cls=model.forward_joint(x)
        loss=segmentation_loss(mask,y)+.5*torch.nn.functional.binary_cross_entropy_with_logits(cls,y.flatten(1).amax(1))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.classifier[-1].weight.grad.abs().sum().item(),0)

    def test_zero_classification_weight_gives_zero_head_gradient(self):
        model=JointResNet18UNet();x=torch.randn(2,3,64,64)
        mask,cls=model.forward_joint(x);y=torch.zeros_like(mask)
        loss=segmentation_loss(mask,y)+0*torch.nn.functional.binary_cross_entropy_with_logits(cls,torch.ones(2))
        loss.backward()
        self.assertTrue(all(p.grad is not None and p.grad.count_nonzero()==0 for p in model.classifier.parameters()))
