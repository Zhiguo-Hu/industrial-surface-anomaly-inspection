import unittest
import torch
from src.unet import SmallUNet,segmentation_loss


class UNetTests(unittest.TestCase):
    def test_output_shape_and_gradients_on_odd_size(self):
        torch.set_num_threads(2)
        model=SmallUNet();x=torch.randn(2,3,33,49)
        output=model(x);self.assertEqual(output.shape,(2,1,33,49))
        loss=segmentation_loss(output,torch.zeros_like(output));loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_loss_rewards_correct_foreground_and_background(self):
        mask=torch.zeros(2,1,16,16);mask[0,:,4:8,4:8]=1
        good=(2*mask-1)*10
        self.assertLess(segmentation_loss(good,mask),segmentation_loss(-good,mask))
