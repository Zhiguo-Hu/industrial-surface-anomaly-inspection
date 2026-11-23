import unittest
import torch
from pathlib import Path
from src.resnet_unet import ResNet18UNet


class ResNetUNetTests(unittest.TestCase):
    def test_random_encoder_shape_and_gradient(self):
        torch.set_num_threads(2)
        model=ResNet18UNet();x=torch.randn(2,3,65,97)
        y=model(x);self.assertEqual(y.shape,(2,1,65,97))
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_pretrained_encoder_and_identical_decoder_initialization(self):
        weights=Path(__file__).resolve().parents[1]/'weights/resnet18-f37072fd.pth'
        if not weights.is_file():self.skipTest('Official weights are not redistributed')
        torch.manual_seed(42);random=ResNet18UNet()
        torch.manual_seed(42);pretrained=ResNet18UNet(weights)
        for key,value in random.state_dict().items():
            if not key.startswith('encoder.'):
                torch.testing.assert_close(value,pretrained.state_dict()[key],rtol=0,atol=0)
        reference=torch.load(weights,map_location='cpu',weights_only=True)
        torch.testing.assert_close(pretrained.encoder.conv1.weight,reference['conv1.weight'],rtol=0,atol=0)
        self.assertFalse(torch.equal(random.encoder.conv1.weight,pretrained.encoder.conv1.weight))
