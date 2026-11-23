import unittest
import numpy as np
from src.segmentation_diagnostics import mask_statistics,area_group,summarize


class SegmentationDiagnosticsTests(unittest.TestCase):
    def test_perfect_empty_and_wrong_masks(self):
        m=np.array([[1,0],[0,0]])
        self.assertEqual(mask_statistics(m,m)['dice'],1)
        wrong=mask_statistics(1-m,m)
        self.assertEqual(wrong['dice'],0);self.assertEqual(wrong['fp_pixels'],3)
        self.assertIsNone(mask_statistics(np.zeros((2,2)),np.zeros((2,2)))['dice'])

    def test_area_boundaries(self):
        self.assertEqual(area_group(0),'normal')
        self.assertEqual(area_group(.001),'0.1_to_1pct')
        self.assertEqual(area_group(.05),'at_least_5pct')

    def test_pooled_metrics_include_normal_false_positive_pixels(self):
        positive={**mask_statistics([[1,0]],[[1,0]]),'label':1,'alarm':1}
        normal={**mask_statistics([[1,0]],[[0,0]]),'label':0,'alarm':0}
        result=summarize([positive,normal])
        self.assertEqual(result['mean_defective_image_dice'],1)
        self.assertAlmostEqual(result['pooled_union_dice'],2/3)
