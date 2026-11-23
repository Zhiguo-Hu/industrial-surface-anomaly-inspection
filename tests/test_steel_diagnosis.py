import unittest
import numpy as np
from scripts.diagnose_steel import summarize_map


class DiagnosisTests(unittest.TestCase):
    def test_peak_coordinates_annotation_and_interior(self):
        heat=np.ones((64,64),dtype=np.float32)
        heat[2,5]=9;heat[32,32]=4
        mask=np.zeros_like(heat,dtype=np.uint8);mask[32,32]=1
        d=summarize_map(heat,mask)
        self.assertEqual((d['peak_x'],d['peak_y']),(5,2))
        self.assertTrue(d['peak_outer_border16'])
        self.assertFalse(d['peak_in_annotation'])
        self.assertEqual(d['interior_peak'],4)
        self.assertEqual(d['defect_max'],4)
        self.assertEqual(d['background_max'],9)

    def test_background_only_has_no_defect_metric(self):
        heat=np.ones((64,64),dtype=np.float32)
        d=summarize_map(heat,np.zeros_like(heat,dtype=np.uint8))
        self.assertNotIn('pixel_auroc',d)
        self.assertEqual(d['contrast'],0)
        self.assertEqual(d['ratio'],1)
