import unittest
import numpy as np
from src.region_scoring import clean_mask,region_scores


class RegionTests(unittest.TestCase):
    def test_diagonal_connected_and_small_region_removed(self):
        p=np.zeros((5,5));p[0,0]=.8;p[1,1]=.9;p[4,4]=1
        mask=clean_mask(p,.5,2)
        self.assertEqual(mask.sum(),2);self.assertTrue(mask[1,1]);self.assertFalse(mask[4,4])

    def test_area_and_component_probability_mass(self):
        p=np.zeros((4,4));p[0,:2]=.8;p[3,3]=.9
        scores,_=region_scores(p,.5,0)
        self.assertAlmostEqual(scores['mask_area_fraction'],3/16)
        self.assertAlmostEqual(scores['largest_component_probability_mass'],1.6/16)
        self.assertEqual(scores['top_0.1pct_mean'],.9)

    def test_empty_and_invalid_probability(self):
        scores,mask=region_scores(np.zeros((4,4)),.5,64)
        self.assertFalse(mask.any());self.assertTrue(all(x==0 for x in scores.values()))
        for p in [np.array([[np.nan]]),np.array([[2.]])]:
            with self.assertRaises(ValueError):clean_mask(p,.5,16)
