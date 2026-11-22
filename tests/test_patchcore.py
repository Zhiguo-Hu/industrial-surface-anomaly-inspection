import unittest
import torch
from src.patchcore import anomaly_maps, coreset_indices, patch_distances


class PatchTests(unittest.TestCase):
    def test_chunked_distance_matches_reference(self):
        torch.manual_seed(2)
        q,b=torch.randn(11,4),torch.randn(7,4)
        torch.testing.assert_close(patch_distances(q,b,3,2),torch.cdist(q,b).min(1).values)

    def test_coreset_unique_repeatable_including_duplicate_vectors(self):
        x=torch.zeros(10,4)
        a=coreset_indices(x,5)
        self.assertEqual(len(a.unique()),5)
        torch.testing.assert_close(a,coreset_indices(x,5))

    def test_map_order_and_constant_smoothing(self):
        s,m=anomaly_maps(torch.tensor([1.,1.,1.,1.,3.,3.,3.,3.]),2,2,2,16)
        torch.testing.assert_close(s,torch.tensor([1.,3.]))
        torch.testing.assert_close(m[0],torch.ones(16,16))
        torch.testing.assert_close(m[1],torch.full((16,16),3.))


if __name__=='__main__': unittest.main()
