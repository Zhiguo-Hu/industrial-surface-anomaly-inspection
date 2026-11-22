import copy
import unittest

import numpy as np
import torch

from src.baseline import nearest_scores, validate_manifest, validation_threshold


class BaselineTests(unittest.TestCase):
    def test_known_distances_and_chunk_equivalence(self):
        bank = torch.tensor([[0., 0.], [3., 4.]])
        query = torch.tensor([[0., 0.], [3., 0.], [3., 4.]])
        scores, indices = nearest_scores(query, bank, 1)
        np.testing.assert_allclose(scores, [0, 3, 0])
        np.testing.assert_array_equal(indices, [0, 0, 1])
        np.testing.assert_array_equal(scores, nearest_scores(query, bank, 64)[0])

    def test_known_quantile(self):
        self.assertAlmostEqual(validation_threshold([0., 1., 2., 3., 4.]), 3.8)

    def test_empty_and_nonfinite_rejected(self):
        for scores in [[], [float('nan')]]:
            with self.assertRaises(ValueError):
                validation_threshold(scores)

    def test_manifest_leakage_and_abnormal_training_rejected(self):
        rows = [dict(image=str(i), sha256=str(i), split=s, label=0,
                     official_split='test' if s=='test' else 'train')
                for i,s in enumerate(['train','val','test'])]
        validate_manifest(rows)
        duplicate = copy.deepcopy(rows)
        duplicate[1]['sha256'] = duplicate[0]['sha256']
        with self.assertRaises(ValueError):
            validate_manifest(duplicate)
        rows[0]['label'] = 1
        with self.assertRaises(ValueError):
            validate_manifest(rows)


if __name__ == '__main__':
    unittest.main()
