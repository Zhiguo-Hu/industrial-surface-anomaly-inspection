import unittest
from src.training_selection import coverage_balanced_rows


class SelectionTests(unittest.TestCase):
    def test_budget_coverage_normal_identity_determinism_and_no_leakage(self):
        normals=[dict(image_id=f'n{i}',label=0,split='train',classes=[]) for i in range(250)]
        positives=[dict(image_id=f'c{c}_{i}',label=1,split='train',classes=[c]) for c in (1,2,3,4) for i in range(300)]
        heldout=[dict(image_id=f'v{i}',label=1,split='val',classes=[2]) for i in range(100)]
        baseline=normals+positives[:250];all_rows=normals+positives+heldout
        output=coverage_balanced_rows(all_rows,baseline)
        self.assertEqual(len(output),500)
        self.assertEqual(len({r['image_id'] for r in output}),500)
        self.assertEqual(output,coverage_balanced_rows(all_rows[::-1],baseline))
        self.assertEqual({r['image_id'] for r in output if not r['label']},{r['image_id'] for r in normals})
        self.assertTrue(all(r['split']=='train' for r in output))
        for c in (1,2,3,4):self.assertGreaterEqual(sum(c in r['classes'] for r in output),60)

    def test_impossible_quota_rejected(self):
        with self.assertRaises(ValueError):coverage_balanced_rows([],[])
