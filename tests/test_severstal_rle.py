import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[1]/'scripts/audit_severstal.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RLETests(unittest.TestCase):
    def test_column_major_and_one_based(self):
        expected=np.array([[1,0,0],[1,0,1]],dtype=np.uint8)
        np.testing.assert_array_equal(audit.decode_rle('1 2 6 1',2,3),expected)
        self.assertEqual(audit.encode_rle(expected),'1 2 6 1')

    def test_invalid_runs_rejected(self):
        for text in ['1','0 1','6 2','2 0','2 3 3 2']:
            with self.assertRaises(ValueError):audit.decode_rle(text,2,3)

    def test_empty_mask(self):
        self.assertEqual(audit.decode_rle('',2,3).sum(),0)


if __name__=='__main__':unittest.main()
