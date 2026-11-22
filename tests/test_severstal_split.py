import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('splitter',Path(__file__).resolve().parents[1]/'scripts/split_severstal.py')
splitter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitter)


class SplitTests(unittest.TestCase):
    def test_allocation_preserves_totals_including_tiny_strata(self):
        for n in range(101):
            counts=splitter.allocate(n)
            self.assertEqual(sum(counts),n)
            self.assertTrue(all(c>=0 for c in counts))
        self.assertEqual(splitter.allocate(100),[70,15,15])

    def test_partition_determinism_and_bank_eligibility(self):
        rows=[{'image_id':str(i),'classes':[] if i<20 else [1,3]} for i in range(40)]
        output=splitter.assign(rows)
        self.assertEqual(output,splitter.assign(rows[::-1]))
        self.assertEqual(len({r['image_id'] for r in output}),40)
        self.assertEqual({s:sum(r['split']==s for r in output) for s in splitter.SPLITS},
                         {'train':28,'val':6,'test':6})
        for row in output:
            self.assertEqual(row['use_for_memory_bank'],row['split']=='train' and not row['classes'])


if __name__=='__main__':unittest.main()
