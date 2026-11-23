import unittest
from src.steel import pilot_selection


class SteelPilotTests(unittest.TestCase):
    def test_selection_disjoint_deterministic_normal_only_and_no_test(self):
        rows=[]
        for split,label,n in [('train',0,400),('train',1,200),('val',0,400),('val',1,200),('test',0,100)]:
            rows += [dict(image_id=f'{split}-{label}-{i}',split=split,label=label) for i in range(n)]
        result=pilot_selection(rows)
        self.assertEqual(result,pilot_selection(rows[::-1]))
        self.assertEqual([len(result[k]) for k in ('train','calibration','evaluation')],[300,200,200])
        flat=[r['image_id'] for group in result.values() for r in group]
        self.assertEqual(len(flat),len(set(flat)))
        self.assertTrue(all(r['label']==0 for r in result['train']+result['calibration']))
        self.assertTrue(all(r['split']!='test' for group in result.values() for r in group))
        self.assertEqual(sum(r['label'] for r in result['evaluation']),100)
