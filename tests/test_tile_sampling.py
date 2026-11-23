import random,unittest
from src.tile_sampling import ClassAwareSampler,SMALL_MAX_PIXELS


class TileSamplingTests(unittest.TestCase):
    def fixture(self):
        classes={0:[],1:[1],2:[1],3:[2],4:[2],5:[3],6:[3],7:[4],8:[4],9:[1,2]}
        pixels={i:(0 if not cs else (100 if i%2 else 10000)) for i,cs in classes.items()}
        return classes,pixels

    def test_class_coverage_positive_only_and_replay(self):
        classes,pixels=self.fixture()
        a=ClassAwareSampler(classes,pixels,'class-balanced',42)
        b=ClassAwareSampler(classes,pixels,'class-balanced',42)
        for _ in range(100):
            draw=a.draw();self.assertEqual(draw,b.draw())
            self.assertTrue(all(c in classes[i] and pixels[i]>0 for c,i in enumerate(draw,1)))

    def test_small_bias_and_no_small_fallback(self):
        classes,pixels=self.fixture();pixels[7]=10000  # Class 4 has no small tiles.
        a=ClassAwareSampler(classes,pixels,'class-balanced',42)
        b=ClassAwareSampler(classes,pixels,'class-small',42,minimum_small_tiles=1,minimum_small_parents=1,tiles_per_parent=1)
        first=sum(pixels[i]<=SMALL_MAX_PIXELS for _ in range(2000) for i in a.draw())
        second=sum(pixels[i]<=SMALL_MAX_PIXELS for _ in range(2000) for i in b.draw())
        self.assertGreater(second,first+700)
        self.assertEqual(b.small[4],[])
        self.assertTrue(all(4 in classes[b.draw()[3]] for _ in range(20)))

    def test_tiny_or_concentrated_small_pools_fall_back(self):
        classes,pixels=self.fixture()
        a=ClassAwareSampler(classes,pixels,'class-balanced',42)
        b=ClassAwareSampler(classes,pixels,'class-small',42)
        self.assertFalse(any(b.eligible.values()))
        self.assertEqual([a.draw() for _ in range(100)],[b.draw() for _ in range(100)])
        classes={i:[i//20+1] for i in range(80)};pixels={i:100 for i in classes}
        concentrated=ClassAwareSampler(classes,pixels,'class-small',42)
        self.assertFalse(any(concentrated.eligible.values()))
        spread=ClassAwareSampler(classes,pixels,'class-small',42,tiles_per_parent=1)
        self.assertTrue(all(spread.eligible.values()))

    def test_invalid_labels_and_missing_class_are_rejected(self):
        classes,pixels=self.fixture();pixels[1]=0
        with self.assertRaises(ValueError):ClassAwareSampler(classes,pixels,'class-small',42)
        classes,pixels=self.fixture();classes[7]=classes[8]=[];pixels[7]=pixels[8]=0
        with self.assertRaises(ValueError):ClassAwareSampler(classes,pixels,'class-small',42)

    def test_sampler_does_not_change_legacy_rng_stream(self):
        classes,pixels=self.fixture();rng=random.Random(42);state=rng.getstate()
        sampler=ClassAwareSampler(classes,pixels,'class-small',42);sampler.draw()
        self.assertEqual(state,rng.getstate())
