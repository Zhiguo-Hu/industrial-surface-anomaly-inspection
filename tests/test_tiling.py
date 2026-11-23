import unittest
import numpy as np
from src.tiling import starts, boxes, crop_tile, Stitcher


class TilingTests(unittest.TestCase):
    def test_long_strip_and_anchored_edge(self):
        xs=starts(1600)
        self.assertEqual(len(xs),12)
        self.assertEqual(xs[-2:],[1280,1344])

    def test_reconstruction_including_padding_and_irregular_shapes(self):
        rng=np.random.default_rng(42)
        for height,width in [(256,1600),(71,93),(333,517),(256,256)]:
            image=rng.integers(0,256,(height,width),dtype=np.uint8)
            stitch=Stitcher(height,width)
            for box in boxes(height,width):stitch.add(crop_tile(image,box),box)
            np.testing.assert_array_equal(stitch.finish(),image)
            mask=np.zeros((height,width),dtype=np.uint8);mask[-1,-1]=1
            stitch=Stitcher(height,width)
            for box in boxes(height,width):stitch.add(crop_tile(mask,box,is_mask=True),box)
            np.testing.assert_array_equal(stitch.finish(),mask)

    def test_overlap_averages_and_missing_tiles_fail(self):
        stitch=Stitcher(2,3)
        stitch.add(np.ones((2,2)),(0,0,2,2))
        with self.assertRaises(ValueError):stitch.finish()
        stitch.add(np.full((2,2),3),(1,0,3,2))
        np.testing.assert_array_equal(stitch.finish(),[[1,2,3],[1,2,3]])

    def test_invalid_stride_and_nonfinite_maps(self):
        for stride in [0,257]:
            with self.assertRaises(ValueError):starts(1600,stride=stride)
        with self.assertRaises(ValueError):Stitcher(1,1).add([[float('nan')]],(0,0,1,1))
