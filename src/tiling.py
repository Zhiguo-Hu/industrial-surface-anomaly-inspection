"""Lossless spatial tiling and overlap averaging; coordinates are x,y,x2,y2."""
import numpy as np


def starts(length, size=256, stride=128):
    if length < 1 or size < 1 or not 1 <= stride <= size:
        raise ValueError('Require positive dimensions and 1 <= stride <= size')
    if length <= size:
        return [0]
    positions = list(range(0, length-size+1, stride))
    if positions[-1] != length-size:
        positions.append(length-size)
    return positions


def boxes(height, width, size=256, stride=128):
    return [(x,y,min(x+size,width),min(y+size,height))
            for y in starts(height,size,stride) for x in starts(width,size,stride)]


def crop_tile(array, box, size=256, is_mask=False):
    x,y,x2,y2 = box
    if not (0 <= x < x2 <= array.shape[1] and 0 <= y < y2 <= array.shape[0]):
        raise ValueError('Invalid crop coordinates')
    tile = array[y:y2,x:x2]
    if tile.shape[0]>size or tile.shape[1]>size:
        raise ValueError('Crop larger than tile size')
    padding = [(0,size-tile.shape[0]),(0,size-tile.shape[1])]+[(0,0)]*(tile.ndim-2)
    # RGB padding repeats the edge; masks use zero. Padded pixels are discarded on stitching.
    return np.pad(tile,padding,mode='constant' if is_mask else 'edge')


class Stitcher:
    def __init__(self, height, width):
        if min(height,width)<1:
            raise ValueError('Invalid output shape')
        self.total = np.zeros((height,width),dtype=np.float64)
        self.count = np.zeros((height,width),dtype=np.uint32)

    def add(self, tile_map, box):
        x,y,x2,y2 = box
        if not (0 <= x < x2 <= self.total.shape[1] and 0 <= y < y2 <= self.total.shape[0]):
            raise ValueError('Invalid stitch coordinates')
        tile_map=np.asarray(tile_map)
        if tile_map.ndim!=2 or tile_map.shape[0]<y2-y or tile_map.shape[1]<x2-x:
            raise ValueError('Expected full-resolution 2-D tile map')
        valid=tile_map[:y2-y,:x2-x]
        if not np.isfinite(valid).all():
            raise ValueError('Non-finite tile map')
        self.total[y:y2,x:x2]+=valid
        self.count[y:y2,x:x2]+=1

    def finish(self):
        if np.any(self.count==0):
            raise ValueError('Uncovered pixels; missing tiles')
        return (self.total/self.count).astype(np.float32)
